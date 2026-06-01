#!/usr/bin/env python3
import argparse
import base64
import hashlib
import heapq
import itertools
import json
import math
import os
import gc
import sys
import threading
import time
import uuid
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# This module is launched as a script (`python v2/llm_worker.py`), so the repo
# root is not on sys.path by default. Add it so the shared `v2.signing` module
# imports the same way it does for the coworld game.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2 import signing


DEFAULT_FLOW_CKPT = "checkpoints/flas-gemma-2-9b-it/flas-gemma-2-9b-it.safetensors"
DEFAULT_MODEL_ID = "unsloth/gemma-2-9b-it"
DEFAULT_MAX_BATCH_SIZE = 4
DEFAULT_MAX_PROMPT_TOKENS = 1024
DEFAULT_SCORING_MAX_PROMPT_TOKENS = 512
DEFAULT_MAX_GENERATION_TOKENS = 128
BENCH_TRIALS = 3

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Lower rank is served first. Tournament-signed requests preempt the unsigned
# public traffic, which is always accepted but never allowed to starve a
# tournament episode.
PRIORITY_TOURNAMENT = 0
PRIORITY_NORMAL = 1


class RequestFuture:
    def __init__(self, op, payload, priority=PRIORITY_NORMAL):
        self.op = op
        self.payload = payload
        self.priority = priority
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.enqueued_at = time.perf_counter()

    def resolve(self, result):
        self.result = result
        self.event.set()

    def reject(self, error):
        self.error = error
        self.event.set()

    def wait(self):
        self.event.wait()
        if self.error is not None:
            raise self.error
        return self.result


class PriorityBatchScheduler:
    """Serves requests highest-priority-first, FIFO within a priority.

    Ordering uses a heap keyed by ``(priority, sequence)`` so tournament-signed
    requests jump ahead of unsigned public traffic. Once the highest-priority
    request is chosen, it is batched with other compatible *waiting* requests
    regardless of their priority, because batching only changes how requests are
    grouped for the GPU, not which one runs next.
    """

    def __init__(self, model, max_batch_size=DEFAULT_MAX_BATCH_SIZE):
        self.model = model
        self.max_batch_size = max_batch_size
        self.cv = threading.Condition()
        self.ready = []  # heap of (priority, sequence, future)
        self.sequence = 0
        self.thread = threading.Thread(target=self.run, name="llm-worker-scheduler", daemon=True)
        self.thread.start()

    def submit(self, op, payload, priority=PRIORITY_NORMAL):
        fut = RequestFuture(op, payload, priority)
        with self.cv:
            heapq.heappush(self.ready, (priority, self.sequence, fut))
            self.sequence += 1
            self.cv.notify()
        return fut

    def queue_depth(self):
        with self.cv:
            return len(self.ready)

    def run(self):
        while True:
            with self.cv:
                while not self.ready:
                    self.cv.wait()
                first = heapq.heappop(self.ready)[2]
                key = compatibility_key(first.op, first.payload)
                batch = [first]
                if key is not None:
                    kept = []
                    while self.ready and len(batch) < self.max_batch_size:
                        candidate_entry = heapq.heappop(self.ready)
                        candidate = candidate_entry[2]
                        if compatibility_key(candidate.op, candidate.payload) == key:
                            batch.append(candidate)
                        else:
                            kept.append(candidate_entry)
                    for entry in kept:
                        heapq.heappush(self.ready, entry)

            try:
                results = self.model.run_batch(first.op, [item.payload for item in batch])
                for fut, result in zip(batch, results):
                    fut.resolve(result)
            except Exception as exc:
                for fut in batch:
                    fut.reject(exc)


def compatibility_key(op, payload):
    concept_key = stable_digest(payload.get("concept", {"type": "text", "text": ""}))
    flas = payload.get("flas", {})
    steps = int(flas.get("steps", 3))
    if op == "generate":
        sampling = payload.get("sampling", {})
        return (
            op,
            concept_key,
            steps,
            int(sampling.get("max_tokens", DEFAULT_MAX_GENERATION_TOKENS)),
            float(sampling.get("temperature", 0.7)),
        )
    if op == "choice-logprobs":
        ordering = payload.get("ordering", {})
        return (op, concept_key, steps, ordering.get("mode", "given_order"))
    return None


def stable_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class FlasModel:
    def __init__(self, flow_ckpt, model_id, layer=None, num_blocks=None):
        self.flow_ckpt = flow_ckpt
        self.model_id = model_id
        self.layer = layer
        self.num_blocks = num_blocks
        self.gen = None
        self.lock = threading.Lock()
        self.loaded_at = None

    def health(self):
        device = None
        cuda_available = False
        peak_vram_mb = None
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device = torch.cuda.get_device_name(0)
                peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
                allocated_vram_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
                reserved_vram_mb = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
            else:
                allocated_vram_mb = None
                reserved_vram_mb = None
        except Exception:
            allocated_vram_mb = None
            reserved_vram_mb = None
            pass
        return {
            "ok": True,
            "loaded": self.gen is not None,
            "model_id": self.model_id if self.gen is not None else None,
            "device": device,
            "cuda_available": cuda_available,
            "peak_vram_mb": peak_vram_mb,
            "allocated_vram_mb": allocated_vram_mb,
            "reserved_vram_mb": reserved_vram_mb,
        }

    def load(self, payload=None):
        payload = payload or {}
        with self.lock:
            if self.gen is not None:
                return {"message": "Model already loaded.", **self.health()}
            from flas.generate import load_generator

            self.flow_ckpt = payload.get("flow_ckpt", self.flow_ckpt)
            self.model_id = payload.get("model_id", self.model_id)
            self.layer = payload.get("layer", self.layer)
            self.num_blocks = payload.get("num_blocks", self.num_blocks)
            self.gen = load_generator(
                self.flow_ckpt,
                model_id=self.model_id,
                layer=self.layer,
                num_blocks=self.num_blocks,
            )
            self.loaded_at = time.time()
            return {"message": "Model loaded.", **self.health()}

    def ensure_loaded(self):
        if self.gen is None:
            self.load()

    def run_batch(self, op, payloads):
        self.ensure_loaded()
        with self.lock:
            try:
                self.cleanup_request_state()
                if op == "generate":
                    return self.generate_batch(payloads)
                if op == "choice-logprobs":
                    return [self.choice_logprobs(payload) for payload in payloads]
                raise ValueError(f"Unsupported operation: {op}")
            finally:
                self.cleanup_request_state()

    def cleanup_request_state(self):
        if self.gen is not None:
            self.gen._remove_hook()
            self.gen._active = False
            self.gen._n_steps = 0
            self.gen._concept_hidden = None
            self.gen._concept_mask = None
            self.gen._flowtimes = None
            self.gen._padding_mask = None
            self.gen._sa_caches = None
            self.gen._is_prefill = True
            self.gen._past_len = 0
            self.gen._position_ids = None
        try:
            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def resolve_concept(self, concept):
        import torch

        concept = concept or {"type": "text", "text": ""}
        concept_type = concept.get("type", "text")
        if concept_type == "text":
            return self.gen.encode_concept(str(concept.get("text", "")))
        if concept_type == "vector":
            return self.vector_concept(concept)
        if concept_type == "random":
            return self.random_concept(concept)
        raise ValueError(f"Unsupported concept type: {concept_type}")

    def vector_concept(self, concept):
        import torch

        if "hidden_b64" in concept:
            hidden = decode_float_tensor(concept["hidden_b64"], concept.get("shape"), concept.get("dtype", "float32"))
        else:
            hidden = torch.tensor(concept["hidden"], dtype=torch.float32)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3:
            raise ValueError("Vector concept hidden must have shape [tokens, hidden_dim] or [1, tokens, hidden_dim].")
        expected_dim = self.hidden_size()
        if hidden.shape[-1] != expected_dim:
            raise ValueError(f"Vector concept hidden_dim {hidden.shape[-1]} does not match model hidden_dim {expected_dim}.")
        mask = concept.get("mask")
        if mask is None:
            mask_t = torch.ones((hidden.shape[0], hidden.shape[1]), dtype=torch.float32)
        else:
            mask_t = torch.tensor(mask, dtype=torch.float32)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
        if mask_t.shape != hidden.shape[:2]:
            raise ValueError("Vector concept mask must have shape [tokens] or [1, tokens].")
        return hidden.to("cuda").to(self.gen._flow_dtype), mask_t.to("cuda")

    def random_concept(self, concept):
        import torch

        tokens = int(concept.get("tokens", 64))
        if tokens <= 0:
            raise ValueError("Random concept tokens must be positive.")
        scale = float(concept.get("scale", 1.0))
        distribution = concept.get("distribution", "normal")
        if distribution not in {"normal", "rademacher"}:
            raise ValueError("Random concept distribution must be 'normal' or 'rademacher'.")
        seed_int = seed_to_int(str(concept.get("seed", "")))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_int)
        hidden_dim = self.hidden_size()
        if distribution == "normal":
            hidden = torch.randn((1, tokens, hidden_dim), generator=generator, dtype=torch.float32)
        else:
            bits = torch.randint(0, 2, (1, tokens, hidden_dim), generator=generator, dtype=torch.int8)
            hidden = bits.float().mul_(2.0).sub_(1.0)
        norm = concept.get("normalize", "unit_rms")
        if norm == "unit_rms":
            rms = hidden.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            hidden = hidden / rms
        elif norm in {None, "none"}:
            pass
        else:
            raise ValueError("Random concept normalize must be 'unit_rms' or 'none'.")
        hidden = hidden * scale
        mask = torch.ones((1, tokens), dtype=torch.float32)
        return hidden.to("cuda").to(self.gen._flow_dtype), mask.to("cuda")

    def hidden_size(self):
        return int(self.gen.llm.config.hidden_size)

    def generate_batch(self, payloads):
        if not payloads:
            return []
        concept_hidden, concept_mask = self.resolve_concept(payloads[0].get("concept"))
        prompts = [str(payload.get("prompt", "")) for payload in payloads]
        flowtimes = [float(payload.get("flas", {}).get("flowtime", 2.0)) for payload in payloads]
        steps = int(payloads[0].get("flas", {}).get("steps", 3))
        sampling = payloads[0].get("sampling", {})
        started = time.perf_counter()
        outputs = self.generate_batch_with_concept(
            prompts=prompts,
            concept_hidden=concept_hidden,
            concept_mask=concept_mask,
            flowtimes=flowtimes,
            n_steps=steps,
            max_tokens=int(sampling.get("max_tokens", DEFAULT_MAX_GENERATION_TOKENS)),
            max_prompt_tokens=int(sampling.get("max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS)),
            temperature=float(sampling.get("temperature", 0.7)),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return [
            {
                "id": payload.get("id"),
                "text": output["generation"],
                "finish_reason": "eos" if output.get("eos") else "length",
                "input_tokens": output["input_tokens"],
                "output_tokens": output["output_tokens"],
                "latency_ms": round(elapsed_ms, 3),
                "batch_size": len(payloads),
            }
            for payload, output in zip(payloads, outputs)
        ]

    def generate_batch_with_concept(self, prompts, concept_hidden, concept_mask, flowtimes, n_steps, max_tokens, max_prompt_tokens, temperature):
        import torch

        gen = self.gen
        out = None
        past_kv = None
        generated = None
        next_logits = None
        input_ids = None
        attention_mask = None
        enc = None
        eos = None
        unfinished = None
        probs = None
        next_token = None
        position_ids = None
        gen._n_steps = n_steps
        formatted = [
            gen.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        bsz = len(formatted)
        enc = gen.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
            add_special_tokens=False,
        ).to("cuda")
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask
        prompt_len = input_ids.shape[1]
        gen._concept_hidden = concept_hidden.expand(bsz, -1, -1).contiguous()
        gen._concept_mask = concept_mask.expand(bsz, -1).contiguous()
        gen._flowtimes = torch.tensor(flowtimes, device="cuda", dtype=torch.float32)
        gen._padding_mask = attention_mask.float()
        gen._sa_caches = [None] * n_steps
        gen._is_prefill = True
        gen._past_len = 0
        gen._position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        gen._install_hook()
        gen._active = True
        eos = torch.zeros(bsz, dtype=torch.bool, device="cuda")
        try:
            with torch.inference_mode():
                out = gen.llm(input_ids, attention_mask=attention_mask, position_ids=gen._position_ids, use_cache=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]
            gen._is_prefill = False
            gen._past_len = prompt_len
            generated = input_ids
            unfinished = torch.ones(bsz, dtype=torch.bool, device="cuda")
            for _ in range(max_tokens):
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                next_token = next_token.masked_fill(~unfinished.unsqueeze(1), gen.tokenizer.pad_token_id)
                generated = torch.cat([generated, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, unfinished.unsqueeze(1).long()], dim=1)
                eos_hit = next_token.squeeze(1) == gen.tokenizer.eos_token_id
                eos = eos | eos_hit
                unfinished = unfinished & ~eos_hit
                if not unfinished.any():
                    break
                gen._padding_mask = attention_mask.float()
                position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
                gen._position_ids = position_ids[:, -1:]
                with torch.inference_mode():
                    out = gen.llm(
                        next_token,
                        attention_mask=attention_mask,
                        position_ids=gen._position_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]
                gen._past_len += 1
            results = []
            for idx in range(bsz):
                gen_ids = generated[idx, prompt_len:]
                text = gen.tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append(
                    {
                        "generation": text,
                        "input_tokens": int(attention_mask[idx, :prompt_len].sum().item()),
                        "output_tokens": int((gen_ids != gen.tokenizer.pad_token_id).sum().item()),
                        "eos": bool(eos[idx].item()),
                    }
                )
            return results
        finally:
            gen._active = False
            del out, past_kv, generated, next_logits, input_ids, attention_mask, enc, eos, unfinished, probs, next_token, position_ids

    def choice_logprobs(self, payload):
        choices = payload.get("choices", [])
        if not choices:
            raise ValueError("choices must be non-empty.")
        if any(not isinstance(choice, str) or not choice.strip() for choice in choices):
            raise ValueError("choices must be non-empty strings.")
        mode = payload.get("ordering", {}).get("mode", "given_order")
        if mode == "given_order":
            orderings = [tuple(range(len(choices)))]
        elif mode == "all_permutations":
            orderings = list(itertools.permutations(range(len(choices))))
        else:
            raise ValueError("ordering.mode must be 'given_order' or 'all_permutations'.")
        concept_hidden, concept_mask = self.resolve_concept(payload.get("concept"))
        flowtime = float(payload.get("flas", {}).get("flowtime", 2.0))
        steps = int(payload.get("flas", {}).get("steps", 3))
        max_prompt_tokens = int(payload.get("max_prompt_tokens", DEFAULT_SCORING_MAX_PROMPT_TOKENS))
        sums = [0.0] * len(choices)
        per_ordering = []
        for ordering in orderings:
            ordered_choices = [choices[idx] for idx in ordering]
            prompt = render_choice_prompt(str(payload.get("prompt", "")), ordered_choices)
            probs = self.choice_probs_for_order(prompt, ordered_choices, concept_hidden, concept_mask, flowtime, steps, max_prompt_tokens)
            mapped = [0.0] * len(choices)
            for local_idx, original_idx in enumerate(ordering):
                mapped[original_idx] = probs[local_idx]
                sums[original_idx] += probs[local_idx]
            per_ordering.append({"ordering": list(ordering), "probabilities": mapped})
        averaged = [value / len(orderings) for value in sums]
        return {
            "id": payload.get("id"),
            "probabilities": averaged,
            "orderings": per_ordering,
        }

    def choice_probs_for_order(self, prompt, choices, concept_hidden, concept_mask, flowtime, steps, max_prompt_tokens):
        token_ids = [self.gen.tokenizer.encode(choice.strip(), add_special_tokens=False) for choice in choices]
        prefixes = distinguishing_prefixes(token_ids)
        raw = self.sequence_next_probs(prompt, concept_hidden, concept_mask, flowtime, steps, prefixes, max_prompt_tokens)
        total = sum(raw)
        if total <= 0:
            return [1.0 / len(raw)] * len(raw)
        return [value / total for value in raw]

    def sequence_next_probs(self, prompt, concept_hidden, concept_mask, flowtime, steps, token_sequences, max_prompt_tokens):
        import torch

        gen = self.gen
        gen._n_steps = steps
        out = None
        input_ids = None
        attention_mask = None
        dist = None
        enc = None
        try:
            formatted = gen.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            results = []
            for sequence in token_sequences:
                prefix_ids = list(sequence[:-1])
                target_id = sequence[-1]
                enc = gen.tokenizer(
                    [formatted],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_prompt_tokens,
                    add_special_tokens=False,
                ).to("cuda")
                input_ids = enc.input_ids
                if prefix_ids:
                    input_ids = torch.cat([input_ids, torch.tensor([prefix_ids], device="cuda", dtype=input_ids.dtype)], dim=1)
                attention_mask = torch.ones_like(input_ids)
                gen._concept_hidden = concept_hidden
                gen._concept_mask = concept_mask
                gen._flowtimes = torch.tensor([flowtime], device="cuda", dtype=torch.float32)
                gen._padding_mask = attention_mask.float()
                gen._sa_caches = [None] * steps
                gen._is_prefill = True
                gen._past_len = 0
                gen._position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
                gen._install_hook()
                gen._active = True
                try:
                    with torch.inference_mode():
                        out = gen.llm(input_ids, attention_mask=attention_mask, position_ids=gen._position_ids, use_cache=True)
                    dist = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
                    results.append(float(dist[target_id].item()))
                finally:
                    gen._active = False
                    del out, input_ids, attention_mask, dist, enc
            return results
        finally:
            gen._active = False
            gen._remove_hook()

    def microbatch_bench(self, payload):
        import torch

        self.ensure_loaded()
        batch_sizes = [int(size) for size in payload.get("batch_sizes", [1, 2, 4])]
        prompt_tokens = int(payload.get("prompt_tokens", 128))
        max_tokens = int(payload.get("max_tokens", 32))
        concept = payload.get("concept", {"type": "random", "seed": "bench", "tokens": 64})
        flas = payload.get("flas", {"flowtime": 2.0, "steps": 3})
        results = []
        for batch_size in batch_sizes:
            latencies = []
            token_rates = []
            status = "ok"
            error = None
            for trial in range(BENCH_TRIALS):
                torch.cuda.reset_peak_memory_stats()
                prompts = [synthetic_prompt(prompt_tokens, i, trial) for i in range(batch_size)]
                reqs = [
                    {
                        "prompt": prompt,
                        "concept": concept,
                        "flas": flas,
                        "sampling": {"max_tokens": max_tokens, "temperature": 0.0},
                    }
                    for prompt in prompts
                ]
                started = time.perf_counter()
                try:
                    outs = self.generate_batch(reqs)
                    elapsed = time.perf_counter() - started
                    total_output = sum(item["output_tokens"] for item in outs)
                    latencies.append(elapsed * 1000)
                    token_rates.append(total_output / elapsed if elapsed > 0 else 0.0)
                    del outs
                    torch.cuda.empty_cache()
                except RuntimeError as exc:
                    status = "error"
                    error = str(exc)
                    if "out of memory" in error.lower():
                        status = "oom"
                        torch.cuda.empty_cache()
                    break
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            results.append(
                {
                    "batch_size": batch_size,
                    "status": status,
                    "error": error,
                    "latency_ms": summarize(latencies),
                    "tokens_per_sec": summarize(token_rates),
                    "peak_vram_mb": peak_vram_mb,
                }
            )
        return {"results": results}


def decode_float_tensor(encoded, shape, dtype):
    import torch

    if shape is None:
        raise ValueError("Base64 vector concepts require shape.")
    raw = base64.b64decode(encoded)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype)
    if torch_dtype is None:
        raise ValueError("Vector dtype must be float16, bfloat16, or float32.")
    tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype)
    return tensor.reshape(shape).float()


def seed_to_int(seed):
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def render_choice_prompt(prompt, choices):
    lines = [prompt.rstrip(), "", "Valid answers, one per line:"]
    for choice in choices:
        lines.append(choice)
    lines.append("")
    lines.append("Answer with the exact choice text.")
    lines.append("Answer:")
    return "\n".join(lines)


def distinguishing_prefixes(token_ids):
    prefixes = []
    for idx, ids in enumerate(token_ids):
        if not ids:
            raise ValueError("Choice tokenized to zero tokens.")
        chosen = None
        for end in range(1, len(ids) + 1):
            prefix = tuple(ids[:end])
            if all(prefix != tuple(other[:end]) for j, other in enumerate(token_ids) if j != idx):
                chosen = prefix
                break
        if chosen is None:
            chosen = tuple(ids)
        prefixes.append(chosen)
    return prefixes


def synthetic_prompt(prompt_tokens, item_idx, trial_idx):
    words = [f"token{(i + item_idx + trial_idx) % 997}" for i in range(prompt_tokens)]
    return "Summarize this synthetic record in one sentence:\n" + " ".join(words)


def summarize(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "p50": round(percentile(ordered, 0.50), 3),
        "p90": round(percentile(ordered, 0.90), 3),
        "min": round(min(ordered), 3),
        "max": round(max(ordered), 3),
    }


def percentile(ordered, q):
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


class WorkerServer:
    def __init__(self, model, max_batch_size):
        self.model = model
        self.scheduler = PriorityBatchScheduler(model, max_batch_size=max_batch_size)
        # Public key is not a secret; absent key means signing is disabled and
        # all traffic runs at normal priority.
        self.public_key = signing.resolve_public_key()

    def request_priority(self, headers, body):
        """Tournament priority for a valid, fresh signature; normal otherwise.

        Unsigned or badly-signed requests are NOT rejected: the worker serves
        everyone. A bad signature simply forfeits the priority bump.
        """
        if self.public_key is None:
            return PRIORITY_NORMAL
        timestamp_raw = headers.get(signing.TIMESTAMP_HEADER)
        signature = headers.get(signing.SIGNATURE_HEADER)
        if not timestamp_raw or not signature:
            return PRIORITY_NORMAL
        try:
            timestamp = int(timestamp_raw)
        except ValueError:
            return PRIORITY_NORMAL
        if signing.verify_request(self.public_key, timestamp, signature, body, now=int(time.time())):
            return PRIORITY_TOURNAMENT
        return PRIORITY_NORMAL

    def submit_many(self, op, payload, priority=PRIORITY_NORMAL):
        requests = payload.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("Payload must include a non-empty requests list.")
        futures = []
        for request in requests:
            if not isinstance(request, dict):
                raise ValueError("Each request must be an object.")
            request.setdefault("id", str(uuid.uuid4()))
            futures.append(self.scheduler.submit(op, request, priority))
        return {"results": [future.wait() for future in futures]}


def make_handler(worker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def do_GET(self):
            if self.path == "/health":
                payload = worker.model.health()
                payload["queue_depth"] = worker.scheduler.queue_depth()
                self.write_json(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                body = self.read_body()
                payload = json.loads(body.decode("utf-8")) if body else {}
                if self.path == "/load":
                    self.write_json(worker.model.load(payload))
                    return
                if self.path == "/generate":
                    priority = worker.request_priority(self.headers, body)
                    self.write_json(worker.submit_many("generate", payload, priority))
                    return
                if self.path == "/choice-logprobs":
                    priority = worker.request_priority(self.headers, body)
                    self.write_json(worker.submit_many("choice-logprobs", payload, priority))
                    return
                if self.path == "/bench/microbatch":
                    self.write_json(worker.model.microbatch_bench(payload))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def read_body(self):
            length = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(length) if length else b""

        def write_json(self, payload, status=HTTPStatus.OK):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Run the v2 local FLAS/Gemma worker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--flow-ckpt", default=DEFAULT_FLOW_CKPT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=DEFAULT_MAX_BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    model = FlasModel(args.flow_ckpt, args.model_id, layer=args.layer, num_blocks=args.num_blocks)
    worker = WorkerServer(model, max_batch_size=args.max_batch_size)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(worker))
    print(f"Serving v2 FLAS worker at http://{args.host}:{args.port}", flush=True)
    print("Use Ctrl-C to stop.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
