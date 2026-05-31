#!/usr/bin/env python3
import argparse
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from app import (
    BEAT_BONUS_POINTS,
    DUPLICATE_ANSWER_PENALTY_POINTS,
    GameState,
    SCORE_SCALE,
    STEERING_FLOWTIME,
    STEERING_STEPS,
    STEERING_TEMPERATURE,
    make_handler,
)


class WorkerClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            try:
                payload = json.loads(body)
                raise RuntimeError(payload.get("error", body)) from exc
            except json.JSONDecodeError:
                raise RuntimeError(body) from exc


class WorkerBackedBackend:
    def __init__(self, args):
        self.args = args
        self.state = GameState()
        self.worker = WorkerClient(args.worker_url)
        self.model_lock = threading.Lock()

    def load(self):
        data = self.worker.post("/load", {})
        return data.get("message", "Worker model loaded.")

    @staticmethod
    def model_safe_text(text):
        replacements = {
            r"\balice\b": "entry one",
            r"\bbob\b": "entry two",
            r"\bcharlie\b": "entry three",
            r"\bplayer\b": "entry",
            r"\bplayers\b": "entries",
            r"\bopponent\b": "alternate entry",
            r"\bopponents\b": "alternate entries",
        }
        safe = str(text)
        for pattern, replacement in replacements.items():
            safe = re.sub(pattern, replacement, safe, flags=re.IGNORECASE)
        return safe

    def ask_charlie(self, payload):
        role = self.validate_role(payload.get("role"))
        question = payload.get("question", "").strip()
        if not question:
            raise ValueError("Question is required.")
        with self.state.lock:
            if len(self.state.players[role]["charlie"]) >= 3:
                raise ValueError("This endpoint already used all 3 Charlie questions.")
            concept = self.state.hidden_concept
        safe_question = self.model_safe_text(question)
        prompt = (
            "Answer the question directly and helpfully.\n\n"
            f"Question: {safe_question}"
        )
        result = self.worker.post(
            "/generate",
            {
                "requests": [
                    {
                        "prompt": prompt,
                        "concept": {"type": "text", "text": concept},
                        "flas": {"flowtime": STEERING_FLOWTIME, "steps": STEERING_STEPS},
                        "sampling": {"max_tokens": 100, "temperature": STEERING_TEMPERATURE},
                    }
                ]
            },
        )
        answer = result["results"][0]["text"]
        with self.state.lock:
            self.state.players[role]["charlie"].append({"question": question, "answer": answer})
            return self.state.view(role)

    def submit_proposals(self, payload):
        role = self.validate_role(payload.get("role"))
        proposals = payload.get("proposals", [])
        if len(proposals) != 3:
            raise ValueError("Submit exactly 3 questions and answers.")
        cleaned = []
        for proposal in proposals:
            question = proposal.get("question", "").strip()
            answer = proposal.get("answer", "").strip()
            if not question or not answer:
                raise ValueError("Every proposed question and hidden answer must be non-empty.")
            cleaned.append({"question": question, "answer": answer})
        with self.state.lock:
            if self.state.phase() != "proposals":
                raise ValueError("Both players must ask 3 Charlie questions before phase 2 begins.")
            self.state.players[role]["proposals"] = cleaned
            return self.state.view(role)

    def submit_answers(self, payload):
        role = self.validate_role(payload.get("role"))
        answers = [str(answer).strip() for answer in payload.get("answers", [])]
        if len(answers) != 3 or any(not answer for answer in answers):
            raise ValueError("Submit exactly 3 non-empty blind answers.")
        with self.state.lock:
            if self.state.phase() != "blind_answers":
                raise ValueError("Both players must submit proposed questions before phase 3 begins.")
            self.state.players[role]["answers"] = answers
            should_score = all(len(player["answers"]) == 3 for player in self.state.players.values())
        if should_score:
            self.score_round()
        with self.state.lock:
            return self.state.view(role)

    def score_round(self):
        with self.state.lock:
            alice = json.loads(json.dumps(self.state.players["alice"]))
            bob = json.loads(json.dumps(self.state.players["bob"]))
            hidden_concept = self.state.hidden_concept

        context = self.scoring_context(alice, bob)
        rows = []
        alice_points = 0.0
        bob_points = 0.0
        with self.model_lock:
            for idx, proposal in enumerate(alice["proposals"]):
                score = self.answer_score(context, proposal["question"], proposal["answer"], bob["answers"][idx], hidden_concept)
                alice_points += score["secret_score_points"]
                bob_points += score["opponent_score_points"]
                rows.append({
                    "submitter": "Alice",
                    "owner": "Alice",
                    "question": proposal["question"],
                    "secret_answer": proposal["answer"],
                    "opponent_answer": bob["answers"][idx],
                    **score,
                })
            for idx, proposal in enumerate(bob["proposals"]):
                score = self.answer_score(context, proposal["question"], proposal["answer"], alice["answers"][idx], hidden_concept)
                bob_points += score["secret_score_points"]
                alice_points += score["opponent_score_points"]
                rows.append({
                    "submitter": "Bob",
                    "owner": "Bob",
                    "question": proposal["question"],
                    "secret_answer": proposal["answer"],
                    "opponent_answer": alice["answers"][idx],
                    **score,
                })

        results = {
            "rows": rows,
            "totals": {
                "alice_points": alice_points,
                "bob_points": bob_points,
                "alice_margin": alice_points,
                "bob_margin": bob_points,
                "alice_difference": alice_points - bob_points,
                "bob_difference": bob_points - alice_points,
            },
            "hidden_concept": hidden_concept,
            "settings": {
                "flowtime": STEERING_FLOWTIME,
                "steps": STEERING_STEPS,
                "temperature": STEERING_TEMPERATURE,
            },
        }
        with self.state.lock:
            self.state.results = results

    def scoring_context(self, alice, bob):
        def transcript(section, player):
            turns = []
            for idx, turn in enumerate(player["charlie"]):
                question = self.model_safe_text(turn["question"])
                answer = self.model_safe_text(turn["answer"])
                turns.append(f"Record {section}.{idx + 1} question: {question}\nRecord {section}.{idx + 1} answer: {answer}")
            return "\n\n".join(turns)

        public_questions = []
        for idx, proposal in enumerate(alice["proposals"]):
            public_questions.append(f"Question group 1.{idx + 1}: {self.model_safe_text(proposal['question'])}")
        for idx, proposal in enumerate(bob["proposals"]):
            public_questions.append(f"Question group 2.{idx + 1}: {self.model_safe_text(proposal['question'])}")

        return "\n\n".join([
            "Reference material:",
            transcript("1", alice),
            transcript("2", bob),
            "Question list:",
            "\n".join(public_questions),
        ])

    def answer_score(self, context, question, secret_answer, opponent_answer, concept):
        conflict = self.answer_conflict(secret_answer, opponent_answer)
        if conflict is not None:
            duplicate_answer_count = len([secret_answer, opponent_answer])
            shared_probability = 1.0 / duplicate_answer_count
            secret_base_points = SCORE_SCALE * shared_probability
            opponent_base_points = SCORE_SCALE * shared_probability
            secret_duplicate_penalty_points = -DUPLICATE_ANSWER_PENALTY_POINTS
            opponent_duplicate_penalty_points = -DUPLICATE_ANSWER_PENALTY_POINTS
            secret_score_points = secret_base_points + secret_duplicate_penalty_points
            opponent_score_points = opponent_base_points + opponent_duplicate_penalty_points
            return {
                "score_points": secret_score_points,
                "secret_score_points": secret_score_points,
                "opponent_score_points": opponent_score_points,
                "base_points": secret_base_points,
                "secret_base_points": secret_base_points,
                "opponent_base_points": opponent_base_points,
                "bonus_points": 0.0,
                "secret_bonus_points": 0.0,
                "opponent_bonus_points": 0.0,
                "duplicate_penalty_points": secret_duplicate_penalty_points,
                "secret_duplicate_penalty_points": secret_duplicate_penalty_points,
                "opponent_duplicate_penalty_points": opponent_duplicate_penalty_points,
                "score_margin": 0.0,
                "average_secret_probability": shared_probability,
                "average_opponent_probability": shared_probability,
                "duplicate_conflict": True,
                "canonical_answer": conflict,
                "orderings": [],
            }
        if not opponent_answer:
            secret_base_points = SCORE_SCALE
            secret_bonus_points = BEAT_BONUS_POINTS
            return {
                "score_points": secret_base_points + secret_bonus_points,
                "secret_score_points": secret_base_points + secret_bonus_points,
                "opponent_score_points": 0.0,
                "base_points": secret_base_points,
                "secret_base_points": secret_base_points,
                "opponent_base_points": 0.0,
                "bonus_points": secret_bonus_points,
                "secret_bonus_points": secret_bonus_points,
                "opponent_bonus_points": 0.0,
                "score_margin": 1.0,
                "average_secret_probability": 1.0,
                "average_opponent_probability": 0.0,
                "duplicate_conflict": False,
                "orderings": [],
            }
        safe_question = self.model_safe_text(question)
        safe_secret_answer = self.model_safe_text(secret_answer)
        safe_opponent_answer = self.model_safe_text(opponent_answer)
        first = self.option_selection_probs(
            context,
            safe_question,
            safe_secret_answer,
            safe_opponent_answer,
            concept,
            reverse=False,
        )
        second = self.option_selection_probs(
            context,
            safe_question,
            safe_secret_answer,
            safe_opponent_answer,
            concept,
            reverse=True,
        )
        first_margin = first["secret_probability"] - first["opponent_probability"]
        second_margin = second["secret_probability"] - second["opponent_probability"]
        average_secret_probability = (first["secret_probability"] + second["secret_probability"]) / 2
        average_opponent_probability = (first["opponent_probability"] + second["opponent_probability"]) / 2
        secret_base_points = SCORE_SCALE * average_secret_probability
        opponent_base_points = SCORE_SCALE * average_opponent_probability
        secret_bonus_points = BEAT_BONUS_POINTS if average_secret_probability > average_opponent_probability else 0.0
        opponent_bonus_points = BEAT_BONUS_POINTS if average_opponent_probability > average_secret_probability else 0.0
        return {
            "score_points": secret_base_points + secret_bonus_points,
            "secret_score_points": secret_base_points + secret_bonus_points,
            "opponent_score_points": opponent_base_points + opponent_bonus_points,
            "base_points": secret_base_points,
            "secret_base_points": secret_base_points,
            "opponent_base_points": opponent_base_points,
            "bonus_points": secret_bonus_points,
            "secret_bonus_points": secret_bonus_points,
            "opponent_bonus_points": opponent_bonus_points,
            "score_margin": (first_margin + second_margin) / 2,
            "average_secret_probability": average_secret_probability,
            "average_opponent_probability": average_opponent_probability,
            "duplicate_conflict": False,
            "orderings": [first, second],
        }

    def answer_conflict(self, first, second):
        first_normalized = self.normalize_answer_for_conflict(first)
        second_normalized = self.normalize_answer_for_conflict(second)
        if not first_normalized or not second_normalized:
            return None
        if first_normalized.startswith(second_normalized) or second_normalized.startswith(first_normalized):
            return first if len(first_normalized) <= len(second_normalized) else second
        return None

    def normalize_answer_for_conflict(self, answer):
        return " ".join(str(answer).strip().casefold().split())

    def option_selection_probs(self, context, question, secret_answer, opponent_answer, concept, reverse):
        choices = [opponent_answer, secret_answer] if reverse else [secret_answer, opponent_answer]
        prompt = (
            f"{context}\n\n"
            "Choose the answer that best answers the question based on the available evidence.\n"
            f"Question: {question}"
        )
        result = self.worker.post(
            "/choice-logprobs",
            {
                "requests": [
                    {
                        "prompt": prompt,
                        "concept": {"type": "text", "text": concept},
                        "flas": {"flowtime": STEERING_FLOWTIME, "steps": STEERING_STEPS},
                        "choices": choices,
                        "ordering": {"mode": "given_order"},
                    }
                ]
            },
        )
        probs = result["results"][0]["probabilities"]
        secret_probability = probs[1] if reverse else probs[0]
        opponent_probability = probs[0] if reverse else probs[1]
        return {
            "order": "opponent_first" if reverse else "secret_first",
            "secret_probability": secret_probability,
            "opponent_probability": opponent_probability,
        }

    def validate_role(self, role):
        if role not in {"alice", "bob"}:
            raise ValueError("Role must be alice or bob.")
        return role


def parse_args():
    parser = argparse.ArgumentParser(description="Run the v2 steering game server backed by an LLM worker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--worker-url", default="http://127.0.0.1:7870")
    return parser.parse_args()


def main():
    args = parse_args()
    backend = WorkerBackedBackend(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    print(f"Serving v2 steering game at http://{args.host}:{args.port}", flush=True)
    print(f"Using LLM worker at {args.worker_url}", flush=True)
    print("Use Ctrl-C to stop.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
