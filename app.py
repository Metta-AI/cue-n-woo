import argparse
import json
import os
import random
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


STEERING_FLOWTIME = 2.0
STEERING_STEPS = 3
STEERING_TEMPERATURE = 0.7

STEERING_CONCEPTS = [
    "terse technical documentation, precise API wording, implementation details",
    "warm supportive therapist, emotionally validating, gentle reflective language",
    "skeptical scientific reviewer, caveats, methodology, uncertainty, evidence",
    "noir detective narration, smoky atmosphere, clipped cynicism, urban mystery",
    "exaggerated pirate speech, nautical slang, ahoy, matey, arr, treasure",
    "formal legal analysis, statutory language, liability, precedent, careful qualifications",
    "children's storybook narration, simple words, wonder, friendly tone",
    "academic philosophy, abstract concepts, epistemology, ontology, careful argument",
    "marketing copy, persuasive benefits, polished product language, call-to-action",
    "military field report, concise observations, tactical status, objective tone",
    "medieval fantasy chronicle, kingdoms, oaths, ancient prophecy, elevated diction",
    "cyberpunk hacker slang, neon cities, networks, implants, corporate dystopia",
    "stand-up comedy, punchlines, self-aware jokes, conversational timing",
    "Zen minimalist prose, calm restraint, sparse language, contemplative imagery",
    "financial analyst memo, market risk, valuation, forecasts, basis points",
    "Victorian letter writing, ornate politeness, nineteenth-century diction",
    "sports commentator excitement, play-by-play energy, momentum, dramatic calls",
    "conspiracy theorist speculation, hidden motives, suspicious connections",
    "cooking show host, sensory food language, ingredients, kitchen enthusiasm",
    "robotic bureaucratic formality, procedural compliance, sterile administrative tone",
    "gothic horror prose, dread, candlelight, decaying mansions, ominous suspense",
    "direct executive briefing, concise recommendations, tradeoffs, risks, next steps",
    "surreal dream logic, impossible imagery, shifting identities, poetic ambiguity",
    "hardboiled newspaper reporting, inverted pyramid, quotes, factual restraint",
    "Socratic tutor, probing questions, incremental reasoning, gentle correction",
    "emergency room triage, urgent prioritization, symptoms, risk assessment",
    "museum curator label, historical context, provenance, material culture",
    "field naturalist journal, species observations, habitat, patient description",
    "startup founder pitch, vision, traction, market opportunity, confident urgency",
    "ancient mythic epic, gods, heroes, fate, ritual grandeur",
    "minimalist haiku-like prose, seasonal imagery, silence, precise sensory detail",
    "dry British wit, understatement, irony, restrained comic timing",
    "optimistic futurist manifesto, technological progress, abundance, bold possibility",
    "pessimistic risk analyst, failure modes, downside scenarios, cautious forecasting",
    "friendly classroom teacher, clear examples, scaffolding, approachable explanations",
    "dense mathematical exposition, definitions, lemmas, notation, proof structure",
    "tabloid celebrity gossip, dramatic reveals, breathless speculation, scandal",
    "travel guidebook, practical tips, landmarks, local customs, itinerary language",
    "product manager spec, user stories, acceptance criteria, metrics, edge cases",
    "religious sermon, moral exhortation, scripture-like cadence, redemption themes",
    "diplomatic cable, geopolitical nuance, cautious phrasing, strategic interests",
    "forensic investigator report, evidence chain, timestamps, physical traces",
    "fitness coach motivation, reps, discipline, progress, energetic encouragement",
    "luxury brand copy, elegance, exclusivity, craftsmanship, refined aspiration",
    "old internet forum post, casual abbreviations, tangents, flamewar energy",
    "poetic romantic lyricism, longing, moonlight, intimate emotion, lush metaphor",
    "safety compliance manual, hazards, required procedures, warnings, checklists",
    "game designer commentary, mechanics, player incentives, balance, emergent play",
    "rural folk tale, plainspoken wisdom, animals, weather, generational memory",
    "cosmic science documentary, galaxies, deep time, awe, explanatory grandeur",
    "snarky tech reviewer, benchmarks, complaints, clever asides, practical verdicts",
    "meditation instructor, breath awareness, body scanning, compassionate presence",
    "courtroom cross-examination, leading questions, contradictions, evidentiary pressure",
    "archaeological expedition log, strata, artifacts, ruins, careful discovery",
    "radio host banter, conversational transitions, audience address, upbeat pacing",
    "policy think tank memo, stakeholders, incentives, implementation constraints",
    "classic adventure serial, cliffhangers, peril, bold explorers, exotic locales",
    "melancholic memoir, reflective memory, regret, intimate personal detail",
    "precision engineering note, tolerances, materials, failure analysis, specifications",
    "urban planning report, transit, zoning, density, public space, tradeoffs",
    "bardic tavern song, rhyming praise, revelry, legends, rhythmic flourish",
]


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Theory of Mind Steering Game</title>
  <style>
    :root {
      --bg: #f6f7f9; --panel: #fff; --ink: #17202a; --muted: #667085;
      --line: #d9dee7; --accent: #1f766b; --accent-dark: #155b52; --error: #b42318;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    main { min-height: 100vh; display: grid; grid-template-columns: 330px minmax(0, 1fr); }
    aside { padding: 18px; border-right: 1px solid var(--line); background: #fbfcfd; }
    section { padding: 22px; }
    h1 { margin: 0 0 12px; font-size: 20px; }
    h2 { margin: 20px 0 10px; color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }
    h3 { margin: 0 0 10px; font-size: 16px; }
    label { display: block; margin: 12px 0 6px; font-size: 13px; font-weight: 650; }
    input, textarea, button, select {
      width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff;
      color: var(--ink); font: inherit; padding: 9px 10px;
    }
    textarea { min-height: 78px; resize: vertical; line-height: 1.45; }
    button { cursor: pointer; background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 700; }
    button:hover { background: var(--accent-dark); }
    button.secondary { background: #fff; color: var(--ink); border-color: var(--line); }
    button.danger { background: #fff; color: var(--error); border-color: #f2b8b5; }
    button:disabled { cursor: not-allowed; opacity: .55; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .grid { max-width: 1240px; margin: 0 auto; display: grid; gap: 14px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .status { min-height: 22px; margin-top: 10px; font-size: 13px; color: var(--muted); white-space: pre-wrap; }
    .error { color: var(--error); }
    .small { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; margin-right: 6px; font-size: 12px; color: var(--muted); background: #fff; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 10px; }
    .metric strong { display: block; font-size: 20px; margin-top: 4px; }
    .callout { border-left: 4px solid var(--accent); background: #eef8f6; padding: 10px; border-radius: 6px; }
    .hidden { display: none; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid var(--line); } }
  </style>
</head>
<body>
<main>
  <aside>
    <h1 id="title">Theory of Mind Game</h1>
    <div>
      <a href="/alice">Alice endpoint</a><br>
      <a href="/bob">Bob endpoint</a>
    </div>
    <button id="loadBtn" class="secondary" style="margin-top:14px">Load model</button>
    <div id="loadStatus" class="status">Loads automatically when Charlie is asked or scoring starts.</div>

    <h2>Charlie Steering</h2>
    <div class="small">Charlie has a hidden randomly selected style/concept. It is revealed only at the end. Settings: flowtime 2, steps 3, temperature .7.</div>

    <h2>Admin</h2>
    <button id="resetBtn" class="danger">Reset Game</button>
  </aside>

  <section>
    <div class="grid">
      <div class="panel">
        <h3>Status</h3>
        <div id="phase"></div>
        <div id="progressCards" class="cards" style="margin-top:10px"></div>
        <div id="status" class="status"></div>
      </div>

      <div id="rootPanel" class="panel hidden">
        <h3>Choose an endpoint</h3>
        <p>Open Alice and Bob in separate browser tabs. Each endpoint hides the player's private answers from the other until reveal.</p>
      </div>

      <div id="transcriptPanel" class="panel hidden">
        <div id="askControls">
          <h3>Phase 1: Private Charlie Questions</h3>
          <div class="small">Ask up to three questions. Charlie answers each with 100 tokens.</div>
          <label for="charlieQuestion">Question for Charlie</label>
          <textarea id="charlieQuestion"></textarea>
          <button id="askBtn">Ask Charlie</button>
        </div>
        <h2>Your Private Transcript</h2>
        <pre id="transcript"></pre>
      </div>

      <div id="proposalPanel" class="panel hidden">
        <h3>Phase 2: Propose Three Questions</h3>
        <div class="small">Enter questions you know the answer to and expect the other player not to know. Your answers stay hidden until reveal.</div>
        <div id="proposalInputs"></div>
        <button id="submitProposalsBtn">Submit Questions + Hidden Answers</button>
      </div>

      <div id="answerPanel" class="panel hidden">
        <h3>Phase 3: Answer Opponent Questions</h3>
        <div class="small">You can see the opponent's questions, but not their hidden answers.</div>
        <div id="answerInputs"></div>
        <button id="submitAnswersBtn">Submit Blind Answers</button>
      </div>

      <div id="waitingPanel" class="panel hidden">
        <h3>Waiting</h3>
        <div id="waitingText" class="status"></div>
      </div>

      <div id="revealPanel" class="panel hidden">
        <h3>Reveal + Score</h3>
        <div id="conceptReveal" class="callout"></div>
        <div id="scoreCards" class="cards" style="margin:12px 0"></div>
        <table>
          <thead><tr><th>Owner</th><th>Question</th><th>Secret</th><th>Opponent Guess</th><th>Score Margin</th></tr></thead>
          <tbody id="scoreRows"></tbody>
        </table>
        <h2>Totals</h2>
        <pre id="totals"></pre>
      </div>

      <div class="panel">
        <h3>Round Overview</h3>
        <div id="overview"></div>
      </div>
    </div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const role = location.pathname.includes("bob") ? "bob" : (location.pathname.includes("alice") ? "alice" : "");
const opponent = role === "alice" ? "bob" : "alice";
const draftPrefix = `steering-game:${role || "root"}`;

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {})
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function getState() {
  const res = await fetch(`/api/state?role=${encodeURIComponent(role)}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function setStatus(text, isError=false) {
  $("status").textContent = text;
  $("status").className = isError ? "status error" : "status";
}

function readDraft(name) {
  try {
    return JSON.parse(localStorage.getItem(`${draftPrefix}:${name}`) || "null") || {};
  } catch {
    return {};
  }
}

function writeDraft(name, value) {
  localStorage.setItem(`${draftPrefix}:${name}`, JSON.stringify(value));
}

function clearDraft(name) {
  localStorage.removeItem(`${draftPrefix}:${name}`);
}

function htmlEscape(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function saveProposalDraft() {
  const next = {};
  for (let i = 0; i < 3; i++) {
    next[`q${i}`] = $(`proposalQ${i}`)?.value || "";
    next[`a${i}`] = $(`proposalA${i}`)?.value || "";
  }
  writeDraft("proposals", next);
}

function saveAnswerDraft() {
  const next = {};
  for (let i = 0; i < 3; i++) next[`a${i}`] = $(`blindA${i}`)?.value || "";
  writeDraft("answers", next);
}

function renderProposalInputs(state) {
  const root = $("proposalInputs");
  if (root.dataset.rendered === "true" && state.me.proposals.length === 0) return;
  const draft = readDraft("proposals");
  root.innerHTML = "";
  for (let i = 0; i < 3; i++) {
    const existing = state.me.proposals[i] || {};
    const question = existing.question || draft[`q${i}`] || "";
    const answer = existing.answer || draft[`a${i}`] || "";
    root.insertAdjacentHTML("beforeend", `
      <label>Question ${i + 1}</label>
      <textarea id="proposalQ${i}">${htmlEscape(question)}</textarea>
      <label>Hidden answer ${i + 1}</label>
      <input id="proposalA${i}" value="${htmlEscape(answer)}">
    `);
  }
  root.querySelectorAll("textarea, input").forEach((input) => {
    input.addEventListener("input", saveProposalDraft);
  });
  root.dataset.rendered = "true";
}

function renderAnswerInputs(state) {
  const root = $("answerInputs");
  if (root.dataset.rendered === "true" && state.me.answers.length === 0) return;
  const draft = readDraft("answers");
  root.innerHTML = "";
  for (let i = 0; i < 3; i++) {
    const q = state.opponent_questions[i]?.question || "";
    const answer = state.me.answers[i] || draft[`a${i}`] || "";
    root.insertAdjacentHTML("beforeend", `
      <label>${opponent} question ${i + 1}</label>
      <textarea disabled>${htmlEscape(q)}</textarea>
      <label>Your answer ${i + 1}</label>
      <input id="blindA${i}" value="${htmlEscape(answer)}">
    `);
  }
  root.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", saveAnswerDraft);
  });
  root.dataset.rendered = "true";
}

function renderScores(state) {
  $("scoreRows").innerHTML = "";
  for (const row of state.results?.rows || []) {
    const tr = document.createElement("tr");
    const margin = row.score_margin ?? ((row.probability ?? 0.5) * 2 - 1);
    for (const value of [row.owner, row.question, row.secret_answer, row.opponent_answer, margin.toFixed(4)]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    $("scoreRows").appendChild(tr);
  }
  const totals = state.results?.totals || {};
  $("totals").textContent = `Alice margin: ${(totals.alice_margin || totals.alice_difference || 0).toFixed(4)}\nBob margin: ${(totals.bob_margin || totals.bob_difference || 0).toFixed(4)}\nAlice - Bob: ${(totals.alice_difference || 0).toFixed(4)}`;
  $("conceptReveal").textContent = state.revealed_concept ? `Charlie's hidden steering concept: ${state.revealed_concept}` : "";
  $("scoreCards").innerHTML = state.results ? `
    <div class="metric">Alice margin<strong>${(totals.alice_margin || totals.alice_raw || 0).toFixed(3)}</strong></div>
    <div class="metric">Bob margin<strong>${(totals.bob_margin || totals.bob_raw || 0).toFixed(3)}</strong></div>
    <div class="metric">Alice - Bob<strong>${totals.alice_difference.toFixed(3)}</strong></div>
  ` : "";
}

function renderOverview(state) {
  const publicAlice = state.public_questions.alice.map((q, i) => `<li>A${i + 1}: ${q.question}</li>`).join("");
  const publicBob = state.public_questions.bob.map((q, i) => `<li>B${i + 1}: ${q.question}</li>`).join("");
  $("progressCards").innerHTML = `
    <div class="metric">Alice Charlie questions<strong>${state.counts.alice_chats}/3</strong></div>
    <div class="metric">Bob Charlie questions<strong>${state.counts.bob_chats}/3</strong></div>
    <div class="metric">Alice challenges<strong>${state.counts.alice_proposals}/3</strong></div>
    <div class="metric">Bob challenges<strong>${state.counts.bob_proposals}/3</strong></div>
    <div class="metric">Alice blind answers<strong>${state.counts.alice_answers}/3</strong></div>
    <div class="metric">Bob blind answers<strong>${state.counts.bob_answers}/3</strong></div>
  `;
  $("overview").innerHTML = `
    <div class="small">Charlie's steering concept is hidden until reveal.</div>
    <h2>Public Alice Questions</h2><ol>${publicAlice || "<li>Not submitted yet.</li>"}</ol>
    <h2>Public Bob Questions</h2><ol>${publicBob || "<li>Not submitted yet.</li>"}</ol>
  `;
}

function show(id, visible) {
  $(id).classList.toggle("hidden", !visible);
}

async function refresh() {
  const state = await getState();
  $("title").textContent = role ? `${role[0].toUpperCase() + role.slice(1)} Endpoint` : "Theory of Mind Game";
  $("phase").innerHTML = `<span class="pill">phase: ${state.phase}</span><span class="pill">Alice chats: ${state.counts.alice_chats}/3</span><span class="pill">Bob chats: ${state.counts.bob_chats}/3</span>`;
  renderOverview(state);

  show("rootPanel", !role);
  const canChat = !!role && state.phase === "private_questions" && state.me.charlie.length < 3;
  const canPropose = !!role && state.phase === "proposals" && state.me.proposals.length < 3;
  const canAnswer = !!role && state.phase === "blind_answers" && state.me.answers.length < 3;

  show("transcriptPanel", !!role);
  show("askControls", canChat);
  show("proposalPanel", canPropose);
  show("answerPanel", canAnswer);
  show("waitingPanel", !!role && state.phase !== "reveal" && !(canChat || canPropose || canAnswer));
  show("revealPanel", state.phase === "reveal");

  $("transcript").textContent = state.me.charlie.length
    ? state.me.charlie.map((turn, i) => `Q${i + 1}: ${turn.question}\nCharlie: ${turn.answer}`).join("\n\n")
    : "No private questions asked yet.";
  renderProposalInputs(state);
  renderAnswerInputs(state);
  renderScores(state);
  if (state.phase === "private_questions") {
    $("waitingText").textContent = "You have asked all 3 private Charlie questions. Waiting for the other player to finish phase 1.";
  } else if (state.phase === "proposals") {
    $("waitingText").textContent = "You have submitted all 3 challenge questions and hidden answers. Waiting for the other player to finish phase 2.";
  } else if (state.phase === "blind_answers") {
    $("waitingText").textContent = "You have submitted your blind answers. Waiting for the other player to finish phase 3.";
  } else {
    $("waitingText").textContent = "Waiting.";
  }
}

$("loadBtn").addEventListener("click", async () => {
  $("loadBtn").disabled = true;
  $("loadStatus").textContent = "Loading Gemma 2 9B IT + FLAS...";
  try {
    const data = await api("/api/load", {});
    $("loadStatus").textContent = data.message;
  } catch (err) {
    $("loadStatus").textContent = err.message;
    $("loadStatus").className = "status error";
  } finally {
    $("loadBtn").disabled = false;
  }
});

$("askBtn").addEventListener("click", async () => {
  $("askBtn").disabled = true;
  setStatus("Asking Charlie...");
  try {
    await api("/api/ask", {role, question: $("charlieQuestion").value});
    $("charlieQuestion").value = "";
    setStatus("Charlie answered.");
    await refresh();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    $("askBtn").disabled = false;
  }
});

$("submitProposalsBtn").addEventListener("click", async () => {
  saveProposalDraft();
  const proposals = [];
  for (let i = 0; i < 3; i++) {
    proposals.push({question: $(`proposalQ${i}`).value, answer: $(`proposalA${i}`).value});
  }
  try {
    await api("/api/propose", {role, proposals});
    clearDraft("proposals");
    setStatus("Questions submitted.");
    await refresh();
  } catch (err) {
    setStatus(err.message, true);
  }
});

$("submitAnswersBtn").addEventListener("click", async () => {
  saveAnswerDraft();
  const answers = [];
  for (let i = 0; i < 3; i++) answers.push($(`blindA${i}`).value);
  try {
    await api("/api/answer", {role, answers});
    clearDraft("answers");
    setStatus("Answers submitted.");
    await refresh();
  } catch (err) {
    setStatus(err.message, true);
  }
});

$("resetBtn").addEventListener("click", async () => {
  if (!confirm("Reset this in-memory game?")) return;
  await api("/api/reset", {});
  clearDraft("proposals");
  clearDraft("answers");
  $("proposalInputs").dataset.rendered = "false";
  $("answerInputs").dataset.rendered = "false";
  await refresh();
});

refresh().catch(err => setStatus(err.message, true));
setInterval(() => refresh().catch(() => {}), 4000);
</script>
</body>
</html>
"""


def empty_player():
    return {"charlie": [], "proposals": [], "answers": []}


class GameState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.players = {"alice": empty_player(), "bob": empty_player()}
        self.results = None
        self.hidden_concept = random.choice(STEERING_CONCEPTS)

    def phase(self):
        if self.results is not None:
            return "reveal"
        if any(len(player["charlie"]) < 3 for player in self.players.values()):
            return "private_questions"
        if any(len(player["proposals"]) < 3 for player in self.players.values()):
            return "proposals"
        if any(len(player["answers"]) < 3 for player in self.players.values()):
            return "blind_answers"
        return "ready_to_score"

    def view(self, role):
        if role not in self.players:
            role = "alice"
        other = "bob" if role == "alice" else "alice"
        return {
            "phase": self.phase(),
            "counts": {
                "alice_chats": len(self.players["alice"]["charlie"]),
                "bob_chats": len(self.players["bob"]["charlie"]),
                "alice_proposals": len(self.players["alice"]["proposals"]),
                "bob_proposals": len(self.players["bob"]["proposals"]),
                "alice_answers": len(self.players["alice"]["answers"]),
                "bob_answers": len(self.players["bob"]["answers"]),
            },
            "me": self.players[role],
            "opponent_questions": [
                {"question": proposal["question"]}
                for proposal in self.players[other]["proposals"]
            ],
            "public_questions": {
                "alice": [
                    {"question": proposal["question"]}
                    for proposal in self.players["alice"]["proposals"]
                ],
                "bob": [
                    {"question": proposal["question"]}
                    for proposal in self.players["bob"]["proposals"]
                ],
            },
            "results": self.results,
            "revealed_concept": self.hidden_concept if self.phase() == "reveal" else None,
        }


class FlasBackend:
    def __init__(self, args):
        self.args = args
        self.gen = None
        self.model_lock = threading.Lock()
        self.state = GameState()

    def load(self):
        with self.model_lock:
            if self.gen is not None:
                return "Model already loaded."
            from flas.generate import load_generator

            self.gen = load_generator(
                self.args.flow_ckpt,
                model_id=self.args.model_id,
                layer=self.args.layer,
                num_blocks=self.args.num_blocks,
            )
            return f"Loaded {self.args.model_id} with FLAS checkpoint {self.args.flow_ckpt}."

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
        if self.gen is None:
            self.load()
        safe_question = self.model_safe_text(question)
        prompt = (
            "Answer the question directly and helpfully.\n\n"
            f"Question: {safe_question}"
        )
        answer = self.generate_text(
            prompt=prompt,
            concept=self.state.hidden_concept,
            flowtime=STEERING_FLOWTIME,
            n_steps=STEERING_STEPS,
            max_tokens=100,
            temperature=STEERING_TEMPERATURE,
        )
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
                raise ValueError("Both players must ask Charlie 3 private questions before phase 2 begins.")
            self.state.players[role]["proposals"] = cleaned
            return self.state.view(role)

    def submit_answers(self, payload):
        role = self.validate_role(payload.get("role"))
        other = "bob" if role == "alice" else "alice"
        answers = [answer.strip() for answer in payload.get("answers", [])]
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
        if self.gen is None:
            self.load()
        with self.state.lock:
            alice = json.loads(json.dumps(self.state.players["alice"]))
            bob = json.loads(json.dumps(self.state.players["bob"]))
            hidden_concept = self.state.hidden_concept

        context = self.scoring_context(alice, bob)
        rows = []
        alice_margin = 0.0
        bob_margin = 0.0
        with self.model_lock:
            for idx, proposal in enumerate(alice["proposals"]):
                score = self.answer_score(
                    context=context,
                    question=proposal["question"],
                    secret_answer=proposal["answer"],
                    opponent_answer=bob["answers"][idx],
                    concept=hidden_concept,
                    flowtime=STEERING_FLOWTIME,
                    n_steps=STEERING_STEPS,
                )
                alice_margin += score["score_margin"]
                rows.append({
                    "owner": "Alice",
                    "question": proposal["question"],
                    "secret_answer": proposal["answer"],
                    "opponent_answer": bob["answers"][idx],
                    **score,
                })
            for idx, proposal in enumerate(bob["proposals"]):
                score = self.answer_score(
                    context=context,
                    question=proposal["question"],
                    secret_answer=proposal["answer"],
                    opponent_answer=alice["answers"][idx],
                    concept=hidden_concept,
                    flowtime=STEERING_FLOWTIME,
                    n_steps=STEERING_STEPS,
                )
                bob_margin += score["score_margin"]
                rows.append({
                    "owner": "Bob",
                    "question": proposal["question"],
                    "secret_answer": proposal["answer"],
                    "opponent_answer": alice["answers"][idx],
                    **score,
                })
        results = {
            "rows": rows,
            "totals": {
                "alice_margin": alice_margin,
                "bob_margin": bob_margin,
                "alice_difference": alice_margin - bob_margin,
                "bob_difference": bob_margin - alice_margin,
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

    def answer_score(self, context, question, secret_answer, opponent_answer, concept, flowtime, n_steps):
        safe_question = self.model_safe_text(question)
        safe_secret_answer = self.model_safe_text(secret_answer)
        safe_opponent_answer = self.model_safe_text(opponent_answer)
        first = self.option_selection_probs(
            context,
            safe_question,
            safe_secret_answer,
            safe_opponent_answer,
            concept,
            flowtime,
            n_steps,
            reverse=False,
        )
        second = self.option_selection_probs(
            context,
            safe_question,
            safe_secret_answer,
            safe_opponent_answer,
            concept,
            flowtime,
            n_steps,
            reverse=True,
        )
        first_margin = first["secret_probability"] - first["opponent_probability"]
        second_margin = second["secret_probability"] - second["opponent_probability"]
        return {
            "score_margin": (first_margin + second_margin) / 2,
            "average_secret_probability": (first["secret_probability"] + second["secret_probability"]) / 2,
            "orderings": [first, second],
        }

    def option_selection_probs(self, context, question, secret_answer, opponent_answer, concept, flowtime, n_steps, reverse):
        option_1, option_2 = (opponent_answer, secret_answer) if reverse else (secret_answer, opponent_answer)
        diff = self.first_differing_token_sequences(secret_answer, opponent_answer)
        if diff is None:
            return {
                "order": "opponent_first" if reverse else "secret_first",
                "secret_probability": 0.5,
                "opponent_probability": 0.5,
            }
        secret_seq, opponent_seq = diff
        prompt = (
            f"{context}\n\n"
            "Choose the answer that best answers the question based on the available evidence.\n"
            f"Question: {question}\n"
            f"Option 1: {option_1}\n"
            f"Option 2: {option_2}\n"
            "Write the selected answer exactly.\n"
            "Answer:"
        )
        probs = self.sequence_next_probs(prompt, concept, flowtime, n_steps, [secret_seq, opponent_seq])
        secret_p = probs[secret_seq]
        opponent_p = probs[opponent_seq]
        total = secret_p + opponent_p
        if total <= 0:
            normalized_secret = 0.5
            normalized_opponent = 0.5
        else:
            normalized_secret = secret_p / total
            normalized_opponent = opponent_p / total
        return {
            "order": "opponent_first" if reverse else "secret_first",
            "secret_probability": normalized_secret,
            "opponent_probability": normalized_opponent,
        }

    def first_differing_token_sequences(self, answer_a, answer_b):
        tokenizer = self.gen.tokenizer
        ids_a = tokenizer.encode(answer_a.strip(), add_special_tokens=False)
        ids_b = tokenizer.encode(answer_b.strip(), add_special_tokens=False)
        for idx, (tok_a, tok_b) in enumerate(zip(ids_a, ids_b)):
            if tok_a != tok_b:
                return tuple(ids_a[: idx + 1]), tuple(ids_b[: idx + 1])
        if len(ids_a) == len(ids_b):
            return None
        if len(ids_a) < len(ids_b):
            return tuple(ids_a), tuple(ids_b[: len(ids_a) + 1])
        return tuple(ids_a[: len(ids_b) + 1]), tuple(ids_b)

    def sequence_next_probs(self, prompt, concept, flowtime, n_steps, token_sequences):
        import torch

        gen = self.gen
        concept_hidden, concept_mask = gen.encode_concept(concept)
        result = {}
        for sequence in token_sequences:
            prefix_ids = list(sequence[:-1])
            target_id = sequence[-1]
            formatted = gen.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = gen.tokenizer(
                [formatted],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            ).to("cuda")
            input_ids = enc.input_ids
            if prefix_ids:
                input_ids = torch.cat([input_ids, torch.tensor([prefix_ids], device="cuda", dtype=input_ids.dtype)], dim=1)
            attention_mask = torch.ones_like(input_ids)
            self.prepare_hook(concept_hidden, concept_mask, flowtime, n_steps, attention_mask)
            try:
                with torch.no_grad():
                    out = gen.llm(input_ids, attention_mask=attention_mask, position_ids=gen._position_ids, use_cache=True)
                    dist = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
                    result[sequence] = float(dist[target_id].item())
            finally:
                self.clear_hook()
        return result

    def prepare_hook(self, concept_hidden, concept_mask, flowtime, n_steps, attention_mask):
        import torch

        gen = self.gen
        gen._n_steps = n_steps
        gen._concept_hidden = concept_hidden
        gen._concept_mask = concept_mask
        gen._flowtimes = torch.tensor([flowtime], device="cuda", dtype=torch.float32)
        gen._padding_mask = attention_mask.float()
        gen._sa_caches = [None] * n_steps
        gen._is_prefill = True
        gen._past_len = 0
        gen._position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        gen._install_hook()
        gen._active = True

    def clear_hook(self):
        import torch

        self.gen._active = False
        self.gen._sa_caches = None
        torch.cuda.empty_cache()

    def generate_text(self, prompt, concept, flowtime, n_steps, max_tokens, temperature):
        with self.model_lock:
            return self.gen.generate_batch(
                prompts=[prompt],
                concept_text=concept,
                flowtimes=[flowtime],
                n_steps=n_steps,
                max_tokens=max_tokens,
                temperature=temperature,
                max_batch=1,
            )[0]["generation"]

    def validate_role(self, role):
        if role not in {"alice", "bob"}:
            raise ValueError("Role must be alice or bob.")
        return role


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}")

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/alice", "/bob"}:
                self.write_html(INDEX_HTML)
                return
            if parsed.path == "/api/state":
                role = parse_qs(parsed.query).get("role", ["alice"])[0]
                with backend.state.lock:
                    self.write_json(backend.state.view(role))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(body)
                if self.path == "/api/load":
                    self.write_json({"message": backend.load()})
                    return
                if self.path == "/api/ask":
                    self.write_json(backend.ask_charlie(payload))
                    return
                if self.path == "/api/propose":
                    self.write_json(backend.submit_proposals(payload))
                    return
                if self.path == "/api/answer":
                    self.write_json(backend.submit_answers(payload))
                    return
                if self.path == "/api/reset":
                    with backend.state.lock:
                        backend.state.reset()
                        self.write_json({"ok": True})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def write_html(self, html):
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def write_json(self, payload, status=HTTPStatus.OK):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Run the theory-of-mind steering game.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--flow-ckpt", default="checkpoints/flas-gemma-2-9b-it/flas-gemma-2-9b-it.safetensors")
    parser.add_argument("--model-id", default="unsloth/gemma-2-9b-it")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    backend = FlasBackend(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    print(f"Serving theory-of-mind game at http://{args.host}:{args.port}")
    print("Use Ctrl-C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
