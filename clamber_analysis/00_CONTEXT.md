# 00 — Research Context

## What This Folder Is

This folder contains the CLAMBER dataset (JSON) and two markdown guides:
- `00_CONTEXT.md` — this file, background on the paper and my research intent
- `01_ANALYSIS_PLAN.md` — the step-by-step analysis plan to execute

---

## The CLAMBER Paper

**Title:** CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models
**Authors:** Tong Zhang, Peixin Qin, Yang Deng, Chen Huang et al. — Sichuan University & NUS
**Venue:** ACL 2024
**Dataset size:** ~12,000 entries

### What CLAMBER Does

CLAMBER is a benchmark that evaluates how well LLMs handle ambiguous user queries across two tasks:

1. **Ambiguity identification** — given a user query, does the LLM correctly detect whether it is ambiguous?
2. **Clarification generation** — if the query is ambiguous, does the LLM generate a useful clarifying question?

The paper evaluates five LLMs: Vicuna-13B, Llama2-13B, Llama2-13B-Instruct, Llama2-70B, and ChatGPT, across four prompting schemes: zero-shot, zero-shot with chain-of-thought, few-shot, and few-shot with chain-of-thought.

### CLAMBER's Ambiguity Taxonomy

Three dimensions, eight subclasses:

**Epistemic Misalignment (FD)** — ambiguity from the LLM's internal knowledge state
- `NK` (Unfamiliar): the query contains an entity the LLM doesn't recognise
- `ICL` (Contradiction): the query contains examples that lead to conflicting interpretations

**Linguistic Ambiguity (LA)** — ambiguity in the wording itself
- `polysemy` (Lexical): a word has multiple meanings
- `co-reference` (Semantic): a pronoun or reference lacks context

**Aleatoric Output (MC)** — query is well-formed but missing information needed for a specific answer
- `whom`: missing personal or preference details
- `what`: missing task-specific details
- `when`: missing temporal details
- `where`: missing spatial details

### Dataset Fields

| Field | Description |
|---|---|
| `question` | The original user query |
| `context` | Optional context, often empty |
| `clarifying_question` | Ground truth clarifying question (human-validated) |
| `require_clarification` | Binary label — 1 = ambiguous, 0 = not ambiguous |
| `category` | `MC`, `FD`, or `LA` |
| `subclass` | `whom`, `what`, `when`, `where`, `NK`, `ICL`, `polysemy`, `co-reference` |
| `predict_ambiguous` | LLM binary prediction — 1 = flagged ambiguous, 0 = not |
| `predict_is_ambiguous_response` | Raw LLM output as JSON string with `Output` (True/False) and `Confidence` (1–5) |
| `predict_clarifying_question` | The LLM's generated clarifying question — main signal for my analysis |

### Key Findings from the Paper

- All LLMs struggle to identify ambiguity even with CoT and few-shot prompting
- Small-scale LLMs tend to over-classify everything as ambiguous (high false positive rate)
- CoT and few-shot prompting can cause overconfidence in smaller models, making them worse
- ChatGPT is the best performer but still only reaches ~54% accuracy and ~53% F1
- ChatGPT struggles most on the `ICL` (contradiction) subclass — it treats contradictory queries as unambiguous (RLHF pushes it to always answer)
- The main failure in clarification generation is asking about the wrong aspect of the query — the model cannot accurately locate its own knowledge gap

### Which LLM Generated the `predict_*` Fields?

The public dataset does not label which model produced the predictions. One goal of this analysis is to infer the model by matching computed accuracy/F1 values against Tables 3 and 6 in the paper.

---

## My Research

### The Core Goal

I am building a benchmark to classify the **internal statistical uncertainty** that LLMs experience, using the clarifying questions they generate as a diagnostic signal.

### The Two Core Uncertainty Types

**Epistemic uncertainty** — a lack of knowledge. The model doesn't know the answer because the fact or entity is outside its training data or beyond its capacity. This is *reducible* — more data or a better model can eliminate it. It typically surfaces with out-of-distribution inputs.

Example: "Vänern is the largest lake in ___" has a deterministic correct answer. Any uncertainty here is a knowledge gap.

**Aleatoric uncertainty** — inherent randomness in the data distribution. Multiple valid answers exist for the same prompt and no amount of additional training resolves this — it is *irreducible*.

Example: "Vänern is ___" has countless valid completions. The uncertainty is in the problem itself, not the model.

### How This Differs from CLAMBER

CLAMBER asks: *what is wrong or underspecified in the user's prompt?* It taxonomises the prompt.

My research asks: *what is the model experiencing internally?* It uses the generated clarifying question as a window into the model's uncertainty state.

The same ambiguous prompt can trigger different uncertainty types in different models. A query containing an unfamiliar entity might cause epistemic uncertainty in a smaller model (genuine knowledge gap) but aleatoric uncertainty in a larger model (it recognises multiple valid referents and cannot choose). CLAMBER's label stays fixed to the prompt; my label depends on the model's response.

### Why I Am Analysing CLAMBER First

This is a warm-up step before building my own benchmark. Goals:

1. Build empirical intuition about what LLM-generated clarifying questions look like at scale
2. Test whether CLAMBER's prompt-level categories correlate with the uncertainty types I care about — e.g. does `NK` reliably produce epistemically-flavoured questions?
3. Understand when the model stays silent (predicts not ambiguous despite being wrong) — this is where suppressed uncertainty lives and is very relevant to my research
4. Identify which LLM produced the `predict_*` fields by matching paper metrics
5. Trial a preliminary annotation approach for uncertainty type labelling

### My Uncertainty Taxonomy (Work in Progress)

| Type | Signal in the clarifying question |
|---|---|
| Epistemic | Model asks to confirm or define something unfamiliar — signals a knowledge gap |
| Aleatoric | Model cannot produce a single answer because multiple valid outputs exist. This covers both discrete disambiguation (e.g. "which World Cup year?") and open-ended personalisation (e.g. "what are your mother's interests?"). Both are irreducible without more context from the user. |

### Important Note on Scope

This CLAMBER analysis is exploratory and diagnostic, not the final benchmark. For the actual research I will use a different, larger dataset and generate clarifying questions from scratch using current LLMs. The output of this analysis is intuition and methodology, not publishable results.
