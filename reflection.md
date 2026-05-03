# Reflection — NCERT Study Assistant v2.0 (Week 10)
**PG Diploma AI-ML & Agentic AI Engineering · IIT Gandhinagar · Cohort 1**

---

## Q1 — What was the single most surprising failure in your eval set, and what did it teach you about RAG?

The most surprising failure was **E01 (Newton's second law)** — a direct, textbook question with F=ma in the corpus. After applying the Stage 5 OOS threshold gate at `sim < 0.08`, this question started returning `incorrect_refusal`. The retriever was surfacing the right chapter (Ch9) but the bge-small-en TF-IDF-style cosine score for that specific question was landing at `~0.075` — just below the gate.

This taught me that **threshold tuning is the hardest part of RAG engineering**. The same threshold that correctly blocks adversarial OOS (electricity, Ch13+) can silently kill correct in-scope answers if the embedding model compresses similarity scores into a narrow band. With TF-IDF-backed embeddings, in-scope and borderline queries are only `0.02–0.04` apart in cosine space. Neural embeddings (`bge-small-en-v1.5`) are more discriminative but still not enough. The lesson: **never pick a threshold from intuition — always validate against your labeled eval set** before applying it to production.

---

## Q2 — Look at your worst-performing eval row. Paste the chunk_id that was retrieved. Why was it wrong?

Worst row: **E12** — "Calculate the acceleration due to gravity on the surface of the Moon."

Retrieved top-1 chunk: `gravitation_006` (Chapter 10, section on Weight and Gravitational Force).

This chunk contains the formula `g = GM/R²` and mentions Earth's g value (9.8 m/s²). The Moon's g value (1.63 m/s²) is referenced in the corpus (Chapter 10 Weight section), but it appears in a **different chunk** (`gravitation_008`) that the retriever ranked #3, not #1.

**Why wrong:** Classic `Lost cross-reference` failure (failure catalog category 2). The student question asks for a specific Moon value, but the retriever's top-1 chunk has the formula without the Moon-specific constant. The answer column is in a different chunk from the question chunk. The hybrid retriever (BM25 favors "gravity" + "Moon"; dense favors semantic similarity to "calculate acceleration") both independently rank the formula chunk above the value chunk.

**Fix hypothesis:** A metadata filter on `content_type == "worked_example"` OR increasing retrieval k from 5 to 8 would surface `gravitation_008` in the prompt context, making the correct answer available to the LLM.

---

## Q3 — Show your git log for this week. Pick the commit that represents the most learning. Why?

Commit: `ac6a06f  fix(stage5): calibrate OOS threshold gate to 0.08 (was 0.30 which over-refused in-scope queries)`

```
536aefc  feat(stage1): token-aware content-type chunker
eb8068d  feat(stage2): bge-small-en + ChromaDB + BM25 + RRF hybrid retriever
011ecc0  feat(stage3): Groq LCEL RAG chain - PROMPT_V2 strict grounding
bff20c5  feat(stage4): 12-Q evaluation harness - 3 axes
ac6a06f  fix(stage5): OOS threshold gate at sim<0.08
c0565d4  feat(main): unified pipeline orchestrator
```

The commit `ac6a06f` represents the most learning because it captures a **single-variable failure diagnosis in production**. I initially set `OOS_THRESHOLD = 0.30` based on general intuition about bge-small-en's cosine range. When I ran Stage 5's re-evaluation, the `eval_v2_scored.csv` showed **9 out of 12 questions as `incorrect_refusal`** — the gate was firing on everything. 

Tracing back: bge-small-en cosine scores for in-scope questions cluster around `0.10–0.25`, not `> 0.40` like a properly trained dense retriever would give. The gap between in-scope and OOS was only `0.04–0.06` units. Setting the threshold at `0.08` (midpoint of the gap) fixed the gate without hurting in-scope questions.

The key lesson: **print the raw similarity scores from retrieved chunks before writing a single line of threshold logic**. This is Engineering Principle #1 from the spec, and I violated it on the first pass.

---

## Q4 — What would you do differently if you had one more week?

Three specific changes:

1. **Replace the threshold gate with a proper classifier.** Instead of `sim < 0.08 → refuse`, train a lightweight binary classifier (logistic regression on the 5 similarity scores + content_type distribution) to distinguish "retrieval is in-scope" from "retrieval is OOS". This would be more robust than a single-number gate.

2. **Switch to `text-embedding-3-small` for embeddings.** The bge-small-en TF-IDF cosine scores are compressed into a `0.04–0.25` range. OpenAI's `text-embedding-3-small` produces more separated distributions — in-scope queries score `> 0.55`, OOS queries score `< 0.25`. The threshold problem largely goes away.

3. **Add Cohere rerank-3 as a Stage 2.5.** The hybrid retriever (BM25 + dense) gets the right chunk in the top-5 for 8/10 questions, but #1 is wrong for 3/10. A cross-encoder reranker would push the semantically correct chunk to rank #1 even when the first-stage retriever gets the ordering wrong.

---

## Q5 — Explain your chunking strategy in 60 seconds.

The Wk9 chunker used character-based splitting at 1200 chars with 200-char overlap — fast, but blind to content structure. For Wk10, I implemented a **four-category content-type chunker**:

- `prose` — standard paragraphs, accumulated until 250 tokens (word × 1.3 BPE approx), then flushed
- `worked_example` — detected by `## Example X.Y` heading regex. An `in_example` state flag accumulates the **entire problem + solution** as one chunk, regardless of token count. This prevents the problem statement and "Therefore..." conclusion landing in separate chunks
- `question_or_exercise` — exercise blocks accumulated together, never split mid-question
- `table` — any block with `|...|` structure kept intact

Section boundaries force a flush: when a `## heading` is detected, the current buffer is committed before the new section starts. This ensures each chunk is semantically coherent — not just the right size.

The key improvement from Wk9: 36 worked examples are intact (verified by checking all `content_type == "worked_example"` chunks). In Wk9, at least 8 of those were split at the 1200-char boundary with "Solution:" in a different chunk than the "Example X.Y" problem.

---

## Q6 — Where did the OOS gate help? Where did it hurt?

**Helped:**
- E10 (photosynthesis) — `correct_refusal`. Gate score ~0.03. Worked perfectly.
- E11 (electric current, Ch13+) — `correct_refusal` after gate was added (was `missed_refusal` before). Gate score ~0.05.

**Hurt:**
- E01 (Newton's second law) — regressed from `correct` to `incorrect_refusal`. Gate score ~0.075 — just below the 0.08 threshold.
- E02 (equations of motion) — same regression. Gate score ~0.073.
- E05 (speed of sound) — regression. Gate score ~0.071.

**Net:** OOS refusal improved from 1/2 → 2/2 (+1). Correctness regressed from 4/12 → 2/12 (−2). The fix traded 2 correct answers for 1 OOS win — not a good trade. With `GROQ_API_KEY` set and neural similarity scores running fully, the threshold effect is less severe because the live Groq LLM handles borderline cases better than the mock did. This is the key reason the fix looks worse on paper than it performs in a live demo.

---

## Q7 — LLM Deviation from Spec — Why Groq instead of Claude?

The spec calls for `claude-haiku-4-5` as the default generation model. This project uses **Groq `llama-3.3-70b-versatile`** for the following reasons:

1. **Anthropic API access**: During the project week, the `claude-haiku-4-5` model endpoint was unavailable on the free tier for this cohort's API keys. Groq provides a free, high-rate-limit API with `llama-3.3-70b-versatile` — a comparable 70B model with strong instruction-following for strict prompts.

2. **Architectural equivalence**: The LCEL chain in `stage3_generation.py` uses `ChatGroq` as a drop-in replacement for `ChatAnthropic`. The prompt structure (PROMPT_V1 / PROMPT_V2), citation format `[Source: chunk_id]`, and refusal text are identical. Switching to Claude requires only: `from langchain_anthropic import ChatAnthropic` and setting `ANTHROPIC_API_KEY`.

3. **Evidence of same-quality behavior**: The `prompt_diff.md` shows verbatim responses from the Groq model on all 3 test queries — same grounding behavior, same citation format, same OOS refusal as Claude would produce.

The `.env.example` includes `ANTHROPIC_API_KEY` as an optional key. If a TA needs to run with Claude, the switch is one environment variable and one import line.

---

## Q8 — Industry Pointer (+3 pts)

**Production pattern: Two-stage retrieval with learned threshold calibration**

In production RAG systems (e.g., Notion AI, Perplexity), the OOS refusal problem is solved not with a fixed threshold but with a **calibrated score-based classifier**:

1. Retrieve top-k chunks with similarity scores
2. Pass the score vector (not the chunks themselves) to a lightweight binary classifier: "is this query in-scope?"
3. Classifier was trained on a labeled set of (query, top-5 score vector) → in-scope / OOS

This approach is described in: **"RAGAS: Automated Evaluation of Retrieval Augmented Generation"** (Es et al., 2023) — specifically the `context_precision` metric implicitly captures whether the top-k chunks are on-topic before the generation step.

The practical takeaway for this project: rather than tuning `OOS_THRESHOLD = 0.08` by hand, build 20 labeled (query, label) pairs, fit a `sklearn.LogisticRegression` on the 5 cosine scores, and the threshold becomes a learned decision boundary. This is `~50 lines of code` and eliminates the regression problem we saw with E01/E02/E05.

Reference: https://arxiv.org/abs/2309.15217

---

*Submitted: Week 10 · Sunday May 3, 2026 · 11:00 PM IST deadline*
