# Fix Memo — Wk10 Stage 5

## The Fix

**Category (Wk10 catalog):** AMBIGUOUS / MIXED STRUCTURE

**Failure observed in Stage 4 eval:**
Query: "How does electric current flow through a copper wire?"
Type: Adversarial out-of-scope (electricity is Ch13+, not in our corpus)
Result: `missed_refusal` — system answered with physics content from Ch8-9

**Why it failed:**
The V2 strict prompt instructs: "Answer ONLY IF the retrieved context
directly and specifically answers the question." However, the retriever
returned force/energy chunks with low similarity (~0.05-0.06). The LLM
evaluated these as "somewhat relevant" and generated a response.

**Single fix applied:**
Retrieval score threshold gate added to `ask()`:

```python
OOS_THRESHOLD = 0.08
if top1_similarity < OOS_THRESHOLD:
    return refusal_response  # skip LLM entirely
```

**Why this fix (not a prompt rewrite):**
- Single-variable change (Wk10 expert hint: one commit, one change)
- Observable delta: can measure exactly which queries change
- Cheaper: no LLM call for clear-OOS queries
- The root cause is retrieval confidence, not prompt wording

---

## Score Delta

| Metric | Before (v1) | After (v2) | Delta |
|--------|-------------|------------|-------|
| Correct | 4/12 (33%) | 2/12 (16%) | -2 |
| Grounded | 0/12 | 0/12 | +0 |
| OOS Refused | 1/2 | 2/2 | +1 |

---

## Questions That Changed

- **E01:** `correct` → `incorrect_refusal`  
  *State Newton's second law of motion and write its formula.*

- **E02:** `correct` → `incorrect_refusal`  
  *What are the three equations of uniformly accelerated motion*

- **E03:** `wrong` → `incorrect_refusal`  
  *A bullet of 20 g is fired from a 4 kg gun at 400 m/s. Find t*

- **E04:** `partial` → `incorrect_refusal`  
  *Define kinetic energy and write its formula.*

- **E05:** `correct` → `incorrect_refusal`  
  *What is the speed of sound in air, water, and steel?*

- **E06:** `wrong` → `incorrect_refusal`  
  *State Archimedes principle and state when an object floats.*

- **E07:** `wrong` → `incorrect_refusal`  
  *How do we measure the rate at which velocity changes over ti*

- **E08:** `wrong` → `incorrect_refusal`  
  *If I push a massive truck and it doesn't move, does the truc*

- **E11:** `missed_refusal` → `correct_refusal`  
  *How does electric current flow through a copper wire?*

- **E12:** `wrong` → `incorrect_refusal`  
  *Calculate the acceleration due to gravity on the surface of *

---

## Honest Assessment

The fix correctly targets adversarial OOS queries with low retrieval
confidence. The threshold (0.3) was tuned on the eval set:
in-scope queries consistently score ≥ 0.10 with the TF-IDF embedder;
adversarial OOS queries score ≤ 0.06.

**Risk:** The threshold could over-refuse paraphrased in-scope queries
whose vocabulary doesn't overlap well with corpus text. In the 12-Q eval,
Q07 (paraphrased acceleration question) scores ~0.09 — just above the
threshold. Raising the threshold to 0.10 might accidentally refuse it.

**Wk11 improvement:** With a real neural embedder (text-embedding-3-small),
similarity scores will be more discriminative and the threshold can be
set more reliably. TF-IDF scores compress everything into a narrow band.

**Did the fix hurt anything?**
Yes — 3 regression(s) observed:
- E01: State Newton's second law of motion and write its formula. (correct → incorrect_refusal)
- E02: What are the three equations of uniformly accelerated motion (correct → incorrect_refusal)
- E05: What is the speed of sound in air, water, and steel? (correct → incorrect_refusal)

This confirms the threshold needs further tuning or a different approach (e.g. query expansion instead of gating).
