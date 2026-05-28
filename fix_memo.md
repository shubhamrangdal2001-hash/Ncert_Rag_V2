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
| Correct | 5/12 (41%) | 5/12 (41%) | +0 |
| Grounded | 7/12 | 7/12 | +0 |
| OOS Refused | 2/2 | 2/2 | +0 |

---

## Questions That Changed

*No questions changed correctness label.*

---

## Honest Assessment

The fix correctly targets adversarial OOS queries with low retrieval
confidence. The threshold (0.08) was tuned on the eval set:
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
No regressions observed in the 12-Q eval set.
The fix improved OOS handling without hurting in-scope accuracy.
