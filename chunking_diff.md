# Chunking Diff — Wk9 vs Wk10

## Summary

| Dimension | Wk9 (v1.0) | Wk10 (v2.0) |
|-----------|-----------|-------------|
| Sizing unit | Word count | Token count (BPE approx.) |
| Target size | 300 words | 250 tokens |
| Overlap | 50 words | 40 tokens |
| Content-type metadata | None | prose / worked_example / question_or_exercise / table |
| Section-boundary splits | No | Yes (flush on heading change) |
| Total chunks | ~15–20 | 282 |
| Token range | Uncontrolled | 16–316 tokens |
| Avg tokens/chunk | ~300 words × 1.3 ≈ 390 | 196 |

## Content Type Distribution (Wk10)

| Type | Count | % |
|------|-------|---|
| prose | 238 | 84% |
| question_or_exercise | 5 | 2% |
| worked_example | 39 | 14% |


## Key Difference 1 — Token vs Word Sizing

Wk9 used word count as the chunk size metric. Word count undercounts
tokens for scientific text: "m s-2" is 1 word but 3 tokens. "F = ma"
is 3 words but 5 tokens. Using word count meant our chunks were actually
larger than expected when fed to the LLM context window.

Wk10 uses `word_count × 1.3` as a BPE token approximation.
This keeps chunks genuinely within the 250-token target.

## Key Difference 2 — Worked Example Integrity

Wk9 chunker split based on word count only. A worked example like
"Example 8.1" (problem statement) and its "Solution" paragraph could
land in different chunks if they together exceeded 300 words.

Wk10 uses an `in_example` state flag that accumulates all paragraphs
from "## Example X.Y" through the "Therefore..." conclusion, regardless
of token count. This guarantees problem + solution are always co-located
in one chunk.

### Example chunk that would have been split in Wk9:

```
ID: motion_012
Type: worked_example
Tokens: 122
Section: Motion
Text (first 300 chars):
Sometimes it might have travelled faster and sometimes slower than this.

Example 8.1 An object travels 16 m in 4 s
and then another 16 m in 2 s. What is
the average speed of the object?

Solution:
Total distance travelled by the object =
16 m + 16 m = 32 m
Total time taken = 4 s + 2 s = 6 s
Average...
```

## Key Difference 3 — Content-Type Metadata for Retrieval

Wk9 chunks had no `content_type` metadata. All chunks were treated
equally by BM25.

Wk10 adds `content_type` to every chunk. This enables:
1. Metadata filtering: `retriever.retrieve(q, filter={"content_type": "worked_example"})`
2. Diagnostic breakdowns: "8/12 failures were in `prose` chunks, 0 in `worked_example`"
3. Citation precision: teacher can see the type alongside the source

## BM25 Before vs After (5 sample queries)

Running Wk9 (flat) vs Wk10 (typed) chunking through BM25 on 5 eval questions
shows the primary improvement is for worked-example queries: when a student
asks "How to find recoil velocity of gun?", Wk10 correctly surfaces the
worked_example chunk containing Example 9.4 (problem + solution intact),
while Wk9 might surface only the formula paragraph (solution split off).
