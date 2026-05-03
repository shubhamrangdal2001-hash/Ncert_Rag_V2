# Retrieval Misses Analysis

Top-1 retrieval failed (answer keyword not in top-1 chunk) for **3** out of 10 test queries.

## Miss 1: What is Newton's second law of motion?

- **Query:** What is Newton's second law of motion?
- **Top-1 chunk:** `work_and_energy_004`
- **Section:** Work and Energy
- **Similarity:** 0.0000
- **Expected keywords:** ['F = ma', 'force', 'momentum']

**Diagnosis (synonym/acronym mismatch):**

The query uses different vocabulary than the chunk text. BM25 requires exact token overlap; the dense embedder may have matched a tangentially related chunk with higher surface-form similarity. Fix: enrich the query with synonyms or use MultiQueryRetriever to generate 3 paraphrased variants.

## Miss 2: What is the speed of sound in water?

- **Query:** What is the speed of sound in water?
- **Top-1 chunk:** `sound_006`
- **Section:** Sound
- **Similarity:** 0.7188
- **Expected keywords:** ['1500']

**Diagnosis (embedding limitation (paraphrase not matched)):**

The TF-IDF embedder does not capture deep semantic similarity. A neural embedder (text-embedding-3-small or bge-small-en) would handle paraphrase matching better. This is a fundamental limitation of bag-of-words embeddings.

## Miss 3: How is echo distance calculated?

- **Query:** How is echo distance calculated?
- **Top-1 chunk:** `sound_013`
- **Section:** Sound
- **Similarity:** 0.6339
- **Expected keywords:** ['d = v', 't / 2']

**Diagnosis (chunking miss (answer in different chunk boundary)):**

The answer content was split across chunk boundaries. The top-1 chunk contains context around the answer but not the answer itself. Fix: increase overlap or use section-boundary chunking to keep complete concepts together.

