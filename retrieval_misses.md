# Retrieval Misses Analysis

Top-1 retrieval failed (answer keyword not in top-1 chunk) for **3** out of 10 test queries.

## Miss 1: What is the speed of sound in water?

- **Query:** What is the speed of sound in water?
- **Top-1 chunk:** `sound_048`
- **Section:** Sound
- **Similarity:** 0.7449
- **Expected keywords:** ['1500']

**Diagnosis (synonym/acronym mismatch):**

The query uses different vocabulary than the chunk text. BM25 requires exact token overlap; the dense embedder may have matched a tangentially related chunk with higher surface-form similarity. Fix: enrich the query with synonyms or use MultiQueryRetriever to generate 3 paraphrased variants.

## Miss 2: A bullet of 20 g fired from 4 kg gun at 400 m/s. Find recoil

- **Query:** A bullet of 20 g fired from 4 kg gun at 400 m/s. Find recoil.
- **Top-1 chunk:** `force_and_laws_of_motion_046`
- **Section:** Force and Laws of Motion
- **Similarity:** 0.7810
- **Expected keywords:** ['v2 = -2', '2 m']

**Diagnosis (embedding limitation (paraphrase not matched)):**

The TF-IDF embedder does not capture deep semantic similarity. A neural embedder (text-embedding-3-small or bge-small-en) would handle paraphrase matching better. This is a fundamental limitation of bag-of-words embeddings.

## Miss 3: How is echo distance calculated?

- **Query:** How is echo distance calculated?
- **Top-1 chunk:** `sound_031`
- **Section:** Sound
- **Similarity:** 0.7070
- **Expected keywords:** ['d = v', 't / 2']

**Diagnosis (chunking miss (answer in different chunk boundary)):**

The answer content was split across chunk boundaries. The top-1 chunk contains context around the answer but not the answer itself. Fix: increase overlap or use section-boundary chunking to keep complete concepts together.

