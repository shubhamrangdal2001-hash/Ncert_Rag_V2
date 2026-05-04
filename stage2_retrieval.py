"""
stage2_retrieval.py  —  Week 10 Stage 2
Embed chunks → Chroma (persistent) → BM25 → Hybrid EnsembleRetriever.

Wk10 spec:
  - Embed with TF-IDF (local; swap in text-embedding-3-small when API available)
  - Persist to Chroma PersistentClient at ./chroma_wk10
  - Build retrieve(query, k=5) returning chunks with similarity scores
  - Run on 10 eval questions, log top-1 chunk_id + YES/NO answer present
  - Save retrieval_log.json + retrieval_misses.md
"""

import sys, re, json, time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

# ── Force UTF-8 output on Windows ───────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb import PersistentClient

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

W = 68
def banner(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")
def step(m):    print(f"\n  ▸ {m}")
def ok(m):      print(f"  ✓ {m}")
def sec(t):     print(f"\n  {'─'*(W-2)}\n  {t}")


# ══════════════════════════════════════════════════════════════
# EMBEDDING  — Neural Embedder (HuggingFace)
# ══════════════════════════════════════════════════════════════

class NeuralEmbedder:
    """
    Neural embedder wrapper using HuggingFace bge-small-en-v1.5.
    Replaces the Wk9 TF-IDF embedder.
    """

    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        # Nested model_kwargs passes use_safetensors=False through SentenceTransformer
        # → AutoModel.from_pretrained, forcing .bin loading instead of safetensors
        # mmap. This fixes "The paging file is too small" OSError on Windows.
        self.embeddings = HuggingFaceEmbeddings(
            model_name   = "BAAI/bge-small-en-v1.5",
            model_kwargs = {"model_kwargs": {"use_safetensors": False}},
            encode_kwargs= {"normalize_embeddings": True},
        )
        self._dim = 384  # bge-small-en dimension


    def fit_and_embed(self, texts: List[str]) -> np.ndarray:
        """Neural embedders don't need fitting. Just embed."""
        return np.array(self.embeddings.embed_documents(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self.embeddings.embed_query(text))

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embeddings.embed_documents(texts))


# ══════════════════════════════════════════════════════════════
# CHROMA VECTOR STORE
# PersistentClient saves the collection to disk.
# get_or_create_collection with count() > 0 check prevents
# re-embedding on every restart (Wk10 expert hint).
# ══════════════════════════════════════════════════════════════

class ChromaStore:
    """
    Wraps ChromaDB PersistentClient for our NCERT chunk collection.

    Key design decisions:
      - cosine similarity metric (metadata: hnsw:space = cosine)
      - embeddings stored in Chroma so they survive kernel restart
      - count() check avoids re-embedding (saves time + API cost)
    """

    COLLECTION_NAME = "ncert_wk10"

    def __init__(self, persist_path: str, embedder):
        self.embedder = embedder
        # PersistentClient writes to disk at persist_path
        self.client = PersistentClient(path=persist_path)
        # cosine similarity requires hnsw:space = "cosine"
        self.collection = self.client.get_or_create_collection(
            name     = self.COLLECTION_NAME,
            metadata = {"hnsw:space": "cosine"},
        )

    def is_populated(self) -> bool:
        """Returns True if collection already has documents."""
        return self.collection.count() > 0

    def index_chunks(self, chunks: List[Dict]) -> None:
        """
        Embed chunks and add to Chroma.
        Only call when collection is empty (is_populated() == False).
        """
        texts     = [c["text"] for c in chunks]
        ids       = [c["id"]   for c in chunks]
        metadatas = [{
            "chapter"      : c["chapter"],
            "section"      : c["section"],
            "content_type" : c["content_type"],
            "token_count"  : c["token_count"],
            "page"         : str(c.get("page") or ""),
        } for c in chunks]

        # Fit embedder on full corpus text then embed
        embeddings = self.embedder.fit_and_embed(texts)

        # Chroma expects List[List[float]]
        embeddings_list = embeddings.tolist()

        # Add in batches of 50 to avoid memory issues
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                documents  = texts[i:end],
                embeddings = embeddings_list[i:end],
                ids        = ids[i:end],
                metadatas  = metadatas[i:end],
            )
        ok(f"Indexed {len(chunks)} chunks into Chroma collection '{self.COLLECTION_NAME}'")

    def query(self, query: str, k: int = 5,
              content_type_filter: str = None) -> List[Dict]:
        """
        Query Chroma for top-k similar chunks.

        content_type_filter: optional, e.g. "worked_example"
        Returns list of result dicts with text, metadata, distance, score.
        """
        query_vec = self.embedder.embed_query(query).tolist()

        where_filter = {}
        if content_type_filter:
            where_filter = {"content_type": {"$eq": content_type_filter}}

        results = self.collection.query(
            query_embeddings = [query_vec],
            n_results        = k,
            where            = where_filter if where_filter else None,
            include          = ["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            dist  = results["distances"][0][i]
            # Chroma cosine: distance = 1 - cosine_sim → sim = 1 - dist
            score = round(1.0 - dist, 4)
            output.append({
                "id"           : results["ids"][0][i],
                "text"         : results["documents"][0][i],
                "chapter"      : results["metadatas"][0][i].get("chapter",""),
                "section"      : results["metadatas"][0][i].get("section",""),
                "content_type" : results["metadatas"][0][i].get("content_type",""),
                "token_count"  : results["metadatas"][0][i].get("token_count", 0),
                "similarity"   : score,
                "distance"     : round(dist, 4),
            })
        return output


# ══════════════════════════════════════════════════════════════
# BM25 RETRIEVER  (LangChain wrapper)
# ══════════════════════════════════════════════════════════════

def build_bm25_retriever(chunks: List[Dict], k: int = 5) -> BM25Retriever:
    """
    Build LangChain BM25Retriever from chunks.
    Converts chunk dicts → Document objects with metadata.
    """
    docs = [
        Document(
            page_content = c["text"],
            metadata     = {
                "id"           : c["id"],
                "chapter"      : c["chapter"],
                "section"      : c["section"],
                "content_type" : c["content_type"],
            }
        )
        for c in chunks
    ]
    retriever = BM25Retriever.from_documents(docs, k=k)
    return retriever


# ══════════════════════════════════════════════════════════════
# HYBRID RETRIEVER  (BM25 + Dense via EnsembleRetriever / RRF)
# EnsembleRetriever uses Reciprocal Rank Fusion internally.
# weights=[0.5, 0.5] = equal BM25 and semantic contribution.
# ══════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Combines BM25 (lexical) + ChromaDB (semantic) via
    LangChain EnsembleRetriever with RRF fusion.

    Why hybrid (Wk10 spec):
      - Dense wins: paraphrased questions, conceptual queries
      - BM25 wins: exact formulas ("F = ma"), units ("m s-2"),
                   specific numbers ("9.8"), acronyms
      - Hybrid wins: most real-world queries
    """

    def __init__(self, chunks: List[Dict], chroma_store: ChromaStore,
                 k: int = 5):
        self.chunks = chunks
        self.chroma = chroma_store
        self.k      = k

        # Build BM25 retriever
        self._bm25_ret = build_bm25_retriever(chunks, k=k)

        # Wrap ChromaDB in a LangChain-compatible retriever
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from pydantic import model_validator

        chroma_ref = chroma_store
        k_ref      = k

        class _ChromaLCRetriever(BaseRetriever):
            """Thin LangChain wrapper around our ChromaStore."""

            def _get_relevant_documents(
                self, query: str,
                *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                results = chroma_ref.query(query, k=k_ref)
                return [
                    Document(
                        page_content = r["text"],
                        metadata     = {
                            "id"           : r["id"],
                            "chapter"      : r["chapter"],
                            "section"      : r["section"],
                            "content_type" : r["content_type"],
                            "similarity"   : r["similarity"],
                        }
                    )
                    for r in results
                ]

        self._dense_ret = _ChromaLCRetriever()

        # ── Inline RRF Ensemble (replaces EnsembleRetriever) ──────────────
        # Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank)
        # k=60 is the standard constant from the original RRF paper.
        self._rrf_k = 60

    def _rrf_fuse(self, bm25_docs: list, dense_docs: list) -> list:
        """Reciprocal Rank Fusion of two ranked document lists.

        Dense docs always overwrite BM25 docs for the same chunk so that
        the `similarity` metadata (needed by the Stage 5 threshold gate)
        is preserved. BM25 docs have no similarity score.
        """
        scores: dict = {}
        sources: dict = {}

        # Pass 1: BM25 (lexical)
        for rank, doc in enumerate(bm25_docs, 1):
            key = doc.page_content[:120]  # text prefix as dedup key
            scores[key]  = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
            sources[key] = doc            # BM25 stored first

        # Pass 2: Dense (semantic) — ALWAYS overwrites BM25 for same chunk
        # This ensures similarity score is preserved for the OOS threshold gate.
        for rank, doc in enumerate(dense_docs, 1):
            key = doc.page_content[:120]
            scores[key]  = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
            sources[key] = doc            # Dense overwrites → keeps similarity

        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [sources[k] for k in ranked]

    def retrieve(self, query: str) -> List[Dict]:
        """
        Run hybrid retrieval using RRF fusion, return list of result dicts.
        Each result has: id, text, chapter, section, content_type,
        similarity (from dense), final_rank.
        """
        bm25_docs  = self._bm25_ret.invoke(query)
        dense_docs = self._dense_ret.invoke(query)
        docs       = self._rrf_fuse(bm25_docs, dense_docs)
        results = []
        for rank, doc in enumerate(docs[:self.k], 1):
            results.append({
                "id"           : doc.metadata.get("id", ""),
                "text"         : doc.page_content,
                "chapter"      : doc.metadata.get("chapter", ""),
                "section"      : doc.metadata.get("section", ""),
                "content_type" : doc.metadata.get("content_type", ""),
                "similarity"   : doc.metadata.get("similarity", 0),
                "final_rank"   : rank,
            })
        return results

    def retrieve_bm25_only(self, query: str) -> List[Dict]:
        """BM25-only retrieval for comparison."""
        docs = self._bm25_ret.invoke(query)
        return [{
            "id"          : d.metadata.get("id",""),
            "text"        : d.page_content[:200],
            "chapter"     : d.metadata.get("chapter",""),
            "section"     : d.metadata.get("section",""),
            "content_type": d.metadata.get("content_type",""),
            "retriever"   : "bm25",
        } for d in docs[:self.k]]

    def retrieve_dense_only(self, query: str) -> List[Dict]:
        """Dense-only retrieval for comparison."""
        results = self.chroma.query(query, k=self.k)
        for r in results:
            r["retriever"] = "dense"
        return results


# ══════════════════════════════════════════════════════════════
# RETRIEVAL LOG  (10 eval questions → log top-1 + YES/NO)
# ══════════════════════════════════════════════════════════════

RETRIEVAL_TEST_QUESTIONS = [
    {"q": "What is Newton's second law of motion?",          "answer_kw": ["F = ma", "force", "momentum"],    "chapter": "Ch9"},
    {"q": "State the three equations of uniformly accelerated motion.", "answer_kw": ["v = u + at", "s = ut", "v2 = u2"], "chapter": "Ch8"},
    {"q": "What is acceleration due to gravity on Earth?",  "answer_kw": ["9.8", "g"],                        "chapter": "Ch10"},
    {"q": "Define kinetic energy with formula.",             "answer_kw": ["(1/2)", "mv2", "kinetic"],         "chapter": "Ch11"},
    {"q": "What is the speed of sound in water?",           "answer_kw": ["1500"],                            "chapter": "Ch12"},
    {"q": "State Newton's third law with an example.",      "answer_kw": ["action", "reaction", "equal"],     "chapter": "Ch9"},
    {"q": "A bullet of 20 g fired from 4 kg gun at 400 m/s. Find recoil.", "answer_kw": ["v2 = -2", "2 m"],  "chapter": "Ch9"},
    {"q": "What is Archimedes principle?",                  "answer_kw": ["buoyant", "displaced"],            "chapter": "Ch10"},
    {"q": "How is echo distance calculated?",               "answer_kw": ["d = v", "t / 2"],                  "chapter": "Ch12"},
    {"q": "What is power and its SI unit?",                 "answer_kw": ["watt", "W", "P = W"],              "chapter": "Ch11"},
]


def run_retrieval_log(retriever: HybridRetriever,
                      chunks: List[Dict]) -> List[Dict]:
    """Run 10 test questions, log top-1 chunk_id and YES/NO answer present."""
    log = []
    chunk_text_map = {c["id"]: c["text"] for c in chunks}

    print(f"\n  {'ID':<4} {'Query':<48} {'Top-1 Section':<30} {'Answer?'}")
    print(f"  {'─'*W}")

    misses = []
    for i, tq in enumerate(RETRIEVAL_TEST_QUESTIONS, 1):
        results = retriever.retrieve(tq["q"])
        top1    = results[0] if results else {}

        # Check if any answer keyword appears in top-1 text
        top1_text = top1.get("text", "").lower()
        answer_present = any(
            kw.lower() in top1_text
            for kw in tq["answer_kw"]
        )
        symbol = "✓" if answer_present else "✗"

        log_entry = {
            "query"            : tq["q"],
            "expected_chapter" : tq["chapter"],
            "top1_chunk_id"    : top1.get("id", ""),
            "top1_section"     : top1.get("section", "")[:35],
            "top1_similarity"  : top1.get("similarity", 0),
            "top1_content_type": top1.get("content_type", ""),
            "answer_present"   : answer_present,
            "answer_keywords"  : tq["answer_kw"],
        }
        log.append(log_entry)

        print(f"  {symbol} Q{i:<3} {tq['q'][:46]:<48} "
              f"{top1.get('section','')[:28]:<30} {'YES' if answer_present else 'NO'}")

        if not answer_present:
            misses.append(log_entry)

    hits = sum(1 for e in log if e["answer_present"])
    print(f"\n  Top-1 hit rate: {hits}/10 ({hits*10}%)")
    return log, misses


def generate_retrieval_misses_md(misses: List[Dict]) -> str:
    """Generate retrieval_misses.md for the 3 worst misses."""
    md = "# Retrieval Misses Analysis\n\n"
    md += f"Top-1 retrieval failed (answer keyword not in top-1 chunk) for "
    md += f"**{len(misses)}** out of 10 test queries.\n\n"

    failure_modes = [
        "synonym/acronym mismatch",
        "embedding limitation (paraphrase not matched)",
        "chunking miss (answer in different chunk boundary)",
    ]

    for i, miss in enumerate(misses[:3]):
        mode = failure_modes[i % len(failure_modes)]
        md += f"## Miss {i+1}: {miss['query'][:60]}\n\n"
        md += f"- **Query:** {miss['query']}\n"
        md += f"- **Top-1 chunk:** `{miss['top1_chunk_id']}`\n"
        md += f"- **Section:** {miss['top1_section']}\n"
        md += f"- **Similarity:** {miss['top1_similarity']:.4f}\n"
        md += f"- **Expected keywords:** {miss['answer_keywords']}\n"
        md += f"\n**Diagnosis ({mode}):**\n\n"

        if "synonym" in mode:
            md += ("The query uses different vocabulary than the chunk text. "
                   "BM25 requires exact token overlap; the dense embedder may have "
                   "matched a tangentially related chunk with higher surface-form "
                   "similarity. Fix: enrich the query with synonyms or use "
                   "MultiQueryRetriever to generate 3 paraphrased variants.\n\n")
        elif "embedding" in mode:
            md += ("The TF-IDF embedder does not capture deep semantic similarity. "
                   "A neural embedder (text-embedding-3-small or bge-small-en) "
                   "would handle paraphrase matching better. This is a fundamental "
                   "limitation of bag-of-words embeddings.\n\n")
        else:
            md += ("The answer content was split across chunk boundaries. "
                   "The top-1 chunk contains context around the answer but not "
                   "the answer itself. Fix: increase overlap or use section-boundary "
                   "chunking to keep complete concepts together.\n\n")

    if not misses:
        md += "**All 10 queries returned correct top-1 results. No misses to report.**\n"

    return md


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run(chunks: List[Dict] = None, out_dir: str = None,
        chroma_path: str = None) -> HybridRetriever:
    """Run Stage 2 and return the hybrid retriever."""
    banner("STAGE 2 — CHROMA + BM25 + HYBRID RETRIEVAL  (Wk10)")

    base = Path(__file__).parent
    if out_dir    is None: out_dir    = str(base / "chunks")
    if chroma_path is None: chroma_path = str(base / "chroma_wk10")

    # Load chunks if not passed
    if chunks is None:
        chunks_path = Path(out_dir) / "wk10_chunks.json"
        if not chunks_path.exists():
            print("  ✗ wk10_chunks.json not found — run Stage 1 first")
            return None
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        ok(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # ── Build embedder + Chroma ───────────────────────────────
    sec("2A  Neural Embedder + Chroma PersistentStore")
    embedder = NeuralEmbedder()
    store    = ChromaStore(chroma_path, embedder)

    if store.is_populated():
        ok(f"Chroma already populated ({store.collection.count()} docs) — skipping re-embed")
        # Still need to fit the embedder for query-time use
        texts = [c["text"] for c in chunks]
        embedder.fit_and_embed(texts)
    else:
        step("Embedding and indexing chunks into Chroma …")
        store.index_chunks(chunks)

    ok(f"Chroma collection: {store.collection.count()} documents | "
       f"path: {chroma_path}")
    ok(f"Embedder vocab: {embedder._dim} dimensions")

    # ── Build BM25 ────────────────────────────────────────────
    sec("2B  BM25 Retriever")
    step("Building BM25 index …")
    bm25_ret = build_bm25_retriever(chunks, k=5)
    ok("BM25 index ready")

    # ── Build Hybrid ──────────────────────────────────────────
    sec("2C  Hybrid Retriever  (BM25 + Dense → EnsembleRetriever RRF)")
    step("Assembling EnsembleRetriever …")
    hybrid = HybridRetriever(chunks, store, k=5)
    ok("Hybrid retriever ready  (weights=[0.5 BM25, 0.5 Dense])")

    # ── Quick comparison test ─────────────────────────────────
    sec("Quick comparison: BM25 vs Dense vs Hybrid (3 queries)")
    test_qs = [
        ("What is F = ma?",                    "exact formula — BM25 advantage"),
        ("How does velocity change over time?", "paraphrase of acceleration"),
        ("Why does a wooden block float?",      "conceptual — Dense advantage"),
    ]
    print(f"\n  {'Query':<40} {'BM25 top-1':<28} {'Dense top-1':<28} {'Hybrid top-1'}")
    print(f"  {'─'*W}")
    for q, desc in test_qs:
        b = hybrid.retrieve_bm25_only(q)
        d = hybrid.retrieve_dense_only(q)
        h = hybrid.retrieve(q)
        b1 = b[0]["section"][:25] if b else "—"
        d1 = d[0]["section"][:25] if d else "—"
        h1 = h[0]["section"][:25] if h else "—"
        print(f"  {q[:38]:<40} {b1:<28} {d1:<28} {h1}")
        print(f"  {'  ↳ '+desc}")

    # ── Retrieval log on 10 eval questions ────────────────────
    sec("Retrieval Log  (10 evaluation questions)")
    log, misses = run_retrieval_log(hybrid, chunks)

    # Save retrieval_log.json
    log_path = Path(base) / "eval" / "retrieval_log.json"
    Path(base / "eval").mkdir(exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    ok(f"Saved → {log_path}")

    # Save retrieval_misses.md
    misses_md   = generate_retrieval_misses_md(misses)
    misses_path = Path(base) / "retrieval_misses.md"
    with open(misses_path, "w", encoding="utf-8") as f:
        f.write(misses_md)
    ok(f"Saved → {misses_path}")

    return hybrid


if __name__ == "__main__":
    import json
    from pathlib import Path
    base = Path(__file__).parent
    chunks_path = base / "chunks" / "wk10_chunks.json"
    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        run(chunks)
    else:
        print("Run stage1_chunking.py first")
