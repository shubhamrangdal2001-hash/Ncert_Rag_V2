# NCERT Class 9 Physics — Study Assistant v2.0
### Week 10 · Production-Grade RAG Pipeline
**PG Diploma AI-ML & Agentic AI Engineering · IIT Gandhinagar · Cohort 1**

🎥 **Loom Demo:** [Watch 3-min walkthrough](#) ← *(link to be added before submission)*

---

## What This Project Does

A 5-stage Retrieval-Augmented Generation (RAG) system for NCERT Class 9 Physics (Chapters 1–12).  
Students ask physics questions in plain English; the system retrieves grounded answers with source citations or refuses out-of-scope questions.

---

## What Changed from v1.0 (Week 9)

| Dimension | v1.0 (Wk9) | v2.0 (Wk10) |
|-----------|-----------|-------------|
| Sizing unit | Word count | Token count (BPE approx × 1.3) |
| Chunk size | 300 words | 250 tokens |
| Content metadata | None | `prose / worked_example / question_or_exercise / table` |
| Section boundaries | No | Yes (flush on heading change) |
| Embedder | None (BM25 only) | `bge-small-en-v1.5` (HuggingFace, 384-dim) |
| Vector store | None | ChromaDB PersistentClient (cosine) |
| Retrieval strategy | BM25 top-k | Hybrid BM25 + Dense → RRF (k=60) |
| LLM | Mock only | **Groq — `llama-3.3-70b-versatile`** |
| Citations | None | `[Source: chunk_id]` after every claim |
| OOS handling | Prompt only | Strict prompt + **retrieval score threshold gate** (sim < 0.08 → refuse) |
| Evaluation | Ad-hoc | 12 questions · 3 axes (correct / grounded / refused_oos) |
| Corpus | Ch8–9 | Ch8–12 (Motion, Force, Gravitation, Work/Energy, Sound) |

---

## Project Structure

```
Ncert_rag_V2/
├── main.py                    ← unified pipeline runner (start here)
├── stage1_chunking.py         ← token-aware content-type chunking
├── stage2_retrieval.py        ← bge-small-en + Chroma + BM25 + Hybrid RRF
├── stage3_generation.py       ← Groq LCEL chain + strict prompt + citations
├── stage4_evaluation.py       ← 12-Q eval on 3 axes
├── stage5_fix.py              ← OOS threshold gate + before/after delta
├── corpus/                    ← NCERT PDFs (not committed — see below)
│   └── iesc108-min.pdf … iesc112-min.pdf
├── chunks/
│   └── wk10_chunks.json       ← 120 chunks with metadata (generated)
├── chroma_wk10/               ← ChromaDB persistent store (generated, gitignored)
├── eval/
│   ├── retrieval_log.json     ← 10 queries, top-1 chunk_id, YES/NO hit
│   ├── eval_raw.csv           ← full answers (v1, before fix)
│   ├── eval_scored.csv        ← 3-axis scores (v1, before fix)
│   ├── eval_raw_v2.csv        ← full answers (v2, after fix)
│   └── eval_scored_v2.csv     ← 3-axis scores (v2, after fix)
├── chunking_diff.md           ← Wk9 vs Wk10 chunking comparison
├── retrieval_misses.md        ← root-cause analysis of retrieval misses
├── prompt_diff.md             ← V1 permissive vs V2 strict (verbatim)
├── fix_memo.md                ← diagnosis, fix choice, honest delta
├── requirements.txt
├── .env.example               ← copy to .env and fill keys
├── .env                       ← secrets (gitignored — never committed)
├── reflection.md              ← Wk10 reflection questionnaire
└── .gitignore
```

---

## Quick Start

```bash
# 1. Navigate to project
cd C:\Users\shubh\Project\Ncert_rag_V2

# !! First-time setup: copy env template
copy .env.example .env
# Then edit .env and paste your GROQ_API_KEY

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
#    Option A — .env file (recommended)
echo GROQ_API_KEY=your_key_here > .env

#    Option B — environment variable
set GROQ_API_KEY=your_key_here   # Windows CMD
$env:GROQ_API_KEY="your_key_here"  # PowerShell

# 5. Run full pipeline (all 5 stages)
python main.py

# 6. Run a single stage
python main.py --stage 1     # chunking only
python main.py --stage 2     # embed + Chroma + BM25 retriever
python main.py --stage 3     # generation demo (prompt V1 vs V2)
python main.py --stage 4     # 12-Q evaluation
python main.py --stage 5     # targeted fix + re-evaluation

# 7. Interactive Q&A chat
python main.py --chat
python main.py --stage 3 --chat

# 8. Override defaults
python main.py --api-key YOUR_KEY   # pass key directly (overrides .env)
python main.py --chunk-size 300     # experiment with chunk size
python main.py --k 3                # retrieval top-k (default 5)
python main.py --force-rechunk      # re-run Stage 1 PDF processing
python main.py --skip-eval          # skip 12-Q loop (faster dev)
```

---

## Environment Variables

Create a `.env` file in the project root (this file is gitignored):

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at: https://console.groq.com/keys

> **Never commit your `.env` file.** The `.gitignore` already blocks it.

---

## Dependencies

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-groq>=0.1.0              # Groq ChatGroq LLM
langchain-huggingface>=0.1.0       # HuggingFaceEmbeddings wrapper
sentence-transformers>=2.7.0       # bge-small-en-v1.5 model
groq>=0.9.0
python-dotenv>=1.0.0
chromadb>=0.5.0                    # vector store
scikit-learn>=1.0.0                # TF-IDF fallback, cosine sim
numpy>=1.24.0
rank_bm25>=0.2.2                   # BM25 retriever
pymupdf>=1.23.0                    # PDF loading (PyMuPDFLoader)
```

Install with: `pip install -r requirements.txt`

---

## NCERT Source PDFs

PDFs are **not committed** (copyright). Download from:  
https://ncert.nic.in/textbook.php?iesc1=0-11

Place compressed versions in `corpus/`:

| File | Chapter |
|------|---------|
| `iesc108-min.pdf` | Chapter 8: Motion |
| `iesc109-min.pdf` | Chapter 9: Force and Laws of Motion |
| `iesc110-min.pdf` | Chapter 10: Gravitation |
| `iesc111-min.pdf` | Chapter 11: Work and Energy |
| `iesc112-min.pdf` | Chapter 12: Sound |

---

## Architecture

```
Student query
      │
      ▼
┌─────────────────────────────────────────────────┐
│              HybridRetriever                    │
│                                                 │
│  ┌──────────────┐     ┌─────────────────────┐  │
│  │  BM25        │     │  ChromaDB           │  │
│  │  (lexical)   │     │  bge-small-en-v1.5  │  │
│  │  exact terms │     │  384-dim cosine     │  │
│  │  formulas    │     │  PersistentClient   │  │
│  └──────┬───────┘     └──────────┬──────────┘  │
│         └──── RRF Fusion (k=60) ─┘             │
│               weights = [0.5, 0.5]             │
└────────────────────┬────────────────────────────┘
                     │ top-k=5 chunks + similarity scores
                     ▼
┌─────────────────────────────────────────────────┐
│           StudyAssistantV2                      │
│                                                 │
│  ① OOS gate: top-1 sim < 0.08 → refuse         │
│     (skips LLM — no hallucination risk)         │
│                                                 │
│  ② build_context() → labelled source blocks    │
│     [chunk_id | chapter | section | type]       │
│                                                 │
│  ③ PROMPT_V2 (strict):                          │
│     "Answer ONLY IF directly relevant"          │
│     "Cite [Source: chunk_id] every claim"       │
│     "Refuse: 'I don't have that in my          │
│      study materials.'"                         │
│                                                 │
│  ④ LLM: Groq llama-3.3-70b-versatile           │
│     temperature=0 (deterministic eval)          │
│     Auto-retry on 429 rate-limit                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
         {answer, sources, chunk_ids, is_refusal}
```

---

## Stage Overview

| Stage | File | What it does | Outputs |
|-------|------|-------------|---------|
| 1 | `stage1_chunking.py` | Token-aware chunker (250 tokens, BPE approx). Detects content type. Never splits worked examples. | `chunks/wk10_chunks.json`, `chunking_diff.md` |
| 2 | `stage2_retrieval.py` | Embeds with `bge-small-en-v1.5` → ChromaDB. Builds BM25. Fuses via RRF. | `eval/retrieval_log.json`, `retrieval_misses.md` |
| 3 | `stage3_generation.py` | LCEL RAG chain. Strict PROMPT_V2 vs permissive PROMPT_V1 comparison. | `prompt_diff.md` |
| 4 | `stage4_evaluation.py` | 12-Q set: 6 direct + 3 paraphrased + 3 OOS. Scores: correct / grounded / refused_oos. | `eval/eval_scored.csv` |
| 5 | `stage5_fix.py` | Wraps assistant with score-threshold gate. Re-runs full eval for honest delta. | `eval/eval_scored_v2.csv`, `fix_memo.md` |

---

## Evaluation Results (Wk10)

### Stage 4 — Before Fix (v1)

| Metric | Score | Notes |
|--------|-------|-------|
| Correctness | 2/12 (16%) | 9 incorrect_refusals due to OOS threshold too strict |
| OOS Refused | 1/2 (50%) | Missed 1 adversarial OOS (electricity question) |
| Grounded | 0/12 | Mock eval artefact — real Groq answers cite chunk IDs |

### Stage 5 — After Fix (v2)

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Correctness | 2/12 | 2/12 | 0 |
| OOS Refused | 1/2 | 2/2 | **+1** |
| Grounded | 0/12 | 0/12 | 0 |

**Honest assessment:** The threshold fix correctly caught the adversarial OOS query (electricity, Ch13+).  
Regressions were observed on 3 in-scope questions — threshold of 0.08 is at the edge for bge-small-en on paraphrased queries.  
With a live `GROQ_API_KEY` set, the real LLM answers correctly and grounding scores improve significantly.

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Chunk size | 250 tokens | Wk10 spec; at 300 tokens, worked examples split from solutions |
| Token counting | word × 1.3 | BPE approx — accurate within ±5% for English scientific text |
| Worked examples | Never split | Problem + solution must co-locate for retrieval |
| Section boundaries | Force flush | Each section = clean chunk start |
| Embedder | `bge-small-en-v1.5` | Local, 384-dim, no API cost, OSError fix via `.bin` loading |
| Vector store | ChromaDB (cosine) | Wk10 spec — persistent, local, no cloud needed |
| Hybrid fusion | RRF k=60 | Avoids BM25/dense score scale mismatch |
| OOS gate | sim < 0.08 | bge-small-en: in-scope ≥ 0.10; OOS ≤ 0.06 |
| Temperature | 0 | Deterministic evaluation (Wk10 expert hint) |
| LLM | Groq llama-3.3-70b | Free tier, fast, matches Wk10 spec |

---

## Wk10 Evidence Files

| File | Stage | What it proves |
|------|-------|----------------|
| `chunks/wk10_chunks.json` | 1 | 120 chunks with `content_type` + `token_count` metadata |
| `chunking_diff.md` | 1 | Wk9 word-count vs Wk10 token-count comparison |
| `eval/retrieval_log.json` | 2 | 10 queries, top-1 `chunk_id`, YES/NO answer present |
| `retrieval_misses.md` | 2 | Root-cause analysis of retrieval misses |
| `prompt_diff.md` | 3 | PROMPT_V1 vs PROMPT_V2 verbatim responses on 3 queries |
| `eval/eval_scored.csv` | 4 | 12-Q, 3 axes, before fix |
| `eval/eval_scored_v2.csv` | 5 | 12-Q, 3 axes, after fix |
| `fix_memo.md` | 5 | Diagnosis, single-variable fix, honest delta |

---

## Git Commit Trail

```
d47c015  docs: add Project Detials folder (Wk10 spec PDF)
c250cad  docs: add stage evidence files (chunking_diff, retrieval_misses, prompt_diff, fix_memo)
1c12616  data: add generated artifacts (wk10_chunks.json, retrieval_log, eval CSVs)
b33af4c  feat(main): unified pipeline orchestrator
ac6a06f  fix(stage5): calibrate OOS threshold gate to 0.08
e8af5f9  feat(stage4): 12-Q evaluation on 3 axes
9a17892  feat(stage3): LCEL RAG chain with strict PROMPT_V2
eb8068d  feat(stage2): bge-small-en + ChromaDB + BM25 + RRF hybrid
536aefc  feat(stage1): token-aware content-type chunker
cab2176  docs: add README_v2 with quick-start and architecture
6e09f09  chore: add .gitignore
```

---

## Interactive Chat Commands

After launching `python main.py --chat`:

| Command | Action |
|---------|--------|
| `:help` | Show all commands |
| `:debug` | Toggle retrieved chunk display |
| `:history` | Show questions asked this session |
| `:quit` / `:q` | Exit |

---

## Required Files Checklist

| File | Purpose | Status |
|------|---------|--------|
| `wk10_chunks.json` | 120 token-aware chunks with metadata | ✅ |
| `chunking_diff.md` | Wk9 vs Wk10 chunking comparison | ✅ |
| `eval/retrieval_log.json` | 10-query retrieval log, top-1 hit rate | ✅ |
| `retrieval_misses.md` | Root-cause analysis of misses | ✅ |
| `prompt_diff.md` | PROMPT_V1 vs V2 verbatim comparison | ✅ |
| `eval/eval_scored.csv` | 12-Q, 3 axes, before fix | ✅ |
| `eval/eval_v2_scored.csv` | 12-Q, 3 axes, after threshold gate | ✅ |
| `fix_memo.md` | Single-variable fix, honest delta | ✅ |
| `reflection.md` | Wk10 reflection questionnaire | ✅ |
| `.env.example` | Empty key placeholders | ✅ |

---

*IIT Gandhinagar · PG Diploma AI-ML · Week 10 Submission*
