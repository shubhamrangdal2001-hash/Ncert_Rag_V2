"""
main.py  —  PariShiksha NCERT RAG  v2.0  (Week 10)
════════════════════════════════════════════════════
Unified pipeline runner. Executes all 5 stages in order,
or any single stage via --stage N.

USAGE
─────
  python main.py                     # full pipeline (all 5 stages)
  python main.py --stage 1           # corpus chunking only
  python main.py --stage 2           # build retrievers only
  python main.py --stage 3           # generation demo only
  python main.py --stage 4           # run 12-Q evaluation
  python main.py --stage 5           # apply fix + re-evaluate
  python main.py --chat              # all stages + interactive Q&A
  python main.py --stage 4 --chat    # eval then chat
  python main.py --api-key KEY       # real Groq API
  python main.py --provider groq     # groq (default mock)
  python main.py --chunk-size 300    # experiment (default 250 tokens)
  python main.py --k 3               # retrieval top-k (default 5)

STAGES
──────
  Stage 1 · Chunking
    Token-aware (250 tokens), content-type metadata
    (prose / worked_example / question_or_exercise / table)
    Section-boundary splits. Outputs: wk10_chunks.json, chunking_diff.md

  Stage 2 · Retrieval
    TF-IDF embedder → ChromaDB PersistentClient (cosine)
    BM25Retriever (LangChain) + EnsembleRetriever (RRF hybrid)
    Outputs: retrieval_log.json, retrieval_misses.md

  Stage 3 · Generation
    Permissive prompt V1 vs strict prompt V2 comparison
    ask() → {answer, sources, chunk_ids}  with [Source: chunk_id] citations
    Output: prompt_diff.md

  Stage 4 · Evaluation
    12 questions: 6 direct + 3 paraphrased + 3 OOS (1 plausibly answerable)
    3 axes: correctness / grounding / refused_when_oos
    Outputs: eval_raw.csv, eval_scored.csv

  Stage 5 · Targeted Fix
    Diagnosis → OOS threshold gate (top-1 similarity < 0.08 → refuse)
    Re-runs full 12-Q eval for honest delta
    Outputs: eval_v2_scored.csv, fix_memo.md

PROJECT LOCATION (Windows):  C:\\Users\\shubh\\Project\\Ncert_rag_V2
"""

import sys
import os
import json
import time
import argparse
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Force UTF-8 output on Windows (cp1252 can't encode box-drawing chars) ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── path setup ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

W = 68   # terminal width constant


def banner(title: str) -> None:
    """Full-width double-line banner."""
    pad   = (W - 2 - len(title)) // 2
    extra = (W - 2 - len(title)) % 2
    print("\n" + "═" * W)
    print("║" + " " * pad + title + " " * (pad + extra) + "║")
    print("═" * W)


def stage_header(n: int, title: str) -> None:
    """Stage divider with number."""
    label = f"  STAGE {n}  —  {title}"
    print("\n" + "─" * W)
    print(label)
    print("─" * W)


def step(msg: str) -> None:
    """In-progress step."""
    print(f"\n  ▸ {msg}")


def ok(msg: str) -> None:
    """Success line."""
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    """Warning (non-fatal)."""
    print(f"  ⚠ {msg}")


def fail(msg: str) -> None:
    """Fatal error — print and exit."""
    print(f"\n  ✗ ERROR: {msg}")
    sys.exit(1)


def section(title: str) -> None:
    """Subsection divider."""
    print(f"\n  {'─' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'─' * (W - 4)}")


def wrap(text: str, indent: int = 4) -> None:
    """Word-wrap and print with indent."""
    prefix = " " * indent
    print(textwrap.fill(
        text, width=W - indent,
        initial_indent=prefix,
        subsequent_indent=prefix,
    ))


def print_startup_info(args: argparse.Namespace) -> None:
    """Configuration summary at startup."""
    banner("PariShiksha  NCERT RAG  v2.0  — Week 10")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Stage    : {'all' if not args.stage else args.stage}")
    key_present = bool(args.api_key or os.environ.get("GROQ_API_KEY"))
    key_status  = "key found" if key_present else "⚠ GROQ_API_KEY not set!"
    print(f"  LLM      : Groq — llama-3.1-8b-instant  ({key_status})")
    print(f"  Chunks   : target={args.chunk_size} tokens | overlap=40")
    print(f"  Retrieval: top-k={args.k} | hybrid (BM25 + dense) | RRF")
    print(f"  Agentic  : {'yes' if args.agentic else 'no'}")
    print(f"  Chroma   : {PROJECT_ROOT / 'chroma_wk10'}")
    print(f"  Chat     : {'yes' if args.chat else 'no'}")
    print()


# ══════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    argparse.ArgumentParser — standard library.
    add_argument():
      --stage      int, choices 1-5; omit = run all
      --chat       store_true flag; drop into interactive Q&A
      --api-key    string; Gemini/Claude API key
      --provider   string; gemini or anthropic (default mock)
      --chunk-size int; token target per chunk (default 250)
      --k          int; top-k for retrieval (default 5)
      --skip-eval  store_true; skip 12-Q loop for dev speed
    """
    p = argparse.ArgumentParser(
        description="PariShiksha NCERT RAG v2.0 — Week 10 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Quick-start examples:
          python main.py                       run all 5 stages
          python main.py --stage 1             chunking only
          python main.py --stage 2             retrieval only
          python main.py --stage 3             generation demo
          python main.py --stage 4             12-Q evaluation
          python main.py --stage 5             targeted fix
          python main.py --chat                full pipeline + interactive chat
          python main.py --api-key KEY         override GROQ_API_KEY from .env
          python main.py --chunk-size 300 --k 3   (experiment)
        """)
    )
    p.add_argument("--stage",      type=int, choices=[1, 2, 3, 4, 5],
                   help="Run single stage (1-5). Omit to run all.")
    p.add_argument("--chat",       action="store_true",
                   help="After pipeline, enter interactive Q&A.")
    p.add_argument("--api-key",    type=str, default="",
                   dest="api_key", help="Groq API key (overrides GROQ_API_KEY in .env).")
    p.add_argument("--chunk-size",    type=int, default=250,
                   dest="chunk_size", help="Token target per chunk (default 250).")
    p.add_argument("--k",             type=int, default=5,
                   help="Top-k for retrieval (default 5).")
    p.add_argument("--skip-eval",     action="store_true", dest="skip_eval",
                   help="Skip 12-Q eval loop (faster dev iteration).")
    p.add_argument("--force-rechunk", action="store_true", dest="force_rechunk",
                   help="Re-run Stage 1 PDF chunking even if wk10_chunks.json exists.")
    p.add_argument("--agentic", action="store_true",
                   help="Use AgenticStudyAssistant for planning + iterative retrieval.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# STAGE RUNNERS
# Each function: imports lazily, runs the stage, returns its output.
# Lazy imports mean --stage 1 never loads Chroma or LangChain.
# ══════════════════════════════════════════════════════════════

def run_stage1(chunk_size: int = 250, force: bool = False) -> List[Dict]:
    """
    Stage 1 — Token-aware content-type chunking.

    If wk10_chunks.json already exists and force=False, loads from disk
    instead of re-running PyMuPDF on the PDFs (avoids memory crashes).
    Pass --force-rechunk to always re-process.

    Returns: list of chunk dicts.
    Saves: wk10_chunks.json, chunking_diff.md
    """
    stage_header(1, "TOKEN-AWARE CONTENT-TYPE CHUNKING")

    chunks_path = PROJECT_ROOT / "chunks" / "wk10_chunks.json"

    # ── Fast path: load from disk if already chunked ──────────────────
    if chunks_path.exists() and not force:
        ok(f"wk10_chunks.json found — loading from disk (use --force-rechunk to re-process)")
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        ok(f"Stage 1 complete: {len(chunks)} chunks (cached)")
        return chunks

    # ── Full path: run PyMuPDF + chunker ──────────────────────────────
    step("Importing stage1_chunking …")
    try:
        from stage1_chunking import run as _run, TARGET_TOKENS
    except ImportError as e:
        fail(f"Cannot import stage1_chunking: {e}")

    # Patch chunk size if different from default
    import stage1_chunking as s1
    original_target = s1.TARGET_TOKENS
    if chunk_size != s1.TARGET_TOKENS:
        s1.TARGET_TOKENS = chunk_size
        step(f"Chunk size overridden: {original_target} → {chunk_size} tokens")

    chunks = _run(out_dir=str(PROJECT_ROOT / "chunks"))

    if chunk_size != original_target:
        s1.TARGET_TOKENS = original_target  # restore

    ok(f"Stage 1 complete: {len(chunks)} chunks")
    return chunks


def run_stage2(chunks: List[Dict], k: int = 5):
    """
    Stage 2 — Embed + Chroma + BM25 + Hybrid retriever.

    Returns: HybridRetriever instance.
    Saves: retrieval_log.json, retrieval_misses.md
    """
    stage_header(2, "CHROMA VECTOR STORE + BM25 + HYBRID RETRIEVAL")

    step("Importing stage2_retrieval …")
    try:
        from stage2_retrieval import (
            NeuralEmbedder, ChromaStore, HybridRetriever,
            run_retrieval_log, generate_retrieval_misses_md,
        )
    except ImportError as e:
        fail(f"Cannot import stage2_retrieval: {e}")

    chroma_path = str(PROJECT_ROOT / "chroma_wk10")
    eval_dir    = str(PROJECT_ROOT / "eval")
    Path(eval_dir).mkdir(exist_ok=True)

    # Build embedder
    section("2A  Neural Embedder")
    step("Creating NeuralEmbedder (HuggingFace bge-small-en-v1.5) …")
    print("      (first run downloads ~200MB model — please wait 2-5 min) ", flush=True)
    emb   = NeuralEmbedder()
    texts = [c["text"] for c in chunks]
    step("Fitting embedder on corpus …")
    emb.fit_and_embed(texts)
    ok(f"Embedder fitted: {emb._dim} vocab dimensions")

    # Build Chroma
    section("2B  ChromaDB PersistentClient")
    store = ChromaStore(chroma_path, emb)
    if store.is_populated():
        ok(f"Chroma already populated ({store.collection.count()} docs) — skipping re-embed")
    else:
        step("Indexing chunks into Chroma …")
        store.index_chunks(chunks)
    ok(f"Chroma: {store.collection.count()} docs | path: {chroma_path}")

    # Build hybrid retriever
    section("2C  Hybrid Retriever (BM25 + Dense → EnsembleRetriever RRF)")
    hybrid = HybridRetriever(chunks, store, k=k)
    ok(f"Hybrid retriever ready | k={k}")

    # Retrieval comparison
    section("Retriever Comparison (3 probe queries)")
    probes = [
        ("What is F = ma?",              "formula — BM25 advantage"),
        ("How does velocity change?",    "paraphrase — Dense advantage"),
        ("Why does wood float in water?","conceptual — Dense advantage"),
    ]
    print(f"\n  {'Query':<38} {'Dense top-1':<28} {'BM25 top-1'}")
    print(f"  {'─' * (W - 2)}")
    for q, note in probes:
        d = hybrid.retrieve_dense_only(q)
        b = hybrid.retrieve_bm25_only(q)
        d1 = d[0]["section"][:25] if d else "—"
        b1 = b[0]["section"][:25] if b else "—"
        print(f"  {q[:36]:<38} {d1:<28} {b1}")
        print(f"  {'  ↳ ' + note}")

    # Retrieval log (10 questions)
    section("Retrieval Log (10 eval questions)")
    log, misses = run_retrieval_log(hybrid, chunks)

    # Save outputs
    log_path = Path(eval_dir) / "retrieval_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    ok(f"Saved → {log_path}")

    miss_md   = generate_retrieval_misses_md(misses)
    miss_path = PROJECT_ROOT / "retrieval_misses.md"
    with open(miss_path, "w", encoding="utf-8") as f:
        f.write(miss_md)
    ok(f"Saved → {miss_path}")

    ok(f"Stage 2 complete: hit rate {sum(1 for l in log if l['answer_present'])}/10")
    return hybrid


def run_stage3(retriever, api_key: str = "", agentic: bool = False):
    """
    Stage 3 — Grounded generation: prompt comparison + demo answers.
    Returns: StudyAssistantV2 instance.
    Saves: prompt_diff.md
    """
    stage_header(3, "GROUNDED GENERATION  (Strict Prompt + Citations)")

    step("Importing stage3_generation …")
    try:
        from stage3_generation import (
            StudyAssistantV2,
            AgenticStudyAssistant,
            build_llm,
            run_prompt_diff,
        )
    except ImportError as e:
        fail(f"Cannot import stage3_generation: {e}")

    # Build LLM — reads GROQ_API_KEY from env if api_key not passed
    step("Building LLM …")
    llm = build_llm(api_key)

    # Build strict assistant
    step("Building StudyAssistantV2 (V2 strict prompt) …")
    assistant_cls = AgenticStudyAssistant if agentic else StudyAssistantV2
    assistant = assistant_cls(retriever, llm, k=5, use_strict_prompt=True)
    ok(f"{assistant_cls.__name__} ready")

    # Prompt diff
    section("Prompt V1 vs V2 Comparison")
    diff_md   = run_prompt_diff(retriever, api_key)
    diff_path = PROJECT_ROOT / "prompt_diff.md"
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(diff_md)
    ok(f"Saved → {diff_path}")

    # Demo answers
    section("Demo: 6 questions (direct + paraphrase + OOS)")
    demo_qs = [
        ("What is Newton's second law?",             "Direct"),
        ("How much does a 10 kg object weigh on Moon?", "Calculation"),
        ("Why does dust fly out when carpet is beaten?","Conceptual"),
        ("How do we measure how fast velocity changes?","Paraphrased"),
        ("Explain how photosynthesis works.",          "OOS → must refuse"),
        ("How does electricity flow in a wire?",       "Adversarial OOS"),
    ]

    for q, label in demo_qs:
        print(f"\n  ── [{label}]  {q[:55]}")
        r = assistant.ask(q)
        status = "⊘ REFUSED" if r["is_refusal"] else "→ ANSWERED"
        answer_short = r["answer"][:120].replace("\n", " ")
        print(f"  {status}: {answer_short}{'…' if len(r['answer'])>120 else ''}")
        if r["chunk_ids"]:
            print(f"  Citations: {r['chunk_ids'][:2]}")

    ok("Stage 3 complete")
    return assistant


def run_stage4(assistant, skip: bool = False):
    """
    Stage 4 — 12-question evaluation on 3 axes.

    Returns: list of result dicts (v1 results, before the fix).
    Saves: eval_raw.csv, eval_scored.csv
    """
    stage_header(4, "EVALUATION  (12 Questions · 3 Axes)")

    if skip:
        warn("--skip-eval set — skipping evaluation loop")
        warn("Remove --skip-eval to see full results")
        return []

    step("Importing stage4_evaluation …")
    try:
        from stage4_evaluation import run as _run_eval, EVAL_SET
    except ImportError as e:
        fail(f"Cannot import stage4_evaluation: {e}")

    step(f"Running {len(EVAL_SET)} questions …")
    results = _run_eval(
        assistant,
        out_dir=str(PROJECT_ROOT / "eval"),
        suffix="",
    )

    ok(f"Stage 4 complete")
    return results


def run_stage5(assistant, v1_results: List[Dict]):
    """
    Stage 5 — One targeted fix + re-evaluation.

    Returns: v2 results list.
    Saves: eval_v2_scored.csv, fix_memo.md
    """
    stage_header(5, "TARGETED FIX  (OOS Threshold Gate)")

    if not v1_results:
        warn("No Stage 4 results to compare against — running Stage 4 first")
        v1_results = run_stage4(assistant)

    step("Importing stage5_fix …")
    try:
        from stage5_fix import run as _run_fix
    except ImportError as e:
        fail(f"Cannot import stage5_fix: {e}")

    v2_results = _run_fix(
        assistant,
        v1_results,
        out_dir=str(PROJECT_ROOT / "eval"),
    )

    ok("Stage 5 complete")
    return v2_results


# ══════════════════════════════════════════════════════════════
# INTERACTIVE CHAT
# ══════════════════════════════════════════════════════════════

def run_chat(assistant) -> None:
    """
    Interactive Q&A REPL.

    Commands:
      :quit / :q       exit
      :debug           toggle retrieved chunk display
      :history         show session questions
      :score           show current eval score if available
      :help            show commands

    The while True loop:
      1. input() blocks until Enter
      2. Parse commands (prefixed with :)
      3. Otherwise → system.ask(question) → print answer
      4. Break on :quit or EOF/Ctrl-C
    """
    banner("INTERACTIVE Q&A  —  NCERT Class 9 Physics  (Chapters 8–12)")

    wrap("Ask any physics question. Type :help for commands. "
         "The assistant answers from NCERT content only "
         "and refuses out-of-scope questions.", indent=2)
    print()
    print("  Topics: Motion · Force · Gravitation · Work/Energy · Sound")
    print()

    history = []
    debug   = False

    while True:
        try:
            prompt_label = "[debug] " if debug else ""
            raw = input(f"  {prompt_label}You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────
        cmd = raw.lower()

        if cmd in (":quit", ":q", "quit", "exit"):
            print("\n  Goodbye!\n")
            break

        if cmd == ":help":
            print("\n  Commands:")
            print("    :quit     exit")
            print("    :debug    toggle retrieved chunk display")
            print("    :history  show questions asked this session")
            print("    :help     show this message\n")
            continue

        if cmd == ":debug":
            debug = not debug
            print(f"\n  Debug mode: {'ON — will show retrieved chunks' if debug else 'OFF'}\n")
            continue

        if cmd == ":history":
            if not history:
                print("\n  No questions yet.\n")
            else:
                print(f"\n  Session history ({len(history)}):")
                for i, q in enumerate(history, 1):
                    print(f"    {i}. {q}")
                print()
            continue

        # ── Answer ────────────────────────────────────────────
        history.append(raw)
        t0 = time.time()

        result = assistant.ask(raw)
        elapsed = time.time() - t0

        print()
        if result["is_refusal"]:
            print(f"  ⊘  {result['answer']}")
        else:
            print("  Assistant ›")
            for line in result["answer"].split("\n"):
                print(f"    {line}")

            if result["sources"]:
                print(f"\n  Sources ({len(result['sources'])}):")
                for s in result["sources"][:2]:
                    print(f"    [{s['chunk_id']}]  {s['chapter']} › {s['section']}")

        if debug and result["retrieved_ids"]:
            print(f"\n  ── Retrieved chunks (debug) ──────────────")
            for rid in result["retrieved_ids"][:3]:
                print(f"    {rid}")

        print(f"\n  ({elapsed:.2f}s | {result['n_retrieved']} chunks retrieved)\n")


# ══════════════════════════════════════════════════════════════
# LAZY CHUNK LOADER
# ══════════════════════════════════════════════════════════════

def load_chunks_from_disk() -> List[Dict]:
    """
    Load wk10_chunks.json if Stage 1 was already run.
    Called when --stage 2/3/4/5 skips Stage 1.
    """
    path = PROJECT_ROOT / "chunks" / "wk10_chunks.json"
    if not path.exists():
        fail(
            f"wk10_chunks.json not found.\n"
            f"  Run Stage 1 first: python main.py --stage 1"
        )
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    ok(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def rebuild_retriever(chunks: List[Dict], k: int = 5):
    """
    Rebuild the hybrid retriever from existing Chroma collection.
    Called when --stage 3/4/5 skips Stage 2.
    """
    step("Rebuilding retriever from existing Chroma collection …")
    from stage2_retrieval import NeuralEmbedder, ChromaStore, HybridRetriever

    emb   = NeuralEmbedder()

    chroma_path = str(PROJECT_ROOT / "chroma_wk10")
    store = ChromaStore(chroma_path, emb)

    if not store.is_populated():
        step("Chroma empty — indexing now …")
        store.index_chunks(chunks)

    hybrid = HybridRetriever(chunks, store, k=k)
    ok(f"Retriever ready ({store.collection.count()} docs, k={k})")
    return hybrid


def rebuild_assistant(retriever, api_key: str = "", agentic: bool = False):
    """
    Rebuild the StudyAssistantV2 when Stage 3 was not run.
    Called when --stage 4/5 or --chat skips Stage 3.
    """
    step("Rebuilding StudyAssistantV2 …")
    from stage3_generation import StudyAssistantV2, AgenticStudyAssistant, build_llm

    llm = build_llm(api_key)  # reads GROQ_API_KEY from env if not passed
    assistant_cls = AgenticStudyAssistant if agentic else StudyAssistantV2
    assistant = assistant_cls(retriever, llm, k=5, use_strict_prompt=True)
    ok(f"{assistant_cls.__name__} ready")
    return assistant


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Main orchestrator — threads state between stages.

    State objects:
      chunks    : List[Dict]          from Stage 1
      retriever : HybridRetriever     from Stage 2
      assistant : StudyAssistantV2    from Stage 3
      v1_results: List[Dict]          from Stage 4
      v2_results: List[Dict]          from Stage 5

    Design: each stage checks whether its prerequisite state
    already exists. If not: loads from disk or rebuilds.
    This means --stage 3 works without re-running Stages 1–2.
    """
    print_startup_info(args)

    # Resolve API key: CLI > env vars > empty (mock)
    api_key = (args.api_key
               or os.environ.get("GROQ_API_KEY", ""))

    run_all   = args.stage is None
    chunks    = None
    retriever = None
    assistant = None
    v1_results = []
    v2_results = []

    # ──────── STAGE 1 ────────────────────────────────────────
    if run_all or args.stage == 1:
        chunks = run_stage1(chunk_size=args.chunk_size, force=args.force_rechunk)

    # ──────── STAGE 2 ────────────────────────────────────────
    if run_all or args.stage == 2:
        if chunks is None:
            step("Stage 1 not run — loading chunks from disk …")
            chunks = load_chunks_from_disk()
        retriever = run_stage2(chunks, k=args.k)

    # ──────── STAGE 3 ────────────────────────────────────────
    if run_all or args.stage == 3:
        if chunks is None:
            chunks = load_chunks_from_disk()
        if retriever is None:
            retriever = rebuild_retriever(chunks, k=args.k)
        assistant = run_stage3(retriever, api_key=api_key, agentic=args.agentic)

    # ──────── STAGE 4 ────────────────────────────────────────
    if run_all or args.stage == 4:
        if chunks is None:
            chunks = load_chunks_from_disk()
        if retriever is None:
            retriever = rebuild_retriever(chunks, k=args.k)
        if assistant is None:
            assistant = rebuild_assistant(retriever, api_key, agentic=args.agentic)
        v1_results = run_stage4(assistant, skip=args.skip_eval)

    # ──────── STAGE 5 ────────────────────────────────────────
    if run_all or args.stage == 5:
        if chunks is None:
            chunks = load_chunks_from_disk()
        if retriever is None:
            retriever = rebuild_retriever(chunks, k=args.k)
        if assistant is None:
            assistant = rebuild_assistant(retriever, api_key, agentic=args.agentic)
        if not v1_results:
            step("No Stage 4 results — running Stage 4 first …")
            v1_results = run_stage4(assistant, skip=args.skip_eval)
        if not args.skip_eval:
            v2_results = run_stage5(assistant, v1_results)

    # ──────── CHAT ───────────────────────────────────────────
    if args.chat:
        if chunks is None:
            chunks = load_chunks_from_disk()
        if retriever is None:
            retriever = rebuild_retriever(chunks, k=args.k)
        if assistant is None:
            assistant = rebuild_assistant(retriever, api_key, agentic=args.agentic)
        run_chat(assistant)

    # ──────── COMPLETION BANNER ──────────────────────────────
    if not args.chat:
        banner("PIPELINE COMPLETE")
        print(f"  Finished : {datetime.now().strftime('%H:%M:%S')}")

        # Show output file summary
        outputs = {
            "wk10_chunks.json"    : PROJECT_ROOT / "chunks"   / "wk10_chunks.json",
            "chunking_diff.md"    : PROJECT_ROOT                / "chunking_diff.md",
            "retrieval_log.json"  : PROJECT_ROOT / "eval"     / "retrieval_log.json",
            "retrieval_misses.md" : PROJECT_ROOT                / "retrieval_misses.md",
            "prompt_diff.md"      : PROJECT_ROOT                / "prompt_diff.md",
            "eval_scored.csv"     : PROJECT_ROOT / "eval"     / "eval_scored.csv",
            "eval_v2_scored.csv"  : PROJECT_ROOT / "eval"     / "eval_v2_scored.csv",
            "fix_memo.md"         : PROJECT_ROOT                / "fix_memo.md",
        }

        section("Output Files")
        for name, path in outputs.items():
            if path.exists():
                size = path.stat().st_size
                print(f"  ✓  {name:<26}  ({size:,} bytes)")
            else:
                print(f"  ○  {name:<26}  (not yet generated)")
        print()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
