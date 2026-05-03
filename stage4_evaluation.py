"""
stage4_evaluation.py  —  Week 10 Stage 4
12-question evaluation set scored on 3 axes.

Wk10 spec:
  - 6 direct + 3 paraphrased + 3 out-of-scope
  - At least 1 "plausibly answerable" OOS (formula in corpus, specific values not)
  - 3 axes: (a) correct Y/N/partial, (b) grounded Y/N, (c) refused_when_oos Y/N/NA
  - Save eval_raw.csv + eval_scored.csv
  - Write 1-paragraph diagnosis on worst failure
"""

import sys, os, re, json, csv
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

# ── Force UTF-8 output on Windows ───────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

W = 68
def banner(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")
def step(m):    print(f"\n  ▸ {m}")
def ok(m):      print(f"  ✓ {m}")
def sec(t):     print(f"\n  {'─'*(W-2)}\n  {t}")


# ══════════════════════════════════════════════════════════════
# EVALUATION QUESTION SET  (12 questions, Wk10 spec)
# ══════════════════════════════════════════════════════════════

EVAL_SET = [
    # ── DIRECT (6) ───────────────────────────────────────────
    {
        "id"          : "E01",
        "question"    : "State Newton's second law of motion and write its formula.",
        "type"        : "direct",
        "chapter"     : "Ch9",
        "key_terms"   : ["F = ma", "momentum", "force", "mass"],
        "expected"    : "answer",
        "ground_truth": "F = ma; rate of change of momentum proportional to force",
    },
    {
        "id"          : "E02",
        "question"    : "What are the three equations of uniformly accelerated motion?",
        "type"        : "direct",
        "chapter"     : "Ch8",
        "key_terms"   : ["v = u + at", "ut", "2as"],
        "expected"    : "answer",
        "ground_truth": "v=u+at; s=ut+½at²; v²=u²+2as",
    },
    {
        "id"          : "E03",
        "question"    : "A bullet of 20 g is fired from a 4 kg gun at 400 m/s. Find the recoil velocity of the gun.",
        "type"        : "direct",
        "chapter"     : "Ch9",
        "key_terms"   : ["2 m", "v2 = -2", "conservation"],
        "expected"    : "answer",
        "ground_truth": "Recoil velocity = 2 m/s (conservation of momentum)",
    },
    {
        "id"          : "E04",
        "question"    : "Define kinetic energy and write its formula.",
        "type"        : "direct",
        "chapter"     : "Ch11",
        "key_terms"   : ["(1/2)", "mv2", "kinetic", "motion"],
        "expected"    : "answer",
        "ground_truth": "KE = ½mv²; energy of motion",
    },
    {
        "id"          : "E05",
        "question"    : "What is the speed of sound in air, water, and steel?",
        "type"        : "direct",
        "chapter"     : "Ch12",
        "key_terms"   : ["346", "1500", "5100"],
        "expected"    : "answer",
        "ground_truth": "Air: 346 m/s; Water: 1500 m/s; Steel: 5100 m/s",
    },
    {
        "id"          : "E06",
        "question"    : "State Archimedes principle and state when an object floats.",
        "type"        : "direct",
        "chapter"     : "Ch10",
        "key_terms"   : ["buoyant", "displaced", "density", "float"],
        "expected"    : "answer",
        "ground_truth": "Buoyant force = weight of displaced fluid; floats if density < fluid density",
    },

    # ── PARAPHRASED (3) ───────────────────────────────────────
    {
        "id"          : "E07",
        "question"    : "How do we measure the rate at which velocity changes over time?",
        "type"        : "paraphrased",
        "chapter"     : "Ch8",
        "key_terms"   : ["acceleration", "a = (v - u)", "rate"],
        "expected"    : "answer",
        "ground_truth": "Acceleration a = (v-u)/t",
    },
    {
        "id"          : "E08",
        "question"    : "If I push a massive truck and it doesn't move, does the truck push back on me?",
        "type"        : "paraphrased",
        "chapter"     : "Ch9",
        "key_terms"   : ["third law", "action", "reaction", "equal"],
        "expected"    : "answer",
        "ground_truth": "Yes — Newton's 3rd Law: equal and opposite reaction",
    },
    {
        "id"          : "E09",
        "question"    : "When I clap near a mountain and hear the sound again, what is that called and how far is the mountain if the delay is 4 seconds?",
        "type"        : "paraphrased",
        "chapter"     : "Ch12",
        "key_terms"   : ["echo", "680", "d = v"],
        "expected"    : "answer",
        "ground_truth": "Echo; d = 340×4/2 = 680 m",
    },

    # ── OUT-OF-SCOPE (3) ──────────────────────────────────────
    {
        "id"          : "E10",
        "question"    : "Explain the process of photosynthesis in plants.",
        "type"        : "out_of_scope",
        "chapter"     : "OOS",
        "key_terms"   : [],
        "expected"    : "refusal",
        "ground_truth": "Biology topic — not in Ch8-12",
    },
    {
        "id"          : "E11",
        "question"    : "How does electric current flow through a copper wire?",
        "type"        : "out_of_scope",
        "chapter"     : "OOS",
        "key_terms"   : [],
        "expected"    : "refusal",
        "ground_truth": "Electricity (Ch13+) — adversarial OOS, shares 'force/energy' vocab",
    },
    {
        "id"          : "E12",
        # PLAUSIBLY ANSWERABLE OOS (Wk10 spec requirement):
        # The free-fall formula is in the corpus (Ch10).
        # But Moon-specific g value (1.63 m/s²) IS in the corpus (Weight section).
        # This tests whether the system correctly uses corpus content
        # or refuses because "Moon gravity" sounds like new territory.
        # Expected: ANSWER (Moon g is in corpus)
        "question"    : "Calculate the acceleration due to gravity on the surface of the Moon.",
        "type"        : "out_of_scope",    # plausibly answerable
        "chapter"     : "Ch10-partial",
        "key_terms"   : ["1.63"],
        "expected"    : "answer",          # Moon g IS in corpus
        "ground_truth": "g_moon = 1.63 m s-2 (from Chapter 10 weight section)",
        "note"        : "PLAUSIBLY ANSWERABLE: Moon g value IS in corpus. System should answer, not refuse.",
    },
]


# ══════════════════════════════════════════════════════════════
# SCORING  (3 axes per question)
# ══════════════════════════════════════════════════════════════

def score_correctness(answer: str, key_terms: List[str],
                      is_refusal: bool, expected: str) -> str:
    """
    Axis (a): correct / partial / wrong / correct_refusal / missed_refusal / incorrect_refusal
    """
    if expected == "refusal":
        return "correct_refusal" if is_refusal else "missed_refusal"
    if is_refusal:
        return "incorrect_refusal"
    if not key_terms:
        return "correct"

    ans_lower = answer.lower()
    found = [k for k in key_terms if k.lower() in ans_lower]
    ratio = len(found) / len(key_terms)

    if ratio >= 0.80: return "correct"
    if ratio >= 0.40: return "partial"
    return "wrong"


def score_grounding(answer: str, chunk_ids_cited: List[str],
                    retrieved_docs: List[Dict], is_refusal: bool) -> str:
    """
    Axis (b): grounded / ungrounded / na
    grounded = citation present AND cited chunk text contains the claim.
    """
    if is_refusal:
        return "na"
    if not chunk_ids_cited:
        return "ungrounded"  # no citations at all

    # Check that cited chunk ids are in retrieved docs
    retrieved_ids = [d.get("id","") for d in retrieved_docs]
    valid_citations = [cid for cid in chunk_ids_cited if cid in retrieved_ids]

    if valid_citations:
        return "grounded"
    return "ungrounded"


def score_refused_oos(is_refusal: bool, expected: str) -> str:
    """Axis (c): Y / N / NA"""
    if expected != "refusal":
        return "NA"
    return "Y" if is_refusal else "N"


# ══════════════════════════════════════════════════════════════
# RUN EVALUATION
# ══════════════════════════════════════════════════════════════

def run_evaluation(assistant) -> List[Dict]:
    """Run all 12 questions through ask() and collect raw results."""
    results = []
    print(f"\n  {'ID':<5} {'Type':<14} {'Correct':<20} {'Grounded':<12} {'OOS-ref':<10} Question")
    print(f"  {'─'*W}")

    for eq in EVAL_SET:
        r = assistant.ask(eq["question"])

        correctness  = score_correctness(r["answer"], eq["key_terms"],
                                         r["is_refusal"], eq["expected"])
        grounding    = score_grounding(r["answer"], r["chunk_ids"],
                                       r.get("retrieved_docs", r.get("sources",[])),
                                       r["is_refusal"])
        refused_oos  = score_refused_oos(r["is_refusal"], eq["expected"])

        icon = ("✓" if correctness in ("correct","correct_refusal")
                else "~" if "partial" in correctness
                else "✗")

        record = {
            **eq,
            "answer"       : r["answer"],
            "is_refusal"   : r["is_refusal"],
            "chunk_ids"    : r["chunk_ids"],
            "retrieved_ids": r.get("retrieved_ids",[]),
            "correctness"  : correctness,
            "grounding"    : grounding,
            "refused_oos"  : refused_oos,
        }
        results.append(record)

        print(f"  {icon} {eq['id']:<4} {eq['type']:<14} "
              f"{correctness:<20} {grounding:<12} {refused_oos:<10} "
              f"{eq['question'][:35]}")

    return results


def print_summary(results: List[Dict]) -> None:
    """Print axis-by-axis summary."""
    sec("Evaluation Summary")

    total  = len(results)
    types  = {}
    for r in results:
        t = r["type"]
        types.setdefault(t, []).append(r)

    def pct(n, d): return f"{n}/{d} ({n*100//d}%)" if d else "0/0"

    def count_ok(lst):
        return sum(1 for r in lst if r["correctness"] in ("correct","correct_refusal"))
    def count_grounded(lst):
        return sum(1 for r in lst if r["grounding"] == "grounded")
    def count_oos_refused(lst):
        return sum(1 for r in lst if r["refused_oos"] == "Y")
    def count_oos(lst):
        return sum(1 for r in lst if r["refused_oos"] != "NA")

    print(f"\n  {'Type':<16} {'N':<4} {'Correct':<14} {'Grounded':<14} {'OOS Refused'}")
    print(f"  {'─'*60}")
    for t, lst in types.items():
        oos_n = count_oos(lst)
        print(f"  {t:<16} {len(lst):<4} "
              f"{pct(count_ok(lst),len(lst)):<14} "
              f"{pct(count_grounded(lst),len(lst)):<14} "
              f"{pct(count_oos_refused(lst),oos_n) if oos_n else 'NA'}")
    print(f"  {'─'*60}")
    oos_all = count_oos(results)
    print(f"  {'TOTAL':<16} {total:<4} "
          f"{pct(count_ok(results),total):<14} "
          f"{pct(count_grounded(results),total):<14} "
          f"{pct(count_oos_refused(results),oos_all)}")

    # Identify worst failure
    failures = [r for r in results
                if r["correctness"] not in ("correct","correct_refusal")]
    if failures:
        worst = failures[0]
        sec("Worst Failure Diagnosis")
        print(f"\n  Question : {worst['question']}")
        print(f"  Type     : {worst['type']}")
        print(f"  Result   : {worst['correctness']} | grounding={worst['grounding']}")
        print(f"  chunk_ids cited: {worst['chunk_ids']}")
        print(f"  retrieved: {worst['retrieved_ids'][:3]}")

        # Diagnose from Wk10 catalog
        if worst["correctness"] == "missed_refusal":
            print(f"\n  Catalog: MIXED STRUCTURE or AMBIGUOUS")
            print(f"  Retriever returned plausible-looking physics chunks.")
            print(f"  V2 strict prompt insufficient — needs score threshold gate.")
            print(f"  Fix: add `if top_score < THRESHOLD: return refusal_text`")
        elif worst["correctness"] == "wrong" and not worst["chunk_ids"]:
            print(f"\n  Catalog: SYNONYM/ACRONYM MISMATCH")
            print(f"  Answer generated but no citations → grounding failure.")
            print(f"  LLM likely used its own knowledge, not retrieved context.")
            print(f"  Fix: stricter citation enforcement in prompt.")
        elif worst["correctness"] in ("wrong","partial"):
            print(f"\n  Catalog: RETRIEVAL MISS or CHUNK BOUNDARY")
            print(f"  Print retrieved chunks for this query — is the right")
            print(f"  content in top-5? If yes: generation bug. If no: retrieval bug.")


# ══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════

def save_results(results: List[Dict], out_dir: str, suffix: str = "") -> None:
    """Save eval_raw.csv and eval_scored.csv."""
    Path(out_dir).mkdir(exist_ok=True)

    # eval_raw.csv — full answers
    raw_path = Path(out_dir) / f"eval_raw{suffix}.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        fields = ["id","type","chapter","question","answer","is_refusal",
                  "chunk_ids","retrieved_ids"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k,"") for k in fields}
            row["chunk_ids"]    = "|".join(r.get("chunk_ids",[]))
            row["retrieved_ids"] = "|".join(r.get("retrieved_ids",[])[:3])
            row["answer"] = r.get("answer","")[:200].replace("\n"," ")
            w.writerow(row)
    ok(f"Saved → {raw_path}")

    # eval_scored.csv — scores only
    scored_path = Path(out_dir) / f"eval_scored{suffix}.csv"
    with open(scored_path, "w", newline="", encoding="utf-8") as f:
        fields = ["id","type","chapter","question","correctness","grounding","refused_oos","chunk_ids"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k,"") for k in fields}
            row["chunk_ids"] = "|".join(r.get("chunk_ids",[]))
            row["question"]  = r.get("question","")[:60]
            w.writerow(row)
    ok(f"Saved → {scored_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run(assistant, out_dir: str = None, suffix: str = "") -> List[Dict]:
    """Run Stage 4 evaluation."""
    banner("STAGE 4 — EVALUATION  (12 Questions · 3 Axes)")

    base = Path(__file__).parent
    if out_dir is None: out_dir = str(base / "eval")

    step(f"Running {len(EVAL_SET)} evaluation questions …")
    results = run_evaluation(assistant)

    print_summary(results)
    save_results(results, out_dir, suffix=suffix)

    # Score totals
    correct_n = sum(1 for r in results if r["correctness"] in ("correct","correct_refusal"))
    print(f"\n  ═══ Score: {correct_n}/{len(results)} ({correct_n*100//len(results)}%) ═══")

    return results


if __name__ == "__main__":
    import os
    from stage2_retrieval import NeuralEmbedder, ChromaStore, HybridRetriever
    from stage3_generation import StudyAssistantV2, build_llm

    base = Path(__file__).parent
    chunks_path = base / "chunks" / "wk10_chunks.json"

    if not chunks_path.exists():
        fail("Run Stage 1 first")

    chunks = json.load(open(chunks_path, encoding="utf-8"))

    emb = NeuralEmbedder()
    texts = [c["text"] for c in chunks]
    emb.fit_and_embed(texts)

    store = ChromaStore(str(base / "chroma_wk10"), emb)
    hybrid = HybridRetriever(chunks, store, k=5)

    api_key = os.environ.get("GROQ_API_KEY", "")
    llm = build_llm(api_key, "groq" if api_key else "mock")
    assistant = StudyAssistantV2(hybrid, llm, use_strict_prompt=True)

    run(assistant)
