"""
stage3_generation.py  —  Week 10 Stage 3
Grounded answer generation with strict prompt + citation.

Wk10 spec:
  - Permissive prompt first → show hallucination
  - Strict prompt → clean refusals + [Source: chunk_id] citations
  - ask(question) returns {answer, sources, chunk_ids}
  - Save prompt_diff.md with verbatim before/after
  - temperature=0 always for evaluation
"""

import sys, os, re, json, time
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

# ── Force UTF-8 output on Windows ───────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_groq import ChatGroq

W = 68
def banner(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")
def step(m):    print(f"\n  ▸ {m}")
def ok(m):      print(f"  ✓ {m}")
def sec(t):     print(f"\n  {'─'*(W-2)}\n  {t}")


# ══════════════════════════════════════════════════════════════
# PROMPT DEFINITIONS  —  V1 (permissive) and V2 (strict)
# ══════════════════════════════════════════════════════════════

# ── V1: Permissive ────────────────────────────────────────────
# Problem: "Answer the question using the context" is a PREFERENCE.
# The LLM interprets it as "prefer context but use general knowledge
# if context is thin." On out-of-scope queries with plausible-looking
# context, this generates confident wrong answers.
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system",
     """You are a study assistant for NCERT Class 9 Science.
Answer the student's question using the provided context.

Context:
{context}"""),
    ("human", "{question}"),
])

# ── V2: Strict ────────────────────────────────────────────────
# Two changes from V1:
#   1. "ONLY IF directly relevant" — conditional forces relevance check
#      before generation begins, not as a preference after
#   2. Explicit refusal text (prescribed) — enables is_refusal flag:
#      'I don't have that in my study materials' in answer.lower()
#   3. Citation instruction: "[Source: chunk_id]" after every claim
#      This is the teacher's ask AND the engineer's debugging tool
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system",
     """You are a precise study assistant for NCERT Class 9 Science (Chapters 8–12: Physics).

STRICT RULES:
1. Answer ONLY if the retrieved context directly and specifically answers the question.
2. After every factual claim, cite the source in square brackets: [Source: chunk_id].
3. If the answer is NOT present in the context, reply EXACTLY:
   "I don't have that in my study materials. Please refer to the relevant chapter."
4. Never use knowledge outside the retrieved context.
5. For calculations: show every step with units.
6. For definitions: use the textbook's own language where possible.

Retrieved Context:
{context}"""),
    ("human", "Question: {question}"),
])

REFUSAL_TEXT = "i don't have that in my study materials"
REFUSAL_RESPONSE = "I don't have that in my study materials. Please refer to the relevant chapter."


def build_context(retrieved: List[Dict]) -> str:
    """
    Format retrieved chunks into labelled context block.
    Each chunk gets a [chunk_id | chapter | section | type] header.
    This header text feeds directly into the [Source: chunk_id] citations.
    """
    parts = []
    for i, r in enumerate(retrieved, 1):
        cid  = r.get("id", f"chunk_{i}")
        ch   = r.get("chapter", "")
        sec  = r.get("section", "")
        ct   = r.get("content_type", "")
        sim  = r.get("similarity", 0)
        hdr  = f"[{cid} | {ch} | {sec} | {ct} | sim={sim:.3f}]"
        parts.append(f"{hdr}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
# LLM  — Groq (llama-3.1-8b-instant)
# ══════════════════════════════════════════════════════════════

def build_llm(api_key: str = ""):
    """
    Build ChatGroq LLM.
    api_key: Groq API key (reads GROQ_API_KEY env var if not passed).
    Raises RuntimeError if no key is available.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "  Set it in .env or pass --api-key KEY\n"
            "  Get a free key at: https://console.groq.com/keys"
        )
    llm = ChatGroq(
        model       = "llama-3.1-8b-instant",
        temperature = 0,
        api_key     = key,
    )
    ok("LLM: Groq — llama-3.1-8b-instant")
    return llm


# ══════════════════════════════════════════════════════════════
# ANSWER SYSTEM
# ══════════════════════════════════════════════════════════════

class StudyAssistantV2:
    """
    Week 10 Study Assistant.
    ask(question) → {answer, sources, chunk_ids, is_refusal}

    LCEL chain:
      RunnableParallel({
        context:  retrieve → format,
        question: passthrough
      })
      | PROMPT_V2
      | llm
      | StrOutputParser()
    """

    def __init__(self, retriever, llm, k: int = 5,
                 use_strict_prompt: bool = True):
        self.retriever   = retriever
        self.llm         = llm
        self.k           = k
        self.prompt      = PROMPT_V2 if use_strict_prompt else PROMPT_V1
        self._last_docs  = []   # store for chunk_ids extraction

        # Build LCEL chain
        def retrieve_and_store(question: str) -> List[Dict]:
            docs = self.retriever.retrieve(question)
            self._last_docs = docs
            return docs

        self.chain = (
            RunnableParallel({
                "context" : RunnableLambda(retrieve_and_store)
                            | RunnableLambda(build_context),
                "question": RunnablePassthrough(),
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str, _retries: int = 3) -> Dict[str, Any]:
        """
        Full RAG pipeline with automatic Groq rate-limit retry.
        Returns {answer, sources, chunk_ids, is_refusal, retrieved_docs}.
        """
        for attempt in range(1, _retries + 1):
            try:
                answer_text = self.chain.invoke(question)
                break  # success
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    # Parse wait time from error message (e.g. "try again in 20m58s")
                    wait = 60  # default 60s
                    m = re.search(r'try again in (\d+)m([\d.]+)s', err_str)
                    if m:
                        wait = int(m.group(1)) * 60 + int(float(m.group(2))) + 5
                    wait = min(wait, 120)  # cap at 2 minutes
                    if attempt < _retries:
                        print(f"  ⚠ Groq rate limit hit — waiting {wait}s (attempt {attempt}/{_retries}) …")
                        time.sleep(wait)
                        continue
                # Non-rate-limit error or out of retries — fail gracefully
                print(f"  ⚠ LLM error on attempt {attempt}: {e}")
                answer_text = "I don't have that in my study materials. Please refer to the relevant chapter."
                break

        # Extract chunk_ids from [Source: X] citations in the answer
        cited_ids = re.findall(r'\[Source:\s*([^\]]+)\]', answer_text)
        cited_ids = list(dict.fromkeys(cited_ids))  # deduplicate, preserve order

        # All retrieved chunk ids (not just cited)
        all_chunk_ids = [d.get("id","") for d in self._last_docs]

        # Source metadata for cited chunks
        sources = []
        for doc in self._last_docs:
            if doc.get("id","") in cited_ids:
                sources.append({
                    "chunk_id"     : doc.get("id",""),
                    "chapter"      : doc.get("chapter",""),
                    "section"      : doc.get("section",""),
                    "content_type" : doc.get("content_type",""),
                    "similarity"   : doc.get("similarity", 0),
                })

        is_refusal = REFUSAL_TEXT in answer_text.lower()

        return {
            "question"      : question,
            "answer"        : answer_text,
            "sources"       : sources,
            "chunk_ids"     : cited_ids,        # cited in answer
            "retrieved_ids" : all_chunk_ids,    # all retrieved (for debugging)
            "is_refusal"    : is_refusal,
            "n_retrieved"   : len(self._last_docs),
        }

    def print_result(self, result: Dict) -> None:
        """Pretty-print an ask() result."""
        print(f"\n  Q: {result['question']}")
        status = "⊘ REFUSAL" if result["is_refusal"] else "→ ANSWERED"
        print(f"  {status}")
        print(f"\n  Answer:")
        for line in result["answer"].split("\n"):
            print(f"    {line}")
        if result["sources"]:
            print(f"\n  Sources ({len(result['sources'])}):")
            for s in result["sources"]:
                print(f"    [{s['chunk_id']}]  {s['chapter']} › {s['section']}")
        elif result["retrieved_ids"]:
            print(f"\n  Retrieved (no citations): {result['retrieved_ids'][:3]}")


class AgenticStudyAssistant(StudyAssistantV2):
    """
    Agentic extension of StudyAssistantV2.

    Adds a light planning loop:
      1. Generate retrieval sub-queries from the user question.
      2. Retrieve for each sub-query and merge by similarity.
      3. Generate final answer from merged context.
      4. If quality checks fail, do one fallback re-plan pass.
    """

    QUERY_PLAN_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """You are a retrieval planner for an NCERT Physics RAG system.
Return up to 3 short retrieval queries (one per line) that can help answer the user question.
Rules:
- Keep each line under 12 words.
- Include formulas/keywords when relevant.
- No numbering, no bullets, no commentary."""),
        ("human", "Question: {question}")
    ])

    def __init__(self, retriever, llm, k: int = 5, use_strict_prompt: bool = True):
        super().__init__(retriever, llm, k=k, use_strict_prompt=use_strict_prompt)
        self.plan_chain = self.QUERY_PLAN_PROMPT | self.llm | StrOutputParser()

    def _parse_plan_queries(self, text: str, question: str) -> List[str]:
        lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            if ln and ln.lower() not in {q.lower() for q in cleaned}:
                cleaned.append(ln[:120])
        if question not in cleaned:
            cleaned.insert(0, question)
        return cleaned[:3]

    def _retrieve_agentic(self, question: str) -> List[Dict]:
        try:
            plan_text = self.plan_chain.invoke({"question": question})
            planned_queries = self._parse_plan_queries(plan_text, question)
        except Exception:
            planned_queries = [question]

        merged: Dict[str, Dict[str, Any]] = {}
        for pq in planned_queries:
            docs = self.retriever.retrieve(pq)
            for rank, doc in enumerate(docs, 1):
                cid = doc.get("id", "")
                if not cid:
                    continue
                score = float(doc.get("similarity", 0.0) or 0.0)
                existing = merged.get(cid)
                if (existing is None) or (score > float(existing.get("similarity", 0.0) or 0.0)):
                    doc_copy = dict(doc)
                    doc_copy["agentic_query"] = pq
                    doc_copy["agentic_rank"] = rank
                    merged[cid] = doc_copy

        ranked = sorted(
            merged.values(),
            key=lambda d: float(d.get("similarity", 0.0) or 0.0),
            reverse=True,
        )
        return ranked[:self.k]

    def _invoke_llm_with_docs(self, question: str, docs: List[Dict], _retries: int = 3) -> str:
        context = build_context(docs)
        for attempt in range(1, _retries + 1):
            try:
                return (self.prompt | self.llm | StrOutputParser()).invoke({
                    "context": context,
                    "question": question,
                })
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait = 60
                    m = re.search(r'try again in (\d+)m([\d.]+)s', err_str)
                    if m:
                        wait = int(m.group(1)) * 60 + int(float(m.group(2))) + 5
                    wait = min(wait, 120)
                    if attempt < _retries:
                        time.sleep(wait)
                        continue
                return REFUSAL_RESPONSE
        return REFUSAL_RESPONSE

    def ask(self, question: str, _retries: int = 3) -> Dict[str, Any]:
        docs = self._retrieve_agentic(question)
        self._last_docs = docs

        answer_text = self._invoke_llm_with_docs(question, docs, _retries=_retries)
        cited_ids = re.findall(r'\[Source:\s*([^\]]+)\]', answer_text)
        cited_ids = list(dict.fromkeys(cited_ids))

        # If non-refusal answer has no citations, do one fallback pass with raw query retrieval.
        if (REFUSAL_TEXT not in answer_text.lower()) and (not cited_ids):
            fallback_docs = self.retriever.retrieve(question)
            if fallback_docs:
                self._last_docs = fallback_docs
                answer_text = self._invoke_llm_with_docs(question, fallback_docs, _retries=_retries)
                cited_ids = re.findall(r'\[Source:\s*([^\]]+)\]', answer_text)
                cited_ids = list(dict.fromkeys(cited_ids))

        all_chunk_ids = [d.get("id", "") for d in self._last_docs]
        sources = []
        for doc in self._last_docs:
            if doc.get("id", "") in cited_ids:
                sources.append({
                    "chunk_id": doc.get("id", ""),
                    "chapter": doc.get("chapter", ""),
                    "section": doc.get("section", ""),
                    "content_type": doc.get("content_type", ""),
                    "similarity": doc.get("similarity", 0),
                })

        is_refusal = REFUSAL_TEXT in answer_text.lower()
        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "chunk_ids": cited_ids,
            "retrieved_ids": all_chunk_ids,
            "is_refusal": is_refusal,
            "n_retrieved": len(self._last_docs),
            "agentic": True,
        }

# ══════════════════════════════════════════════════════════════
# PROMPT DIFF DEMO  (permissive vs strict on same queries)
# ══════════════════════════════════════════════════════════════

def run_prompt_diff(retriever, api_key: str = "") -> str:
    """
    Run same 3 queries through V1 (permissive) and V2 (strict).
    Save verbatim responses to prompt_diff.md.
    """
    llm = build_llm(api_key)

    v1_system = StudyAssistantV2(retriever, llm, use_strict_prompt=False)
    v2_system = StudyAssistantV2(retriever, llm, use_strict_prompt=True)

    test_cases = [
        {
            "q"      : "What is Newton's second law of motion?",
            "type"   : "Direct in-scope",
            "note"   : "Both should answer; V2 adds citations",
        },
        {
            "q"      : "Explain how photosynthesis works in plants.",
            "type"   : "Out-of-scope (Biology)",
            "note"   : "V1 may hallucinate; V2 must refuse",
        },
        {
            "q"      : "How does electric current flow through a conductor?",
            "type"   : "Adversarial OOS (same-domain physics, not in Ch8-12)",
            "note"   : "V1 likely hallucinates; V2 should refuse",
        },
    ]

    diff_md  = "# Prompt Diff — Permissive V1 vs Strict V2\n\n"
    diff_md += "**Key change:** V1 says 'answer using context' (preference). "
    diff_md += "V2 says 'answer ONLY IF directly relevant' + explicit refusal text + "
    diff_md += "citation instruction ([Source: chunk_id]).\n\n"
    diff_md += "---\n\n"

    print(f"\n  {'─'*(W-2)}\n  Running prompt comparison (3 queries)")

    for i, tc in enumerate(test_cases, 1):
        r1 = v1_system.ask(tc["q"])
        r2 = v2_system.ask(tc["q"])

        v1_status = "HALLUCINATED" if (not r1["is_refusal"] and tc["type"] != "Direct in-scope") else ("REFUSED" if r1["is_refusal"] else "ANSWERED")
        v2_status = "✓ REFUSED" if r2["is_refusal"] else ("✓ CITED" if r2["chunk_ids"] else "ANSWERED")

        print(f"\n  Q{i}: [{tc['type']}] {tc['q'][:50]}")
        print(f"    V1 → {v1_status}")
        print(f"    V2 → {v2_status}")

        diff_md += f"## Query {i}: {tc['type']}\n\n"
        diff_md += f"**Question:** {tc['q']}\n\n"
        diff_md += f"**Note:** {tc['note']}\n\n"
        diff_md += f"### V1 Response (Permissive Prompt)\n\n"
        diff_md += f"```\n{r1['answer']}\n```\n\n"
        diff_md += f"**V1 verdict:** {'Refused' if r1['is_refusal'] else 'Answered'} | "
        diff_md += f"Citations: {r1['chunk_ids'] or 'none'}\n\n"
        diff_md += f"### V2 Response (Strict Prompt)\n\n"
        diff_md += f"```\n{r2['answer']}\n```\n\n"
        diff_md += f"**V2 verdict:** {'Refused' if r2['is_refusal'] else 'Answered'} | "
        diff_md += f"Citations: {r2['chunk_ids'] or 'none'}\n\n"
        diff_md += "---\n\n"

    diff_md += """## Analysis

### Why V2 is safer

**V1 failure on out-of-scope:** "Answer the question using the context" is
interpreted as a preference. When the retriever returns physics chunks that
look plausible (they contain words like "force", "energy", "motion"), the LLM
uses them as a jumping-off point and generates from general knowledge.

**V2 constraint:** "Answer ONLY IF the retrieved context directly and
specifically answers the question" is a conditional check. The LLM must
evaluate context relevance before deciding to generate. The prescribed
refusal text ("I don't have that in my study materials") is detectable
programmatically:

```python
is_refusal = "i don't have that in my study materials" in answer.lower()
```

### Why citations matter (beyond the teacher's ask)

Without [Source: chunk_id] in the response, a wrong answer is a black box.
With citations, every wrong answer is a 30-second diagnosis:

1. Print retrieved_ids → was the right chunk even retrieved?
2. If yes: check cited_ids → did the model cite it? → generation bug
3. If cited: check chunk text → is the information actually there? → chunk bug
4. If no: → retrieval/chunking bug
"""
    return diff_md


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run(retriever=None, api_key: str = "", out_dir: str = None) -> "StudyAssistantV2":
    """Build the StudyAssistantV2 and run Stage 3 demonstration."""
    banner("STAGE 3 — GROUNDED GENERATION  (Strict Prompt + Citations)")

    base = Path(__file__).parent
    if out_dir is None: out_dir = str(base)

    # Build LLM
    step("Building LLM …")
    llm = build_llm(api_key)

    # Build assistant (strict V2 by default)
    step("Building StudyAssistantV2 with strict prompt …")
    assistant = StudyAssistantV2(retriever, llm, k=5, use_strict_prompt=True)
    ok("StudyAssistantV2 ready")

    # ── Prompt diff demo ──────────────────────────────────────
    sec("Prompt V1 vs V2 comparison")
    diff_md = run_prompt_diff(retriever, api_key)

    diff_path = Path(out_dir) / "prompt_diff.md"
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(diff_md)
    ok(f"Saved → {diff_path}")

    # ── Demo: strict assistant on 5 questions ─────────────────
    sec("Demo: ask() with strict prompt (5 questions)")
    demo_qs = [
        ("What is Newton's second law of motion? Write the formula.", "Direct"),
        ("How much does a 10 kg object weigh on the Moon?", "Calculation"),
        ("Why does a wooden block float in water?", "Conceptual"),
        ("How is rate of velocity change measured over time?", "Paraphrased"),
        ("Explain the carbon cycle in nature.", "OOS → must refuse"),
    ]

    for q, label in demo_qs:
        print(f"\n  ── [{label}]")
        result = assistant.ask(q)
        assistant.print_result(result)

    return assistant


if __name__ == "__main__":
    import json
    from stage2_retrieval import NeuralEmbedder, ChromaStore, HybridRetriever

    base   = Path(__file__).parent
    chunks = json.load(open(base / "chunks/wk10_chunks.json", encoding="utf-8"))

    emb   = NeuralEmbedder()
    store = ChromaStore(str(base / "chroma_wk10"), emb)
    hybrid = HybridRetriever(chunks, store, k=5)

    api_key = os.environ.get("GROQ_API_KEY", "")
    run(retriever=hybrid, api_key=api_key)

