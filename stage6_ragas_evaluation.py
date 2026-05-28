"""
stage6_ragas_evaluation.py  —  Week 10 Stage 6
Ragas Evaluation Matrix integration.

This script runs the Ragas evaluation framework on the 12-question EVAL_SET.
It calculates:
  - answer_relevancy
  - faithfulness
  - context_precision
  - context_recall
  
And saves the resulting matrix to eval/ragas_evaluation_matrix.csv.
"""

import sys, os, json
import pandas as pd
from pathlib import Path

# Fix Windows encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import our project modules
from stage2_retrieval import NeuralEmbedder, ChromaStore, HybridRetriever
from stage3_generation import StudyAssistantV2, build_llm
from stage4_evaluation import EVAL_SET

W = 68
def banner(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")
def step(m):    print(f"\n  ▸ {m}")
def ok(m):      print(f"  ✓ {m}")
def sec(t):     print(f"\n  {'─'*(W-2)}\n  {t}")

def run_ragas_evaluation(assistant, eval_set):
    """
    1. Runs the RAG pipeline to gather answers and contexts.
    2. Formats into Hugging Face Dataset.
    3. Runs Ragas evaluate.
    4. Returns a pandas DataFrame.
    """
    
    # Ragas requires lists for each column
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    step(f"Generating answers and contexts for {len(eval_set)} questions...")
    for i, eq in enumerate(eval_set, 1):
        q = eq["question"]
        gt = eq["ground_truth"]
        
        # We also pass questions labeled "refusal" as their ground truth string is available.
        # Note: Ragas metrics might penalize the LLM if the ground truth is a refusal statement
        # and the LLM correctly refuses, because Ragas is tuned for factual similarity.
        
        print(f"  [{i}/{len(eval_set)}] Q: {q[:50]}...")
        result = assistant.ask(q)
        
        ans = result["answer"]
        # Retrieved docs were stored in _last_docs inside ask()
        # We need the raw text of the contexts to pass to Ragas
        docs = assistant._last_docs if hasattr(assistant, '_last_docs') else []
        retrieved_texts = [d.get("text", "") for d in docs]
        
        questions.append(q)
        answers.append(ans)
        contexts_list.append(retrieved_texts)
        ground_truths.append(gt)

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    step("Initializing Ragas evaluators with Groq LLM and HuggingFace Embeddings...")
    eval_llm = LangchainLLMWrapper(assistant.llm)
    eval_embeddings = LangchainEmbeddingsWrapper(assistant.retriever.chroma.embedder.embeddings)
    
    metrics = [
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall
    ]
    
    step("Running Ragas evaluation metrics (this will make API calls)...")
    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False,  # Skip failed rows instead of crashing
        # Optionally add max_workers=1 to avoid hitting Groq rate limits too quickly
        # max_workers=1
    )
    
    return ragas_result.to_pandas()

def run():
    banner("STAGE 6 — RAGAS EVALUATION MATRIX")
    base = Path(__file__).parent
    
    # 1. Setup Retrieval
    step("Loading chunks and setting up Retrieval...")
    chunks_path = base / "chunks" / "wk10_chunks.json"
    if not chunks_path.exists():
        print("  ✗ chunks not found. Run Stage 1 first.")
        sys.exit(1)
        
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
        
    emb = NeuralEmbedder()
    store = ChromaStore(str(base / "chroma_wk10"), emb)
    hybrid = HybridRetriever(chunks, store, k=5)
    
    # 2. Setup LLM & Assistant
    api_key = os.environ.get("GROQ_API_KEY", "")
    llm = build_llm(api_key)
    assistant = StudyAssistantV2(hybrid, llm, use_strict_prompt=True)
    
    # 3. Run Ragas Evaluation
    df_result = run_ragas_evaluation(assistant, EVAL_SET)
    
    # 4. Save Results
    out_dir = base / "eval"
    out_dir.mkdir(exist_ok=True)
    
    csv_path = out_dir / "ragas_evaluation_matrix.csv"
    df_result.to_csv(csv_path, index=False, encoding="utf-8")
    
    ok(f"Ragas evaluation complete. Results saved to {csv_path}")
    
    # Print summary metrics
    sec("Evaluation Summary (Mean Scores)")
    print(f"  Answer Relevancy:  {df_result['answer_relevancy'].mean():.4f}")
    print(f"  Faithfulness:      {df_result['faithfulness'].mean():.4f}")
    print(f"  Context Precision: {df_result['context_precision'].mean():.4f}")
    print(f"  Context Recall:    {df_result['context_recall'].mean():.4f}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Load from .env file
    run()
