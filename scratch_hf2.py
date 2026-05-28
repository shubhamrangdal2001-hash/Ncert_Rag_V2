import time
print("importing...")
from langchain_huggingface import HuggingFaceEmbeddings
print("loading embeddings...")
t0 = time.time()
try:
    embeddings = HuggingFaceEmbeddings(
        model_name   = "BAAI/bge-small-en-v1.5",
        model_kwargs = {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": True},
    )
    print("success!", time.time() - t0)
except Exception as e:
    print("Error:", e)
