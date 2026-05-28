import re

with open("README.md", "r", encoding="utf-8") as f:
    text = f.read()

# Add Current Project Status
status_text = """## Current Project Status

**Status: Complete (v2.0)**
The NCERT Class 9 Physics Study Assistant v2.0 pipeline is fully operational.
- All 5 stages (Chunking, Retrieval, Generation, Evaluation, Fix) execute successfully.
- Integrated with `llama-3.1-8b-instant` via Groq.
- The hybrid retriever (BM25 + Dense `bge-small-en-v1.5`) successfully achieves a 70% top-1 hit rate on evaluation questions.
- The Strict Prompt and OOS Threshold Gate correctly refuse out-of-scope questions without hallucinating.
- Full execution output is appended at the bottom of this README.

---

"""
if "## Current Project Status" not in text:
    text = text.replace("## What Changed from v1.0 (Week 9)", status_text + "## What Changed from v1.0 (Week 9)")

# Ensure Pipeline Output is present at the end
out_section_header = "## Pipeline Execution Output"

# Clean output
with open("full_pipeline_output.txt", "r", encoding="utf-8") as f:
    out = f.read()

out_lines = out.splitlines()
clean_out_lines = []
for line in out_lines:
    if line.startswith("C:\\Python314") or line.strip().startswith("from pydantic.v1.fields"):
        continue
    clean_out_lines.append(line)
out = "\n".join(clean_out_lines)

# Strip existing Pipeline Output if it exists
idx = text.find(out_section_header)
if idx != -1:
    text = text[:idx].strip()
else:
    text = text.strip()

# Append clean output
text += f"\n\n{out_section_header}\n\n```text\n{out.strip()}\n```\n"

with open("README.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Applied clean changes to README.md")
