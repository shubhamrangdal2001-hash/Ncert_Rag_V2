import re

# Read current README
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# Find the start of the block to replace
idx = content.find("*IIT Gandhinagar · PG Diploma AI-ML · Week 10 Submission*")
if idx != -1:
    content = content[:idx + len("*IIT Gandhinagar · PG Diploma AI-ML · Week 10 Submission*")] + "\n\n---\n\n## Pipeline Execution Output\n\n```text\n"
else:
    print("Could not find anchor text.")
    exit(1)

# Read output
with open("full_pipeline_output.txt", "r", encoding="utf-8") as f:
    out = f.read()

# Remove python warnings
out = re.sub(r'C:\\\\Python314[^\n]*\n[^\n]*\n\n', '', out)

content += out
content += "\n```\n"

with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed README.md")
