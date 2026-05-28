const sendBtn = document.getElementById("send");
const questionInput = document.getElementById("question");
const answerBox = document.getElementById("answer");
const sourcesList = document.getElementById("sources");
const modeSelect = document.getElementById("mode");
const kInput = document.getElementById("k");

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  answerBox.textContent = "Thinking...";
  sourcesList.innerHTML = "";

  try {
    const payload = {
      question,
      agentic: modeSelect.value === "agentic",
      k: Number(kInput.value || 5),
    };

    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }

    answerBox.textContent = data.answer || "No answer.";
    const sources = data.sources || [];
    if (!sources.length) {
      const li = document.createElement("li");
      li.textContent = "No citations returned.";
      sourcesList.appendChild(li);
      return;
    }

    for (const src of sources) {
      const li = document.createElement("li");
      li.textContent = `${src.chunk_id} | ${src.chapter} | ${src.section}`;
      sourcesList.appendChild(li);
    }
  } catch (err) {
    answerBox.textContent = `Error: ${err.message}`;
  }
}

sendBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    askQuestion();
  }
});
