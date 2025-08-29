# Rynix's Chatbot (Streamlit + LangChain Ollama, Streaming)

A minimal Streamlit chatbot that talks to local Ollama models via LangChain’s ChatOllama with streaming responses, chat memory trimming, and a one-click chat summary.

Core file:

- app.py — Streamlit app with Settings sidebar, streaming chat, history trimming, and summary

Key functions (clickable in editor):

- [def get_ollama_base_url()](app.py:26) — Detects Ollama endpoint (env OLLAMA_BASE_URL, Docker host fallback, or localhost)
- [def is_ollama_available()](app.py:39) — Pings base_url/api/tags to show reachability warning
- [def build_llm()](app.py:54) — Builds ChatOllama with streaming + advanced params (temperature, top_p, top_k, num_predict, num_ctx)
- [def summarize_with_llm()](app.py:70) — Summarizes the chat using the same model
- [def main()](app.py:90) — Streamlit UI, chat loop, history trimming, and actions

Models (dropdown by default):

- llama3.2
- deepseek-r1:1.5b

Features

- Streaming assistant output (token-by-token visualization)
- Memory trimming (configurable Max History) to keep context small
- Context size control (num_ctx)
- Advanced generation settings: Temperature, Top-p, Top-k, Max new tokens
- Summarize chat button (uses same selected model)

Prerequisites

- Ollama installed and running
  - https://ollama.com
  - Start service: Windows/macOS app/daemon or `ollama serve` (Linux)
  - Pull models:
    - `ollama pull llama3.2`
    - `ollama pull deepseek-r1:1.5b`
- For Docker use, the app connects to host Ollama via `http://host.docker.internal:11434` by default (overrideable).

Environment variables

- OLLAMA_BASE_URL — Optional. Example values:
  - Local: `http://localhost:11434`
  - Docker-to-host: `http://host.docker.internal:11434`
  - Remote host: `http://<your-host-ip>:11434`

Run locally (no Docker)

1. Install Python 3.10+ and dependencies:
   pip install -r requirements.txt
2. Start Streamlit:
   streamlit run app.py

Run with Docker (CPU default)
Execute:
powershell -Command "docker stop openlm-chat 2>$null; docker stop openlm-chat-gpu 2>$null; docker build -t openlm-streamlit .; docker run --rm --name openlm-chat -p 8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 openlm-streamlit"

Then open:
http://localhost:8501

Optional: GPU container

- A CUDA-enabled Dockerfile is provided as [Dockerfile.gpu](Dockerfile.gpu).
- Build and run (requires NVIDIA GPU, proper drivers, and Docker GPU support):
  docker build -f Dockerfile.gpu -t openlm-streamlit-gpu .
  docker run --rm --gpus all --name openlm-chat-gpu -p 8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 openlm-streamlit-gpu
- Ollama’s GPU usage is automatic on the host; the app just calls the server.

UI usage

- Select a model from the sidebar (ensure it’s pulled in Ollama).
- Adjust Max History and Context Size to control memory and prompt window.
- Use Advanced settings to tune sampling.
- Type your message in the chat box; responses stream in real time.
- Click “Summarize chat” to get a concise multi-sentence summary using the same model.

Troubleshooting

- “Ollama not reachable” warning in sidebar:

  - Ensure Ollama is running and the model is pulled:
    - `ollama serve` (if not started)
    - `ollama pull llama3.2`
  - Verify OLLAMA_BASE_URL:
    - Local: `http://localhost:11434`
    - Docker-to-host: `http://host.docker.internal:11434`
  - Test from container with curl (optional): `curl $OLLAMA_BASE_URL/api/tags`

- Slow responses / long first token:

  - Models load on first request; subsequent prompts are faster.
  - Reduce “Max new tokens” and/or temperature; try a smaller model.

- GPU not used by Ollama:

  - Ollama auto-detects host GPU. Check Ollama logs and driver installation.
  - On Windows, use WSL2 for best compatibility.

License

- MIT (adapt as needed for coursework)
