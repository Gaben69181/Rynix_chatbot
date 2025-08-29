# Rynix's Chatbot (Streamlit + LangChain Ollama, Streaming)

A minimal Streamlit chatbot that talks to local Ollama models via LangChain’s ChatOllama with streaming responses, chat memory trimming, and a one-click chat summary.

Core file:

- app.py — Streamlit app with Settings sidebar, streaming chat, history trimming, and summary

Key functions (clickable in editor):

- [def get_first_secret()](app.py:40) — Resolve API keys from Streamlit secrets/env
- [def get_ollama_base_url()](app.py:58) — Detect Ollama endpoint (env OLLAMA_BASE_URL, Docker host fallback, or localhost)
- [def is_ollama_available()](app.py:71) — Pings base_url/api/tags to show reachability warning
- [def build_llm()](app.py:86) — Builds provider-specific chat model (Ollama/OpenAI/Gemini) with streaming
- [def summarize_with_llm()](app.py:143) — Summarizes the chat using the same model
- [def main()](app.py:163) — Streamlit UI, provider & model selection, chat loop, summary

Providers and models (select in sidebar):

- Ollama
  - llama3.2
  - deepseek-r1:1.5b
- OpenAI
  - gpt-4o-mini
  - gpt-4o
  - gpt-3.5-turbo
- Gemini
  - gemini-1.5-flash
  - gemini-1.5-pro

Features

- Streaming assistant output (token-by-token visualization)
- Memory trimming (configurable Max History) to keep context small
- Context size control (num_ctx)
- Advanced generation settings: Temperature, Top-p, Top-k, Max new tokens
- Summarize chat button (uses same selected model)

Prerequisites

- If using Ollama:
  - Install and run Ollama — https://ollama.com
  - Start service: Windows/macOS app/daemon or `ollama serve` (Linux)
  - Pull models:
    - `ollama pull llama3.2`
    - `ollama pull deepseek-r1:1.5b`
  - For Docker use, the app connects to host Ollama via `http://host.docker.internal:11434` by default (overrideable).
- If using OpenAI:
  - An OpenAI API key with access to the chosen model(s).
- If using Gemini:
  - A Google Generative AI key (GOOGLE_API_KEY or GEMINI_API_KEY) with access to the chosen model(s).

Environment variables

- For Ollama:
  - OLLAMA_BASE_URL (optional)
    - Local: `http://localhost:11434`
    - Docker-to-host: `http://host.docker.internal:11434`
    - Remote host: `http://<your-host-ip>:11434`
- For OpenAI:
  - OPENAI_API_KEY (required when provider=OpenAI)
- For Gemini:
  - GOOGLE_API_KEY or GEMINI_API_KEY (required when provider=Gemini)

Run locally (no Docker)

1. Install Python 3.10+ and dependencies:
   pip install -r requirements.txt
2. Set provider-specific env if needed (examples):
   - OpenAI (Windows cmd):
     set OPENAI_API_KEY=sk-...
   - Gemini (Windows cmd):
     set GOOGLE_API_KEY=AIza...
   - Ollama local (optional override):
     set OLLAMA_BASE_URL=http://localhost:11434
3. Start Streamlit:
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

- Pilih Provider di sidebar: Ollama, OpenAI, atau Gemini.
- Pilih Model sesuai Provider. Jika Ollama, pastikan model sudah di-pull di server Ollama.
- Adjust Max History dan Context Size untuk mengontrol memori dan jendela konteks (num_ctx untuk Ollama).
- Gunakan Advanced settings untuk mengatur temperature, top-p; top-k berlaku untuk Ollama/Gemini.
- Ketik pesan di chat box; respons akan streaming secara real time.
- Klik “Summarize chat” untuk ringkasan singkat percakapan dengan model yang sama.

Troubleshooting

- “Ollama not reachable” warning in sidebar:

  - Pastikan Ollama berjalan dan model sudah di-pull:
    - `ollama serve` (jika belum jalan)
    - `ollama pull llama3.2`
  - Verifikasi OLLAMA_BASE_URL:
    - Local: `http://localhost:11434`
    - Docker-to-host: `http://host.docker.internal:11434`
  - Test dari container (opsional): `curl $OLLAMA_BASE_URL/api/tags`

- “OPENAI_API_KEY not set” (provider=OpenAI):

  - Tambahkan OPENAI_API_KEY di environment atau Streamlit secrets.

- “GOOGLE_API_KEY / GEMINI_API_KEY not set” (provider=Gemini):

  - Tambahkan salah satu key di environment atau Streamlit secrets.

- Slow responses / long first token:

  - Model warm-up di permintaan pertama; berikutnya lebih cepat.
  - Kurangi “Max new tokens” dan/atau temperature; pilih model yang lebih kecil.

- GPU not used by Ollama:
  - Ollama auto-detect GPU host. Cek logs dan driver.
  - Di Windows, gunakan WSL2 untuk kompatibilitas terbaik.

License

- MIT (adapt as needed for coursework)
