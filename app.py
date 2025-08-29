import os
from typing import List, Dict

import streamlit as st

# LangChain imports (ChatOllama prioritized from langchain_ollama)
try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore
    except Exception:
        try:
            from langchain.chat_models import ChatOllama  # type: ignore
        except Exception:
            ChatOllama = None  # type: ignore

from langchain.schema import HumanMessage, AIMessage  # type: ignore
from langchain.memory import ConversationBufferMemory  # type: ignore
from langchain.chains import LLMChain  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


def get_ollama_base_url() -> str:
    env_url = os.getenv("OLLAMA_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    try:
        in_docker = os.path.exists("/.dockerenv")
    except Exception:
        in_docker = False
    if in_docker:
        return "http://host.docker.internal:11434"
    return "http://localhost:11434"


def is_ollama_available(base_url: str) -> bool:
    if not requests:
        return True
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def clear_memory():
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)


def build_llm(model: str, temperature: float, top_p: float, top_k: int, max_new_tokens: int, num_ctx: int):
    if ChatOllama is None:
        raise RuntimeError("ChatOllama is not available. Please ensure langchain-community is installed.")
    base = get_ollama_base_url()
    return ChatOllama(
        base_url=base,
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=max_new_tokens,
        num_ctx=num_ctx,
        streaming=True,
    )


def summarize_with_llm(llm: "ChatOllama", history: List[Dict[str, str]]) -> str:
    # Build transcript
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    transcript = "\n".join(lines)
    prompt = (
        "You are a helpful assistant that summarizes chats.\n"
        "Please produce a concise, neutral summary (4-8 sentences) of the following conversation. "
        "Focus on key intents, decisions, and open questions. Avoid redundancy.\n\n"
        f"{transcript}\n\nSummary:"
    )
    try:
        res = llm.invoke(prompt)  # type: ignore
        return getattr(res, "content", str(res)).strip()
    except Exception as e:
        return f"Summary failed: {e}"


def main():
    st.set_page_config(layout="wide", page_title="My Local Chatbot", page_icon="🤖")
    st.title("Rynix's Chatbot")

    # Sidebar
    st.sidebar.header("Settings")
    model_options = ["llama3.2", "deepseek-r1:1.5b"]
    MODEL = st.sidebar.selectbox("Choose a Model", model_options, index=0)
    MAX_HISTORY = 2 
    # max_history = st.sidebar.number_input("Max History", min_value=1, max_value=10, value=2, step=1)
    CONTEXT_SIZE = 8192 
    # context_size = st.sidebar.number_input("Context Size", min_value=1024, max_value=16384, value=8192, step=1024)

    with st.sidebar.expander("Advanced settings", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
        top_k = st.slider("Top-k", 0, 200, 50, 1)
        max_new_tokens = st.slider("Max new tokens", 16, 4096, 256, 8)

    btn_cols = st.sidebar.columns(2)
    clear_clicked = btn_cols[0].button("Clear chat")
    summarize_clicked = btn_cols[1].button("Summarize chat")

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "summary" not in st.session_state:
        st.session_state.summary = ""

    # Reset on setting changes or explicit clear
    changed = False
    if "prev_context_size" not in st.session_state or st.session_state.prev_context_size != CONTEXT_SIZE:
        changed = True
    if "prev_model" not in st.session_state or st.session_state.prev_model != MODEL:
        changed = True
    for key, val in [
        ("prev_temperature", temperature),
        ("prev_top_p", top_p),
        ("prev_top_k", top_k),
        ("prev_max_new_tokens", max_new_tokens),
    ]:
        if key not in st.session_state or st.session_state[key] != val:
            changed = True
    if clear_clicked or changed:
        clear_memory()
        st.session_state.prev_context_size = CONTEXT_SIZE
        st.session_state.prev_model = MODEL
        st.session_state.prev_temperature = temperature
        st.session_state.prev_top_p = top_p
        st.session_state.prev_top_k = top_k
        st.session_state.prev_max_new_tokens = max_new_tokens
        st.session_state.summary = ""

    # Connectivity info
    base_url = get_ollama_base_url()
    if not is_ollama_available(base_url):
        st.sidebar.warning(f"Ollama not reachable at {base_url}. Start Ollama and pull the model: ollama pull {MODEL}")

    # Build LLM and chain
    llm = build_llm(MODEL, temperature, top_p, top_k, max_new_tokens, CONTEXT_SIZE)
    prompt_template = PromptTemplate(
        input_variables=["history", "human_input"],
        template="{history}\nUser: {human_input}\nAssistant:"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.memory)

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Show summary if available
    if st.session_state.summary:
        st.markdown("###Conversation Summary")
        st.info(st.session_state.summary)

    # Trim function
    def trim_memory():
        while len(st.session_state.chat_history) > MAX_HISTORY * 2:
            st.session_state.chat_history.pop(0)
            if st.session_state.chat_history:
                st.session_state.chat_history.pop(0)

    # Summarize button logic
    if summarize_clicked and st.session_state.chat_history:
        with st.spinner("Summarizing conversation…"):
            st.session_state.summary = summarize_with_llm(llm, st.session_state.chat_history)

    # Handle user input
    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        trim_memory()

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            try:
                for chunk in chain.stream({"human_input": prompt}):
                    if isinstance(chunk, dict) and "text" in chunk:
                        text_chunk = chunk["text"]
                        full_response += text_chunk
                        response_container.markdown(full_response)
            except Exception as e:
                response_container.error(f"Generation failed: {e}")
                full_response = ""

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        trim_memory()


if __name__ == "__main__":
    main()
