"""
models.py
Centralized model configuration and Groq LLM initialization.
Supports Streamlit secrets and local .env fallback.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load local environment variables if available
load_dotenv(override=True)

# Centralized Dictionary of Verified Available Groq Models
AVAILABLE_MODELS: Dict[str, Dict[str, str]] = {
    "Model 1 - Fast (GPT-OSS 20B)": {
        "id": "openai/gpt-oss-20b",
        "description": "Fast, lightweight model with minimal latency. Great for everyday standard queries.",
        "badge": "⚡ Fast"
    },
    "Model 2 - Balanced (Qwen 3.8 27B)": {
        "id": "qwen/qwen3.8-27b",
        "description": "High precision and dialect accuracy. Recommended for complex multi-table joins.",
        "badge": "⚖️ Balanced"
    },
    "Model 3 - Powerful (GPT-OSS 120B)": {
        "id": "openai/gpt-oss-120b",
        "description": "Flagship 120B reasoning model. Excels at complex analytical queries and self-correction.",
        "badge": "🧠 Powerful"
    }
}

DEFAULT_MODEL_NAME = "Model 2 - Balanced (Qwen 3.8 27B)"

def get_groq_api_key() -> Optional[str]:
    """
    Retrieves the Groq API key with cross-platform fallback:
    1. Streamlit Secrets (st.secrets["GROQ_API_KEY"]) for Streamlit Community Cloud.
    2. Environment variable (GROQ_API_KEY) loaded from .env for local run.
    """
    # 1. Try Streamlit Secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
            if key and str(key).strip():
                return str(key).strip()
    except Exception:
        pass
        
    # 2. Fallback to OS Environment / .env
    key = os.getenv("GROQ_API_KEY")
    if key and key.strip() and key != "your_groq_api_key_here":
        return key.strip()
        
    return None

def get_llm(model_id: str, temperature: float = 0.0, max_tokens: int = 400) -> ChatGroq:
    """
    Instantiates a ChatGroq LLM for the given model_id with temperature 0.
    Raises ValueError if API key is missing.
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured! Please set it in your .env file or Streamlit Cloud secrets."
        )
        
    return ChatGroq(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )
