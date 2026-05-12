"""Unified LLM client: Groq primary, Ollama fallback."""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class LLMError(Exception):
    pass


def _build_groq_client():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


_groq_client = _build_groq_client()


def _ollama_chat(system: str, user: str, expect_json: bool) -> str:
    import ollama
    fmt = "json" if expect_json else ""
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=fmt,
        options={"temperature": 0.3},
    )
    return resp["message"]["content"]


def call_llm(system: str, user: str, expect_json: bool = True) -> str:
    """Try Groq first, fall back to Ollama. Raises LLMError if both fail."""
    if _groq_client is not None:
        try:
            kwargs = {"response_format": {"type": "json_object"}} if expect_json else {}
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                **kwargs,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("Groq call failed, falling back to Ollama: %s", e)

    try:
        return _ollama_chat(system, user, expect_json)
    except Exception as e:
        raise LLMError(f"Both Groq and Ollama failed: {e}") from e


def provider_status() -> dict:
    """For the sidebar status indicators."""
    return {
        "groq_configured": _groq_client is not None,
        "groq_model": GROQ_MODEL,
        "ollama_model": OLLAMA_MODEL,
    }
