from unittest.mock import patch, MagicMock

import pytest

from llm import call_llm, LLMError


def test_groq_success_returns_content():
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
    with patch("llm._groq_client") as mock_groq:
        mock_groq.chat.completions.create.return_value = fake_resp
        out = call_llm("system", "user", expect_json=True)
        assert out == '{"ok": true}'


def test_groq_failure_falls_back_to_ollama():
    with patch("llm._groq_client") as mock_groq, \
         patch("llm._ollama_chat") as mock_ollama:
        mock_groq.chat.completions.create.side_effect = RuntimeError("groq down")
        mock_ollama.return_value = '{"from": "ollama"}'
        out = call_llm("system", "user", expect_json=True)
        assert out == '{"from": "ollama"}'
        mock_ollama.assert_called_once()


def test_both_providers_failing_raises_llm_error():
    with patch("llm._groq_client") as mock_groq, \
         patch("llm._ollama_chat") as mock_ollama:
        mock_groq.chat.completions.create.side_effect = RuntimeError("groq down")
        mock_ollama.side_effect = RuntimeError("ollama down")
        with pytest.raises(LLMError):
            call_llm("system", "user")
