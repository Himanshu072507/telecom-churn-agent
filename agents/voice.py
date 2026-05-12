"""Agent 2: Voice Agent — generates a retention call script + TTS audio."""
import io
import json
import logging
from pathlib import Path

from gtts import gTTS

from guardrails import GuardrailViolation, validate_script
from llm import call_llm, LLMError
from schemas import AnalystOutput, OfferOutput, VoiceOutput

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "voice.txt"
SYSTEM_PROMPT = "You output strict JSON only."


def _fallback_script(customer: dict, offer: OfferOutput) -> VoiceOutput:
    name = customer.get("name", "there")
    return VoiceOutput(
        opening_line=f"Hi {name}, thank you for being a valued customer.",
        key_talking_points=["thank for tenure", "introduce offer", "invite questions"],
        full_script=(
            f"Hi {name}, thank you for being with us. We appreciate your continued "
            f"trust. I wanted to share an offer with you today: {offer.offer_details}. "
            f"Would this be helpful for you? I'm happy to answer any questions."
        ),
        do_not_say=["fallback script used due to validation failure"],
        estimated_call_duration_sec=60,
    )


def generate_script(
    customer: dict, analysis: AnalystOutput, offer: OfferOutput
) -> VoiceOutput:
    user_msg = PROMPT_PATH.read_text() \
        .replace("{analyst_json}", analysis.model_dump_json()) \
        .replace("{customer_json}", json.dumps(customer)) \
        .replace("{offer_json}", offer.model_dump_json())

    for attempt in range(2):
        try:
            raw = call_llm(SYSTEM_PROMPT, user_msg, expect_json=True)
            parsed = json.loads(raw)
            script = VoiceOutput(**parsed)
            validate_script(script)
            return script
        except Exception as e:  # broad: LLM, parse, schema, guardrail
            logger.warning("Voice attempt %d failed: %s", attempt + 1, e)

    return _fallback_script(customer, offer)


def script_to_audio_bytes(script: VoiceOutput) -> bytes:
    """Convert script.full_script to MP3 bytes via gTTS."""
    tts = gTTS(text=script.full_script, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()
