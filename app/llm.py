"""Gemini LLM client with model routing and fallback."""
import asyncio
import logging
from google import genai
from google.genai import types

from app.config import (
    GEMINI_API_KEY, EMBEDDING_MODEL,
    ORCHESTRATOR_MODEL, GENERATOR_PRO_MODEL, GENERATOR_FLASH_MODEL, UTILITY_MODEL,
    FALLBACK_PRO_MODEL, FALLBACK_FLASH_MODEL, FALLBACK_UTILITY_MODEL,
)
from app.observability import observe

logger = logging.getLogger(__name__)

_client = None

# Model role -> (primary, fallback)
MODEL_MAP = {
    "orchestrator": (ORCHESTRATOR_MODEL, FALLBACK_FLASH_MODEL),
    "pro": (GENERATOR_PRO_MODEL, FALLBACK_PRO_MODEL),
    "flash": (GENERATOR_FLASH_MODEL, FALLBACK_FLASH_MODEL),
    "utility": (UTILITY_MODEL, FALLBACK_UTILITY_MODEL),
}


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# Timeouts per model role (seconds)
_TIMEOUT = {"pro": 45, "flash": 30, "orchestrator": 30, "utility": 30}
_EMBED_TIMEOUT = 15


@observe(as_type="generation")
async def generate(
    prompt: str,
    system: str = "",
    history: list[dict] | None = None,
    model: str = "flash",
    temperature: float = 0.3,
    max_tokens: int = 4000,
) -> str:
    """Generate a response using Gemini with automatic fallback."""
    client = get_client()
    primary, fallback = MODEL_MAP.get(model, (GENERATOR_FLASH_MODEL, FALLBACK_FLASH_MODEL))
    timeout = _TIMEOUT.get(model, 30)

    contents = []
    if history:
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    for model_id in (primary, fallback):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model_id, contents=contents, config=config,
                ),
                timeout=timeout,
            )
            return response.text or ""
        except asyncio.TimeoutError:
            logger.warning("Model %s timed out after %ds", model_id, timeout)
            if model_id == primary:
                continue
            raise TimeoutError(f"Gemini generation timed out after {timeout}s")
        except Exception as e:
            if model_id == primary:
                logger.warning("Primary model %s failed, trying fallback %s: %s", primary, fallback, e)
                continue
            logger.error("Fallback model %s also failed: %s", fallback, e)
            raise

    return ""


@observe(name="embed")
async def embed(texts: list[str]) -> list[list[float]]:
    """Create embeddings using Gemini embedding model."""
    client = get_client()
    result = await asyncio.wait_for(
        asyncio.to_thread(
            client.models.embed_content, model=EMBEDDING_MODEL, contents=texts,
        ),
        timeout=_EMBED_TIMEOUT,
    )
    return [e.values for e in result.embeddings]
