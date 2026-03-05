"""Gemini LLM client with model routing."""
from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GEMINI_EMBEDDING_MODEL

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def generate(
    prompt: str,
    system: str = "",
    history: list[dict] | None = None,
    model: str = "flash",
    temperature: float = 0.3,
    max_tokens: int = 4000,
) -> str:
    """Generate a response using Gemini. model='flash' or 'pro'."""
    client = get_client()
    model_id = GEMINI_PRO_MODEL if model == "pro" else GEMINI_FLASH_MODEL

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

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config,
    )
    return response.text or ""


def embed(texts: list[str]) -> list[list[float]]:
    """Create embeddings using Gemini embedding model."""
    client = get_client()
    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in result.embeddings]


def classify_complexity(message: str) -> str:
    """Quick classification: 'simple' or 'complex'."""
    simple_indicators = [
        len(message) < 20,
        message.strip().endswith("?") and len(message) < 40,
        any(w in message.lower() for w in ["cam on", "ok", "da", "vang", "chao", "hi", "hello"]),
    ]
    if sum(simple_indicators) >= 2:
        return "simple"
    return "complex"
