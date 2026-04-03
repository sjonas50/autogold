"""Anthropic embeddings API client for text-embedding-3-small."""

import os

import httpx
from loguru import logger

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://api.openai.com/v1"


async def embed_text(text: str) -> list[float]:
    """Embed a single text string using OpenAI's text-embedding-3-small.

    Args:
        text: The text to embed.

    Returns:
        List of floats (1536 dimensions).

    Raises:
        RuntimeError: If the API call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Required for embeddings via text-embedding-3-small."
        )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{OPENAI_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
                "dimensions": EMBEDDING_DIMENSIONS,
            },
        )

        if resp.status_code != 200:
            logger.error(f"Embedding API error: {resp.status_code} {resp.text}")
            raise RuntimeError(f"Embedding API returned {resp.status_code}: {resp.text}")

        data = resp.json()
        return data["data"][0]["embedding"]


async def embed_texts_batch(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """Embed multiple texts in batches.

    Args:
        texts: List of text strings to embed.
        batch_size: Max texts per API call (OpenAI limit is 2048).

    Returns:
        List of embedding vectors in the same order as input texts.
    """
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    all_embeddings: list[list[float]] = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            resp = await client.post(
                f"{OPENAI_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": batch,
                    "dimensions": EMBEDDING_DIMENSIONS,
                },
            )

            if resp.status_code != 200:
                logger.error(f"Embedding batch API error: {resp.status_code} {resp.text}")
                raise RuntimeError(f"Embedding API returned {resp.status_code}")

            data = resp.json()
            # Sort by index to preserve order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend([d["embedding"] for d in sorted_data])

            if i + batch_size < len(texts):
                logger.debug(f"Embedded {i + batch_size}/{len(texts)} texts")

    return all_embeddings
