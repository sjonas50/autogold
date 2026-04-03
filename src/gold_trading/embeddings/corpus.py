"""Pine Script v6 RAG corpus ingestion — clone, chunk, embed, and load into pgvector.

Clones codenamedevan/pinescriptv6, applies code-aware chunking (never splits
fenced code blocks), embeds via text-embedding-3-small, loads into pinescript_corpus table.

Usage:
    uv run python -m gold_trading.embeddings.corpus
"""

import asyncio
import re
import subprocess
import tempfile
from pathlib import Path

import asyncpg
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.embeddings.client import embed_texts_batch

REPO_URL = "https://github.com/codenamedevan/pinescriptv6.git"
MAX_CHUNK_TOKENS = 800  # Approximate token limit per chunk
OVERLAP_LINES = 5


def clone_corpus(target_dir: Path) -> Path:
    """Clone the Pine Script v6 documentation corpus."""
    if (target_dir / ".git").exists():
        logger.info(f"Corpus already cloned at {target_dir}")
        return target_dir

    logger.info(f"Cloning {REPO_URL} to {target_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(target_dir)],
        check=True,
        capture_output=True,
    )
    return target_dir


def find_markdown_files(corpus_dir: Path) -> list[Path]:
    """Find all markdown files in the corpus, sorted by path."""
    files = sorted(corpus_dir.rglob("*.md"))
    # Exclude git and non-content files
    return [f for f in files if ".git" not in str(f) and f.name != "LICENSE.md"]


def chunk_markdown(content: str, source_file: str) -> list[dict]:
    """Split markdown into chunks, preserving code blocks.

    Code-aware: fenced code blocks (```...```) are never split across chunks.
    Each chunk gets a type: 'concept', 'reference', or 'example'.
    """
    chunks = []
    current_lines: list[str] = []
    current_tokens = 0
    in_code_block = False
    code_block_lines: list[str] = []

    lines = content.split("\n")

    for line in lines:
        # Track code block boundaries
        if line.strip().startswith("```"):
            if in_code_block:
                # End of code block — add entire block to current chunk
                code_block_lines.append(line)
                block_text = "\n".join(code_block_lines)
                block_tokens = len(block_text.split())

                # If block is too large, it becomes its own chunk
                if current_tokens + block_tokens > MAX_CHUNK_TOKENS and current_lines:
                    chunks.append("\n".join(current_lines))
                    current_lines = []
                    current_tokens = 0

                current_lines.extend(code_block_lines)
                current_tokens += block_tokens
                code_block_lines = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
                code_block_lines = [line]
            continue

        if in_code_block:
            code_block_lines.append(line)
            continue

        # Regular line
        line_tokens = len(line.split())

        if current_tokens + line_tokens > MAX_CHUNK_TOKENS and current_lines:
            chunks.append("\n".join(current_lines))
            # Overlap: keep last N lines for context
            overlap = current_lines[-OVERLAP_LINES:] if len(current_lines) > OVERLAP_LINES else []
            current_lines = overlap.copy()
            current_tokens = sum(len(ln.split()) for ln in current_lines)

        current_lines.append(line)
        current_tokens += line_tokens

    # Flush remaining
    if current_lines:
        chunks.append("\n".join(current_lines))

    # Handle unclosed code block
    if code_block_lines:
        chunks.append("\n".join(code_block_lines))

    # Classify chunks
    result = []
    for i, text in enumerate(chunks):
        chunk_type = "concept"
        if "```" in text:
            chunk_type = "example"
        elif re.search(r"^#+\s+(ta\.|math\.|str\.|array\.|map\.)", text, re.MULTILINE):
            chunk_type = "reference"

        result.append(
            {
                "source_file": source_file,
                "chunk_index": i,
                "content": text.strip(),
                "chunk_type": chunk_type,
                "token_count": len(text.split()),
            }
        )

    return result


async def ingest_corpus(corpus_dir: Path | None = None) -> int:
    """Clone, chunk, embed, and load the Pine Script v6 corpus.

    Args:
        corpus_dir: Path to use for the corpus. If None, uses a temp directory.

    Returns:
        Number of chunks ingested.
    """
    if corpus_dir is None:
        corpus_dir = Path(tempfile.mkdtemp()) / "pinescriptv6"

    # Clone
    clone_corpus(corpus_dir)

    # Find files
    md_files = find_markdown_files(corpus_dir)
    logger.info(f"Found {len(md_files)} markdown files")

    # Chunk all files
    all_chunks = []
    for md_file in md_files:
        relative = str(md_file.relative_to(corpus_dir))
        content = md_file.read_text(errors="replace")
        chunks = chunk_markdown(content, relative)
        all_chunks.extend(chunks)

    logger.info(f"Generated {len(all_chunks)} chunks from {len(md_files)} files")

    if not all_chunks:
        logger.warning("No chunks generated — corpus may be empty")
        return 0

    # Embed in batches
    texts = [c["content"] for c in all_chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = await embed_texts_batch(texts, batch_size=50)

    # Load into DB
    conn = await asyncpg.connect(get_database_url())
    try:
        # Clear existing corpus
        await conn.execute("DELETE FROM pinescript_corpus")

        # Insert all chunks
        for chunk, embedding in zip(all_chunks, embeddings, strict=True):
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            await conn.execute(
                """
                INSERT INTO pinescript_corpus
                    (source_file, chunk_index, content, embedding, token_count, chunk_type)
                VALUES ($1, $2, $3, $4::vector, $5, $6)
                ON CONFLICT (source_file, chunk_index) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    token_count = EXCLUDED.token_count,
                    chunk_type = EXCLUDED.chunk_type
                """,
                chunk["source_file"],
                chunk["chunk_index"],
                chunk["content"],
                embedding_str,
                chunk["token_count"],
                chunk["chunk_type"],
            )

        count = await conn.fetchrow("SELECT COUNT(*) as cnt FROM pinescript_corpus")
        logger.info(f"Corpus ingestion complete: {count['cnt']} chunks in DB")
        return count["cnt"]

    finally:
        await conn.close()


async def search_corpus(
    query_embedding: list[float],
    limit: int = 10,
    chunk_type: str | None = None,
) -> list[dict]:
    """Search the Pine Script corpus by vector similarity.

    Args:
        query_embedding: 1536-dim embedding of the search query.
        limit: Max results to return.
        chunk_type: Optional filter ('concept', 'reference', 'example').

    Returns:
        List of dicts with content, source_file, chunk_type, similarity score.
    """
    conn = await asyncpg.connect(get_database_url())
    try:
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        if chunk_type:
            rows = await conn.fetch(
                """
                SELECT source_file, chunk_index, content, chunk_type, token_count,
                       embedding <=> $1::vector AS distance
                FROM pinescript_corpus
                WHERE chunk_type = $3
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                embedding_str,
                limit,
                chunk_type,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT source_file, chunk_index, content, chunk_type, token_count,
                       embedding <=> $1::vector AS distance
                FROM pinescript_corpus
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                embedding_str,
                limit,
            )

        return [
            {
                "source_file": r["source_file"],
                "chunk_index": r["chunk_index"],
                "content": r["content"],
                "chunk_type": r["chunk_type"],
                "token_count": r["token_count"],
                "distance": float(r["distance"]),
            }
            for r in rows
        ]

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(ingest_corpus())
