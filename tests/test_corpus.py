"""Tests for Pine Script v6 corpus ingestion — chunking and code preservation."""

from gold_trading.embeddings.corpus import chunk_markdown


class TestCodeAwareChunking:
    def test_code_blocks_never_split(self):
        """Fenced code blocks must stay in a single chunk."""
        content = """# Strategy Example

Some intro text here.

```pinescript
//@version=6
strategy("My Strategy", process_orders_on_close=true)
length = input.int(14, "Length")
src = close
ma = ta.sma(src, length)
if ta.crossover(src, ma)
    strategy.entry("Long", strategy.long)
if ta.crossunder(src, ma)
    strategy.close("Long")
```

More text after the code block.
"""
        chunks = chunk_markdown(content, "test.md")
        # Find the chunk containing the code block
        code_chunks = [c for c in chunks if "```" in c["content"]]
        assert len(code_chunks) >= 1

        # The code block should be complete (both opening and closing ```)
        for chunk in code_chunks:
            count = chunk["content"].count("```")
            assert count % 2 == 0, "Code block is split across chunks"

    def test_code_block_classified_as_example(self):
        """Chunks containing code blocks should be typed 'example'."""
        content = """```pinescript
strategy("Test")
```"""
        chunks = chunk_markdown(content, "test.md")
        assert any(c["chunk_type"] == "example" for c in chunks)

    def test_regular_text_classified_as_concept(self):
        """Text without code should be typed 'concept'."""
        content = "# Concepts\n\nThis is a concept about Pine Script strategies."
        chunks = chunk_markdown(content, "test.md")
        assert all(c["chunk_type"] == "concept" for c in chunks)

    def test_multiple_code_blocks_preserved(self):
        """Multiple code blocks in one document are all preserved."""
        content = """# Examples

```pinescript
a = close
```

Some text.

```pinescript
b = open
```
"""
        chunks = chunk_markdown(content, "test.md")
        all_content = " ".join(c["content"] for c in chunks)
        assert "a = close" in all_content
        assert "b = open" in all_content

    def test_chunk_metadata(self):
        """Each chunk has required metadata fields."""
        content = "# Test\n\nSome content here about strategies."
        chunks = chunk_markdown(content, "concepts/strategies.md")

        for chunk in chunks:
            assert "source_file" in chunk
            assert "chunk_index" in chunk
            assert "content" in chunk
            assert "chunk_type" in chunk
            assert "token_count" in chunk
            assert chunk["source_file"] == "concepts/strategies.md"
            assert chunk["token_count"] > 0

    def test_large_document_chunked(self):
        """Large documents are split into multiple chunks."""
        # Generate ~2000 words
        content = "# Big Document\n\n" + "\n".join(
            f"This is sentence number {i} about gold trading strategies and Pine Script."
            for i in range(200)
        )
        chunks = chunk_markdown(content, "big.md")
        assert len(chunks) > 1

    def test_overlap_between_chunks(self):
        """Consecutive chunks should have overlapping context."""
        content = "# Doc\n\n" + "\n".join(
            f"Line {i}: Important information about trading gold futures." for i in range(100)
        )
        chunks = chunk_markdown(content, "overlap.md")
        if len(chunks) >= 2:
            # Last lines of chunk N should appear at start of chunk N+1
            last_lines_0 = chunks[0]["content"].split("\n")[-3:]
            first_lines_1 = chunks[1]["content"].split("\n")[:10]
            # At least some overlap
            overlap = set(last_lines_0) & set(first_lines_1)
            assert len(overlap) > 0, "No overlap between consecutive chunks"

    def test_empty_content(self):
        """Empty content produces no chunks."""
        chunks = chunk_markdown("", "empty.md")
        assert len(chunks) == 0 or all(c["content"].strip() == "" for c in chunks)

    def test_unclosed_code_block(self):
        """Unclosed code block is still captured."""
        content = """```pinescript
strategy("Unclosed")
a = close
"""
        chunks = chunk_markdown(content, "test.md")
        all_content = " ".join(c["content"] for c in chunks)
        assert "strategy" in all_content
