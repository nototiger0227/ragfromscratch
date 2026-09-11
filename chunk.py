"""Chunking utilities with section-aware and overlap-aware splitting."""
import re

from config import CHUNK_SIZE, CHUNK_OVERLAP


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(text: str) -> list[str]:
    """Break large text into sections using blank lines and heading-like lines."""
    lines = [line.rstrip() for line in text.split("\n")]
    sections: list[str] = []
    current: list[str] = []

    def flush():
        nonlocal current
        block = "\n".join(part.strip() for part in current if part.strip())
        if block:
            sections.append(block)
        current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                flush()
            continue

        is_heading = bool(re.match(r"^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 ,&()/-]{2,}:?)$", stripped))
        if is_heading and current:
            flush()

        current.append(stripped)

    flush()
    return sections


def chunk_section(section: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunk a section while preserving overlap and avoiding oversized pieces."""
    if len(section) <= chunk_size:
        return [section.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(section):
        end = min(start + chunk_size, len(section))
        chunk = section[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(section):
            break
        start += max(1, chunk_size - overlap)

    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Section-aware chunking with a fallback to fixed-size chunks for oversized text."""
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    sections = split_into_sections(cleaned)
    if not sections:
        sections = [cleaned]

    chunks: list[str] = []
    for section in sections:
        chunks.extend(chunk_section(section, chunk_size, overlap))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


if __name__ == "__main__":
    sample = "# Revenue\nThe company generated significant revenue in 2024.\n\n# Risk\nSupply chain interruptions remain a key concern."
    for i, c in enumerate(chunk_text(sample, chunk_size=80, overlap=20)):
        print(f"chunk {i}: {c[:60]}... (len={len(c)})")
