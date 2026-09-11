"""Lightweight financial signal extraction for immediate post-ingestion summaries."""
import re


INSIGHT_PATTERNS = {
    "revenue": re.compile(r"\b(?:revenue|sales|net sales|total revenue)\b[^.\n]{0,120}", re.IGNORECASE),
    "profit": re.compile(r"\b(?:net income|net profit|operating profit|gross profit|earnings)\b[^.\n]{0,120}", re.IGNORECASE),
    "debt": re.compile(r"\b(?:total debt|long[- ]term debt|short[- ]term debt|borrowings|debt)\b[^.\n]{0,120}", re.IGNORECASE),
    "risks": re.compile(r"[^.\n]{0,100}\b(?:risk|risks|uncertainty|uncertainties|threat|exposure)\b[^.\n]{0,180}[.\n]", re.IGNORECASE),
    "outlook": re.compile(r"[^.\n]{0,100}\b(?:outlook|forecast|guidance|expects?|expectation|future)\b[^.\n]{0,180}[.\n]", re.IGNORECASE),
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -:;\n\t")


def _unique_matches(pattern: re.Pattern[str], text: str, limit: int = 5) -> list[str]:
    matches: list[str] = []
    seen: set[str] = set()
    for match in pattern.findall(text):
        cleaned = _clean(match)
        key = cleaned.lower()
        if cleaned and key not in seen:
            matches.append(cleaned)
            seen.add(key)
        if len(matches) >= limit:
            break
    return matches


def extract_financial_insights(text: str) -> dict[str, list[str]]:
    """Extract high-signal financial sentences without requiring another model call."""
    return {name: _unique_matches(pattern, text) for name, pattern in INSIGHT_PATTERNS.items()}