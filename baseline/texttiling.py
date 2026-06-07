"""NLTK TextTiling baseline for paragraph segmentation."""

import json
from pathlib import Path

from nltk.tokenize import TextTilingTokenizer

SENTENCE_DELIMITER = "\n\n"


def _ensure_nltk_resources() -> None:
    """Download NLTK resources required by TextTilingTokenizer."""
    import nltk

    for resource in ("stopwords",):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _load_records(jsonl_path: Path) -> list[dict]:
    """Load sentence records from a JSONL file."""
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _records_to_text(records: list[dict]) -> str:
    """Join sentence texts into a single document for TextTiling."""
    return SENTENCE_DELIMITER.join(record["text"] for record in records)


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace so paragraph text can be matched to sentence records."""
    return " ".join(text.split())


def _paragraphs_to_boundary_ids(records: list[dict], paragraphs: list[str]) -> list[int]:
    """
    Map TextTiling paragraph strings back to boundary sentence ids.

    If a split occurs between sentence a and sentence b, sentence b's id
    is included in the returned list.
    """
    boundary_ids: list[int] = []
    sentence_index = 0

    for paragraph_index, paragraph in enumerate(paragraphs):
        if paragraph_index > 0 and sentence_index < len(records):
            boundary_ids.append(records[sentence_index]["id"])

        remaining = _normalize_whitespace(paragraph)
        while sentence_index < len(records) and remaining:
            sentence = _normalize_whitespace(records[sentence_index]["text"])
            if remaining.startswith(sentence):
                remaining = remaining[len(sentence):].lstrip()
                sentence_index += 1
            else:
                break

    return boundary_ids


def segment_texttiling(jsonl_path: str | Path) -> list[int]:
    """
    Segment a sentence-level JSONL file using NLTK TextTiling.

    Args:
        jsonl_path: Path to a JSONL file where each line contains at least
            ``id`` and ``text`` fields.

    Returns:
        Sentence ids marking the start of each new paragraph after the first.
        For example, if the split falls between sentence a and sentence b,
        then sentence b's id appears in the list.
    """
    path = Path(jsonl_path)
    records = _load_records(path)
    if len(records) <= 1:
        return []

    _ensure_nltk_resources()

    text = _records_to_text(records)
    tokenizer = TextTilingTokenizer()
    paragraphs = tokenizer.tokenize(text)

    return _paragraphs_to_boundary_ids(records, paragraphs)
