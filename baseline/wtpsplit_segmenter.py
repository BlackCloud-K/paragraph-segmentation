"""wtpsplit deep learning baseline for paragraph segmentation."""

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "sat-3l-sm"
DEFAULT_PARAGRAPH_THRESHOLD = 0.96
SENTENCE_SEPARATOR = " "


_model = None
_model_name: str | None = None


def _load_records(jsonl_path: Path) -> list[dict]:
    """Load sentence records from a JSONL file."""
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_sat_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load and cache the wtpsplit SaT model."""
    global _model, _model_name

    if _model is None or _model_name != model_name:
        from wtpsplit import SaT

        _model = SaT(model_name)
        _model_name = model_name

    return _model


def _records_to_text(records: list[dict]) -> str:
    """Join sentence texts into one document for wtpsplit."""
    return SENTENCE_SEPARATOR.join(record["text"] for record in records)


def _predict_boundary_probabilities(model, text: str) -> np.ndarray:
    """Return sentence-boundary probabilities for each character in the document."""
    result = model.predict_proba(text, return_paragraph_probabilities=True)
    if isinstance(result, tuple):
        sentence_probs, _ = result
    else:
        sentence_probs, _ = next(iter(result))
    return np.asarray(sentence_probs)


def _score_sentence_gaps(records: list[dict], sentence_probs: np.ndarray) -> list[tuple[int, float]]:
    """
    Score each adjacent sentence pair using wtpsplit boundary probability.

    The score is taken at the last character of the previous sentence, which
    is the closest model signal to our pre-segmented sentence boundary.
    """
    scored_gaps: list[tuple[int, float]] = []
    position = 0

    for index, record in enumerate(records):
        if index + 1 < len(records):
            boundary_char = position + len(record["text"]) - 1
            score = float(sentence_probs[boundary_char])
            scored_gaps.append((records[index + 1]["id"], score))
        position += len(record["text"])
        if index + 1 < len(records):
            position += len(SENTENCE_SEPARATOR)

    return scored_gaps


def _gaps_to_boundary_ids(
    scored_gaps: list[tuple[int, float]],
    paragraph_threshold: float,
) -> list[int]:
    """
    Keep only the highest-scoring sentence gaps.

    ``paragraph_threshold`` is interpreted as a percentile in [0, 1]:
    only gaps with scores at or above that percentile are kept.
    Higher values are stricter and produce fewer boundaries.
    """
    if not scored_gaps:
        return []

    scores = [score for _, score in scored_gaps]
    cutoff = float(np.percentile(scores, paragraph_threshold * 100))
    return [sentence_id for sentence_id, score in scored_gaps if score >= cutoff]


def segment_wtpsplit(
    jsonl_path: str | Path,
    paragraph_threshold: float = DEFAULT_PARAGRAPH_THRESHOLD,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[int]:
    """
    Segment a sentence-level JSONL file using wtpsplit boundary probabilities.

    Important limitation: wtpsplit is trained to detect sentence/newline
    boundaries in raw text, not semantic paragraph shifts in oral transcripts.
    For our JSONL inputs, this baseline therefore scores only the gaps between
    already segmented sentences and keeps the highest-scoring gaps.

    Args:
        jsonl_path: Path to a JSONL file where each line contains at least
            ``id`` and ``text`` fields.
        paragraph_threshold: Percentile cutoff in [0, 1]. Only gaps scoring at
            or above this percentile are kept. Higher values yield fewer
            breakpoints.
        model_name: wtpsplit SaT model id to load locally.

    Returns:
        Sentence ids marking the start of each new paragraph after the first.
        For example, if the split falls between sentence a and sentence b,
        then sentence b's id appears in the list.
    """
    path = Path(jsonl_path)
    records = _load_records(path)
    if len(records) <= 1:
        return []

    model = _get_sat_model(model_name)
    text = _records_to_text(records)
    sentence_probs = _predict_boundary_probabilities(model, text)
    scored_gaps = _score_sentence_gaps(records, sentence_probs)

    return _gaps_to_boundary_ids(scored_gaps, paragraph_threshold)
