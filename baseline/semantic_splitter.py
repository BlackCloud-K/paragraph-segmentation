"""Semantic embedding threshold baseline for paragraph segmentation."""

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BOTTOM_PERCENTILE = 5

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


def _get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load and cache the sentence embedding model."""
    global _model, _model_name

    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
        _model_name = model_name

    return _model


def _encode_sentences(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode sentence texts into normalized embedding vectors."""
    model = _get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def _adjacent_cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each adjacent sentence pair."""
    if len(embeddings) < 2:
        return np.array([], dtype=float)

    return np.sum(embeddings[:-1] * embeddings[1:], axis=1)


def _similarities_to_boundary_ids(
    records: list[dict],
    similarities: np.ndarray,
    bottom_percentile: float = BOTTOM_PERCENTILE,
) -> list[int]:
    """
    Pick split points where adjacent similarity falls in the bottom percentile.

    A gap at index i sits between sentence i and sentence i + 1, so the
    boundary id is the id of sentence i + 1.
    """
    if len(similarities) == 0:
        return []

    threshold = np.percentile(similarities, bottom_percentile)
    boundary_ids = [
        records[gap_index + 1]["id"]
        for gap_index, similarity in enumerate(similarities)
        if similarity <= threshold
    ]
    return boundary_ids


def segment_semantic_splitter(
    jsonl_path: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    bottom_percentile: float = BOTTOM_PERCENTILE,
) -> list[int]:
    """
    Segment a sentence-level JSONL file using embedding similarity drops.

    Each sentence is embedded with a lightweight sentence-transformers model.
    Adjacent cosine similarities are computed, and gaps in the lowest
    ``bottom_percentile`` percent are treated as paragraph boundaries.

    Args:
        jsonl_path: Path to a JSONL file where each line contains at least
            ``id`` and ``text`` fields.
        model_name: Hugging Face model id for local embedding inference.
        bottom_percentile: Percentile cutoff for low-similarity split points.

    Returns:
        Sentence ids marking the start of each new paragraph after the first.
        For example, if the split falls between sentence a and sentence b,
        then sentence b's id appears in the list.
    """
    path = Path(jsonl_path)
    records = _load_records(path)
    if len(records) <= 1:
        return []

    texts = [record["text"] for record in records]
    embeddings = _encode_sentences(texts, model_name=model_name)
    similarities = _adjacent_cosine_similarities(embeddings)

    return _similarities_to_boundary_ids(
        records,
        similarities,
        bottom_percentile=bottom_percentile,
    )
