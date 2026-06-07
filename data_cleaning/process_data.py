"""Process TED transcript text files into sentence-level JSONL records."""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "TEDs"
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])(?:["\'\)\]]*)\s+')
TITLE_LINE_PATTERN = re.compile(r'^[a-z0-9_]+$')


def resolve_data_path(folder_path: str | Path) -> Path:
    """Resolve a data folder path against the project root when relative."""
    folder = Path(folder_path)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    return folder.resolve()


def get_output_dir(input_folder: Path) -> Path:
    """Return the output folder named `<input_folder.name>_cleaned`."""
    return input_folder.parent / f"{input_folder.name}_cleaned"


def _is_title_line(line: str) -> bool:
    """Return True for slug-like metadata lines such as talk titles."""
    stripped = line.strip()
    return bool(stripped) and TITLE_LINE_PATTERN.match(stripped)


def read_text_file(file_path: Path) -> str:
    """Read a transcript file and return body text without slug title lines."""
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    paragraphs = [
        line.strip()
        for line in lines
        if line.strip() and not _is_title_line(line)
    ]
    return " ".join(paragraphs)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK when available, otherwise punctuation."""
    text = text.strip()
    if not text:
        return []

    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        return [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
    except ImportError:
        return _split_by_punctuation(text)


def _split_by_punctuation(text: str) -> list[str]:
    """Fallback sentence splitter based on terminal punctuation."""
    parts = SENTENCE_SPLIT_PATTERN.split(text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences


def sentences_to_records(sentences: list[str]) -> list[dict]:
    """Convert sentence strings into JSONL-ready records."""
    return [
        {"id": index, "text": sentence, "is_boundary": 0}
        for index, sentence in enumerate(sentences)
    ]


def write_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_file(input_path: Path, output_path: Path) -> int:
    """Process one transcript file and write its JSONL output."""
    text = read_text_file(input_path)
    sentences = split_sentences(text)
    records = sentences_to_records(sentences)
    write_jsonl(records, output_path)
    return len(records)


def process_folder(folder_path: str | Path) -> dict[str, int]:
    """
    Process all .txt files in a folder into matching .jsonl files.

    Output is written to a sibling folder named `<folder_name>_cleaned`.

    Returns a mapping of input file names to the number of sentences written.
    """
    folder = resolve_data_path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {folder}")

    output_dir = get_output_dir(folder)
    results: dict[str, int] = {}
    for input_path in sorted(folder.glob("*.txt")):
        output_path = output_dir / input_path.with_suffix(".jsonl").name
        sentence_count = process_file(input_path, output_path)
        results[input_path.name] = sentence_count

    return results


if __name__ == "__main__":
    summary = process_folder(DEFAULT_INPUT_DIR)
    output_dir = get_output_dir(resolve_data_path(DEFAULT_INPUT_DIR))
    print(f"Output directory: {output_dir}")
    for file_name, sentence_count in summary.items():
        print(f"{file_name}: {sentence_count} sentences")
