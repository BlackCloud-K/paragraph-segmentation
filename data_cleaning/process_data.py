"""Process TED transcript text files into sentence-level JSONL records."""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "TEDs"
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])(?:["\'\)\]]*)\s+')
TITLE_LINE_PATTERN = re.compile(r'^[a-z0-9_]+$')
OPEN_QUOTES = {'"', '\u2018', '\u201c'}
CLOSE_QUOTES = {'"', '\u2019', '\u201d'}
SENTENCE_END_MASKS = {'.': '\x00', '!': '\x01', '?': '\x02'}
SENTENCE_END_UNMASKS = {mask: char for char, mask in SENTENCE_END_MASKS.items()}
APOSTROPHE_CHARS = {"'", '\u2019'}
POST_QUOTE_NARRATION_SPLIT = re.compile(r'(?<=[.!?][""\u201d])\s+(?=[A-Z])')


def _is_apostrophe(text: str, index: int) -> bool:
    """Return True when a single quote character is part of a contraction."""
    char = text[index]
    if char not in APOSTROPHE_CHARS:
        return False
    if index > 0 and text[index - 1].isalpha():
        return index + 1 < len(text) and text[index + 1].isalpha()
    return False


def _mask_punctuation_in_quotes(text: str) -> str:
    """Replace sentence-ending punctuation inside quotes with placeholders."""
    result: list[str] = []
    in_quotes = False

    for index, char in enumerate(text):
        if char == '"':
            in_quotes = not in_quotes
            result.append(char)
            continue

        if char in OPEN_QUOTES:
            in_quotes = True
            result.append(char)
            continue

        if char in CLOSE_QUOTES and not _is_apostrophe(text, index):
            in_quotes = False
            result.append(char)
            continue

        if in_quotes and char in SENTENCE_END_MASKS:
            result.append(SENTENCE_END_MASKS[char])
        else:
            result.append(char)

    return "".join(result)


def _unmask_punctuation(text: str) -> str:
    """Restore masked sentence-ending punctuation."""
    for mask, char in SENTENCE_END_UNMASKS.items():
        text = text.replace(mask, char)
    return text


def _split_after_closed_quotes(sentences: list[str]) -> list[str]:
    """Split narration that resumes immediately after a closing quote."""
    result: list[str] = []
    for sentence in sentences:
        parts = POST_QUOTE_NARRATION_SPLIT.split(sentence)
        result.extend(part.strip() for part in parts if part.strip())
    return result


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

    masked_text = _mask_punctuation_in_quotes(text)

    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        sentences = sent_tokenize(masked_text)
    except ImportError:
        sentences = _split_by_punctuation(masked_text)

    sentences = [
        _unmask_punctuation(sentence.strip())
        for sentence in sentences
        if sentence.strip()
    ]
    return _split_after_closed_quotes(sentences)


def _split_by_punctuation(text: str) -> list[str]:
    """Fallback sentence splitter based on terminal punctuation."""
    parts = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


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
