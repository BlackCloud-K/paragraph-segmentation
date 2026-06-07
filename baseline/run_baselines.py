"""Run all baseline segmentation methods on labeled JSONL files."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.texttiling import segment_texttiling

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "TEDs_labeled"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "teds_results"

BaselineFn = Callable[[str | Path], list[int]]

BASELINE_METHODS: dict[str, BaselineFn] = {
    "texttiling": segment_texttiling,
}


def resolve_data_path(folder_path: str | Path) -> Path:
    """Resolve a folder path against the project root when relative."""
    folder = Path(folder_path)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    return folder.resolve()


def write_result(
    output_path: Path,
    method_name: str,
    source_file: str,
    boundaries: list[int],
) -> None:
    """Write baseline segmentation result to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method_name,
        "source_file": source_file,
        "boundaries": boundaries,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run_baselines(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict[str, Path]]:
    """
    Run every registered baseline on each JSONL file in the input folder.

    Returns a mapping of method name to source filename to output path.
    """
    input_folder = resolve_data_path(input_dir)
    output_folder = resolve_data_path(output_dir)

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_folder}")

    jsonl_files = sorted(input_folder.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in: {input_folder}")

    results: dict[str, dict[str, Path]] = {
        method_name: {} for method_name in BASELINE_METHODS
    }

    for method_name, segment_fn in BASELINE_METHODS.items():
        for input_path in jsonl_files:
            boundaries = segment_fn(input_path)
            output_path = output_folder / f"{input_path.stem}.{method_name}.json"
            write_result(
                output_path=output_path,
                method_name=method_name,
                source_file=input_path.name,
                boundaries=boundaries,
            )
            results[method_name][input_path.name] = output_path

    return results


if __name__ == "__main__":
    summary = run_baselines()
    output_dir = resolve_data_path(DEFAULT_OUTPUT_DIR)
    print(f"Output directory: {output_dir}")
    for method_name, file_results in summary.items():
        print(f"\n{method_name}:")
        for source_name, result_path in file_results.items():
            print(f"  {source_name} -> {result_path.name}")
