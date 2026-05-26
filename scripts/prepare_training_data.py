import argparse
import json
from pathlib import PurePosixPath
from typing import Any


REQUIRED_KEYS = {"images", "messages", "solution"}


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Training data must be a JSON list.")
    return data


def normalize_image_path(path: str, keep_parts: int) -> str:
    normalized = path.replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in {"/", ""}]
    if keep_parts <= 0 or keep_parts >= len(parts):
        return "/".join(parts)
    return "/".join(parts[-keep_parts:])


def normalize_item(item: dict[str, Any], keep_image_parts: int) -> dict[str, Any]:
    missing = REQUIRED_KEYS - set(item)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")

    output = dict(item)
    output["images"] = [normalize_image_path(str(path), keep_image_parts) for path in item.get("images", [])]
    solution = dict(item.get("solution") or {})
    table = solution.get("table")
    if isinstance(table, str):
        json.loads(table)
    elif table is not None:
        solution["table"] = json.dumps(table, ensure_ascii=False)
    output["solution"] = solution
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-image-parts", type=int, default=5)
    args = parser.parse_args()

    data = load_json(args.input)
    normalized = [normalize_item(item, args.keep_image_parts) for item in data]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(json.dumps({"samples": len(normalized), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
