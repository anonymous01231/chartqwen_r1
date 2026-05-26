import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/chartqwen_r1_train_2876.json")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_count = sum(len(item.get("images", [])) for item in data)
    message_count = sum(len(item.get("messages", [])) for item in data)
    label_types = Counter()
    table_rows = []
    for item in data:
        solution = item.get("solution", {})
        label = solution.get("label", "")
        label_types["numeric" if str(label).replace(".", "", 1).isdigit() else "text"] += 1
        table = solution.get("table")
        try:
            table = json.loads(table) if isinstance(table, str) else table
            table_rows.append(len(table.get("rows", {})))
        except Exception:
            pass

    summary = {
        "samples": len(data),
        "images": image_count,
        "messages": message_count,
        "file_size_bytes": os.path.getsize(args.data),
        "label_types": dict(label_types),
        "avg_table_rows": sum(table_rows) / len(table_rows) if table_rows else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
