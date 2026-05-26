import argparse
import json
import math
import re
from typing import Any, Optional


def parse_numeric(text: Any) -> Optional[float]:
    matches = re.findall(r"[-+]?\d*\.?\d+", str(text))
    return float(matches[0]) if len(matches) == 1 else None


def normalize_text(text: Any) -> str:
    return str(text).strip().rstrip(".").lower()


def relaxed_match(prediction: Any, target: Any, tol: float = 0.05) -> bool:
    pred_text = str(prediction).strip().rstrip(".")
    target_text = str(target).strip()
    pred_value = parse_numeric(pred_text)
    target_value = parse_numeric(target_text)
    if pred_value is not None and target_value is not None:
        threshold = abs(target_value) * tol
        return (
            abs(pred_value - target_value) <= threshold
            or abs(pred_value * 100 - target_value) <= threshold
            or abs(pred_value / 100 - target_value) <= threshold
        )
    return normalize_text(pred_text) == normalize_text(target_text)


def extract_python_code(response: str) -> Optional[str]:
    match = re.search(r"<python>\s*(.*?)</python>", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_answer_from_code(response: str) -> Any:
    code = extract_python_code(response)
    if not code:
        return ""
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    env = {"__builtins__": safe_builtins, "math": math}
    try:
        exec(code, env, env)
    except Exception:
        return ""
    answer = env.get("answer", "")
    if answer == True:
        return "yes"
    if answer == False:
        return "no"
    return answer


def load_rows(path: str):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Inference result JSON/JSONL file.")
    parser.add_argument("--prediction-field", default="model_response")
    parser.add_argument("--target-field", default="gt")
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = load_rows(args.input)
    scored_rows = []
    correct = 0
    executable = 0
    for row in rows:
        response = row.get(args.prediction_field, "")
        pred = extract_answer_from_code(response)
        target = row.get(args.target_field, "")
        is_executable = extract_python_code(response) is not None and pred != ""
        is_correct = relaxed_match(pred, target, args.tol)
        executable += int(is_executable)
        correct += int(is_correct)
        scored = dict(row)
        scored["pred_answer"] = pred
        scored["is_executable"] = is_executable
        scored["is_correct"] = is_correct
        scored_rows.append(scored)

    summary = {
        "count": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "executable_rate": executable / len(rows) if rows else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": scored_rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
