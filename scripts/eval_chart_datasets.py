import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from swift.llm import InferRequest, PtEngine, RequestConfig, get_model_tokenizer, get_template
from swift.tuners import Swift
from tqdm import tqdm


SYSTEM_PROMPT = """You are a vision-language assistant. You are given a chart image and a query about the chart.
Internally plan the steps needed to answer the query, but do NOT output your reasoning process.
Your response must contain exactly one block and nothing else:

<python>
# The code must define a variable named `table` with chart data and a variable named `answer` with the final answer.
# Use Python code to compute the answer from the recovered table. Do NOT print anything.
</python>
"""


def build_engine(model_path: str, lora_path: str | None = None):
    if lora_path:
        model, tokenizer = get_model_tokenizer(model_path)
        model = Swift.from_pretrained(model, lora_path)
        template = get_template(model.model_meta.template, tokenizer, default_system=None)
        engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    else:
        engine = PtEngine(model_path, max_batch_size=1)
    return engine, RequestConfig(max_tokens=1024, temperature=0)


def load_json_or_jsonl(path: str):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(dataset_name: str, src_file: str, image_dir: str | None):
    raw_data = load_json_or_jsonl(src_file)
    dataset = []

    if dataset_name == "ChartX":
        for item in raw_data:
            qa = item.get("QA")
            if not qa:
                continue
            dataset.append({
                "image": os.path.join(image_dir or "", item["img"]),
                "query": qa["input"],
                "gt": qa["output"],
            })
    elif dataset_name == "ChartBench":
        for item in raw_data:
            if item.get("type", {}).get("task") == "CR":
                continue
            for conv in item.get("conversation", []):
                query = conv["query"]
                if item.get("type", {}).get("QA") == "Acc+":
                    query += " The answer should be yes or no."
                dataset.append({"image": item["image"], "query": query, "gt": conv["label"]})
    elif dataset_name in {"ChartQA_h", "ChartQA_a"}:
        if image_dir is None:
            raise ValueError("ChartQA evaluation requires --image-dir")
        for item in raw_data:
            dataset.append({
                "image": os.path.join(image_dir, item["imgname"]),
                "query": item["query"],
                "gt": item["label"],
            })
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset


def infer_single(item, engine, request_config, system_prompt):
    req = InferRequest(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["query"]},
        ],
        images=[item["image"]],
    )
    resp = engine.infer([req], request_config)
    output = dict(item)
    output["model_response"] = resp[0].choices[0].message.content
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--dataset-name", required=True, choices=["ChartQA_h", "ChartQA_a", "ChartX", "ChartBench"])
    parser.add_argument("--src-file", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT
    if args.system_prompt:
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    engine, request_config = build_engine(args.model, args.lora)
    dataset = build_dataset(args.dataset_name, args.src_file, args.image_dir)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(infer_single, item, engine, request_config, system_prompt) for item in dataset]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            results.append(fut.result())
            if len(results) % 100 == 0:
                with open(args.output, "w", encoding="utf-8") as f:
                    for row in results:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.output, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
