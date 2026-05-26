import json
import math
import re
from typing import Any, Dict, List, Optional

from swift.plugin import ORM, orms


class BaseCodeORM(ORM):
    """Base reward helper for executable chart-reasoning trajectories."""

    SAFE_MODULES = {"math", "statistics", "itertools", "functools", "collections"}

    @staticmethod
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_name = name.split(".")[0]
        if root_name not in BaseCodeORM.SAFE_MODULES:
            raise ImportError(f"Import of module '{name}' is not allowed.")
        return __import__(name, globals, locals, fromlist, level)

    SAFE_BUILTINS = {
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
        "__import__": _safe_import,
    }

    def extract_code(self, response: str) -> Optional[str]:
        match = re.search(r"<python>\s*(.*?)</python>", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def exec_code(self, code: str) -> Optional[Dict[str, Any]]:
        env = {"__builtins__": self.SAFE_BUILTINS, "math": math}
        try:
            exec(code, env, env)
            return env
        except Exception:
            return None


class PythonExecutableRewardORM(BaseCodeORM):
    def __call__(self, completions: List[str], solution: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for content, _ in zip(completions, solution):
            code = self.extract_code(content)
            rewards.append(1.0 if code and self.exec_code(code) is not None else 0.0)
        return rewards


class AnswerAccuracyRewardORM(BaseCodeORM):
    @staticmethod
    def parse_numeric(text: Any) -> Optional[float]:
        matches = re.findall(r"[-+]?\d*\.?\d+", str(text))
        return float(matches[0]) if len(matches) == 1 else None

    def compute_accuracy(self, prediction: Any, target: Any, tol: float = 0.05) -> int:
        prediction = str(prediction).strip().rstrip(".")
        target = str(target).strip()
        p_val = self.parse_numeric(prediction)
        t_val = self.parse_numeric(target)
        if p_val is not None and t_val is not None:
            threshold = abs(t_val) * tol
            return int(
                abs(p_val - t_val) <= threshold
                or abs(p_val * 100 - t_val) <= threshold
                or abs(p_val / 100 - t_val) <= threshold
            )
        return int(prediction.lower() == target.lower())

    def __call__(self, completions: List[str], solution: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            code = self.extract_code(content)
            local_vars = self.exec_code(code) if code else None
            if local_vars is None:
                rewards.append(0.0)
                continue
            pred_answer = str(local_vars.get("answer", ""))
            if pred_answer == "True":
                pred_answer = "yes"
            elif pred_answer == "False":
                pred_answer = "no"
            rewards.append(float(self.compute_accuracy(pred_answer, sol["label"])))
        return rewards


class VariableStructureRewardORM(BaseCodeORM):
    REQUIRED_VARS = ("table", "answer")

    def __call__(self, completions: List[str], solution: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for content, _ in zip(completions, solution):
            code = self.extract_code(content)
            local_vars = self.exec_code(code) if code else None
            if local_vars is None:
                rewards.append(0.0)
                continue
            reward = 0.75 if all(v in local_vars for v in self.REQUIRED_VARS) else 0.0
            extra_vars = [v for v in local_vars if v not in self.REQUIRED_VARS and v not in {"__builtins__", "math"}]
            if extra_vars:
                reward += 0.25
            rewards.append(reward)
        return rewards


class TableCorrectnessRewardORM(BaseCodeORM):
    @staticmethod
    def normalize_table(table: Any) -> Dict[str, Any]:
        if isinstance(table, str):
            table = json.loads(table)
        table = dict(table)
        rows = table.get("rows", {})
        normalized_rows = {}
        for key, values in rows.items():
            normalized_values = []
            for value in values:
                if isinstance(value, str):
                    try:
                        value = float(value) if "." in value else int(value)
                    except Exception:
                        pass
                normalized_values.append(value)
            normalized_rows[key] = normalized_values
        table["rows"] = normalized_rows
        return table

    @staticmethod
    def transpose_table(table: Dict[str, Any]) -> Dict[str, Any]:
        column_header = list(table["columns"].keys())[0]
        column_names = table["columns"][column_header]
        return {
            "columns": {column_header: list(table["rows"].keys())},
            "rows": {
                column_name: [table["rows"][row_key][col_idx] for row_key in table["rows"]]
                for col_idx, column_name in enumerate(column_names)
            },
        }

    def table_orientations(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = [table]
        try:
            candidates.append(self.transpose_table(table))
        except Exception:
            pass
        return candidates

    def compare_tables(self, pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
        reward = 0.0
        try:
            gt_col_key = list(gt["columns"].keys())[0].lower()
            pred_col_key = list(pred["columns"].keys())[0].lower()
            gt_vals = [str(c).lower() for c in gt["columns"][list(gt["columns"].keys())[0]]]
            pred_vals = [str(c).lower() for c in pred["columns"][list(pred["columns"].keys())[0]]]
            if gt_col_key == pred_col_key:
                reward += 0.05
            for gt_val, pred_val in zip(gt_vals, pred_vals):
                if gt_val == pred_val:
                    reward += 0.2 / max(len(gt_vals), len(pred_vals), 1)
        except Exception:
            pass

        try:
            gt_rows = gt["rows"]
            pred_rows = pred["rows"]
            total_cells = len(gt_rows) * (len(next(iter(gt_rows.values()))) if gt_rows else 0)
            if total_cells > 0:
                per_cell_reward = 0.5 / total_cells
                for key in set(gt_rows.keys()) & set(pred_rows.keys()):
                    for gt_val, pred_val in zip(gt_rows[key], pred_rows[key]):
                        if gt_val == pred_val:
                            reward += per_cell_reward
        except Exception:
            pass

        try:
            gt_rows = gt["rows"]
            pred_rows = pred["rows"]
            value_cols = list(gt["columns"].values())[0]
            if value_cols:
                per_col_reward = 0.125 / len(value_cols)
                for col_idx in range(len(value_cols)):
                    gt_col_vals, pred_col_vals = [], []
                    for key in gt_rows:
                        if key in pred_rows:
                            gt_col_vals.append(gt_rows[key][col_idx])
                            pred_col_vals.append(pred_rows[key][col_idx])
                    try:
                        gt_nums = [float(x) for x in gt_col_vals]
                        pred_nums = [float(x) for x in pred_col_vals]
                    except Exception:
                        continue
                    if sorted(range(len(gt_nums)), key=lambda i: gt_nums[i]) == sorted(range(len(pred_nums)), key=lambda i: pred_nums[i]):
                        reward += per_col_reward

            common_keys = set(gt_rows.keys()) & set(pred_rows.keys())
            if common_keys:
                per_row_reward = 0.125 / len(common_keys)
                for key in common_keys:
                    try:
                        gt_nums = [float(x) for x in gt_rows[key]]
                        pred_nums = [float(x) for x in pred_rows[key]]
                    except Exception:
                        continue
                    if sorted(range(len(gt_nums)), key=lambda i: gt_nums[i]) == sorted(range(len(pred_nums)), key=lambda i: pred_nums[i]):
                        reward += per_row_reward
        except Exception:
            pass
        return reward

    def compare_with_orientation_invariance(self, pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
        best_reward = 0.0
        for pred_candidate in self.table_orientations(pred):
            for gt_candidate in self.table_orientations(gt):
                best_reward = max(best_reward, self.compare_tables(pred_candidate, gt_candidate))
        return min(best_reward, 1.0)

    def __call__(self, completions: List[str], solution: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            code = self.extract_code(content)
            local_vars = self.exec_code(code) if code else None
            if local_vars is None or not isinstance(local_vars.get("table"), dict):
                rewards.append(0.0)
                continue
            gt_table = self.normalize_table(sol["table"])
            rewards.append(self.compare_with_orientation_invariance(local_vars["table"], gt_table))
        return rewards


orms["PythonExecutable_reward"] = PythonExecutableRewardORM
orms["AnswerAccuracy_reward"] = AnswerAccuracyRewardORM
orms["VariableStructure_reward"] = VariableStructureRewardORM
orms["TableCorrectness_reward"] = TableCorrectnessRewardORM
