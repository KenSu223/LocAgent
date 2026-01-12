import argparse
import ast
import json
import os
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download


REPO_ID = "somethingone/MULocBench"
DEFAULT_INPUT_FILE = "all_issues_with_pr_commit_comment_all_project_0922.json"


def load_mulocbench(input_path: Optional[str]) -> List[Dict]:
    if input_path:
        path = input_path
    else:
        path = hf_hub_download(
            repo_id=REPO_ID, filename=DEFAULT_INPUT_FILE, repo_type="dataset"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_instance_id(issue: Dict) -> str:
    issue_num = str(issue.get("iss_html_url", "")).rstrip("/").split("/")[-1]
    organization = issue.get("organization", "unknown-org")
    repo_name = issue.get("repo_name", "unknown-repo")
    return f"{organization}__{repo_name}-{issue_num}"


def build_problem_statement(issue: Dict) -> str:
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def parse_loc_key(loc_key: str) -> Optional[Dict[str, Optional[str]]]:
    try:
        cls_name, func_name, _line = ast.literal_eval(loc_key)
    except Exception:
        return None

    cls_name = None if cls_name in (None, "None") else str(cls_name)
    func_name = None if func_name in (None, "None") else str(func_name)
    return {"class": cls_name, "function": func_name}


def build_file_changes(issue: Dict) -> List[Dict]:
    file_changes: List[Dict] = []
    for file_entry in issue.get("file_loc", {}).get("files", []):
        path = file_entry.get("path")
        if not path:
            continue

        loc_map = file_entry.get("Loc") or {}
        entities: List[str] = []
        modules: List[str] = []

        for loc_key in loc_map.keys():
            if loc_key == "(None, None, None)":
                continue
            parsed = parse_loc_key(loc_key)
            if not parsed:
                continue

            cls_name = parsed["class"]
            func_name = parsed["function"]
            if not cls_name and not func_name:
                continue

            if func_name and cls_name:
                entity = f"{cls_name}.{func_name}"
            elif func_name:
                entity = func_name
            else:
                entity = cls_name

            if entity not in entities:
                entities.append(entity)
            module = entity.split(".")[0]
            if module not in modules:
                modules.append(module)

        changes: Dict[str, List[str]] = {}
        if modules:
            changes["edited_modules"] = [f"{path}:{m}" for m in modules]
        if entities:
            changes["edited_entities"] = [f"{path}:{e}" for e in entities]

        file_changes.append({"file": path, "changes": changes})
    return file_changes


def convert_issue(issue: Dict) -> Dict:
    instance_id = make_instance_id(issue)
    repo_full = f"{issue.get('organization')}/{issue.get('repo_name')}"
    return {
        "instance_id": instance_id,
        "repo": repo_full,
        "base_commit": issue.get("base_commit"),
        "problem_statement": build_problem_statement(issue),
        "patch": "",
        "file_loc": issue.get("file_loc"),
    }


def build_gt_record(issue: Dict, file_changes: List[Dict]) -> Dict:
    instance_id = make_instance_id(issue)
    repo_full = f"{issue.get('organization')}/{issue.get('repo_name')}"
    return {
        "instance_id": instance_id,
        "file_changes": file_changes,
        "repo": repo_full,
        "base_commit": issue.get("base_commit"),
        "problem_statement": build_problem_statement(issue),
        "patch": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to MULocBench JSON file. If omitted, download from Hugging Face.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/datasets/mulocbench",
        help="Directory to write converted dataset and GT files.",
    )
    args = parser.parse_args()

    issues = load_mulocbench(args.input_file)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_path = os.path.join(args.output_dir, "mulocbench.jsonl")
    gt_path = os.path.join(args.output_dir, "gt_location.jsonl")

    with open(dataset_path, "w", encoding="utf-8") as dataset_f, open(
        gt_path, "w", encoding="utf-8"
    ) as gt_f:
        for issue in issues:
            dataset_rec = convert_issue(issue)
            file_changes = build_file_changes(issue)
            gt_rec = build_gt_record(issue, file_changes)

            dataset_f.write(json.dumps(dataset_rec) + "\n")
            gt_f.write(json.dumps(gt_rec) + "\n")

    print(f"Wrote converted dataset to: {dataset_path}")
    print(f"Wrote ground-truth locations to: {gt_path}")


if __name__ == "__main__":
    main()
