import argparse
import json
import os
from typing import Iterable, Set


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_repo(repo: str) -> str:
    return repo.strip().lower()


def repo_matches(repo_field: str, repo_names: Set[str]) -> bool:
    repo_field = normalize_repo(repo_field)
    if repo_field in repo_names:
        return True
    for name in repo_names:
        if repo_field.endswith(f"/{name}"):
            return True
    return False


def filter_dataset(
    input_dataset: str,
    output_dataset: str,
    repo_names: Set[str],
) -> Set[str]:
    instance_ids: Set[str] = set()
    with open(output_dataset, "w", encoding="utf-8") as out_f:
        for record in load_jsonl(input_dataset):
            repo_field = record.get("repo", "")
            if not repo_field:
                continue
            if repo_matches(repo_field, repo_names):
                instance_ids.add(record["instance_id"])
                out_f.write(json.dumps(record) + "\n")
    return instance_ids


def filter_gt(
    input_gt: str,
    output_gt: str,
    instance_ids: Set[str],
) -> None:
    with open(output_gt, "w", encoding="utf-8") as out_f:
        for record in load_jsonl(input_gt):
            if record.get("instance_id") in instance_ids:
                out_f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", required=True, help="Path to mulocbench.jsonl")
    parser.add_argument("--input_gt", required=True, help="Path to gt_location.jsonl")
    parser.add_argument(
        "--repos",
        required=True,
        help="Comma-separated repo names (e.g., scikit-learn,flask).",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_dataset = os.path.join(args.output_dir, "mulocbench_subset.jsonl")
    output_gt = os.path.join(args.output_dir, "gt_location.jsonl")
    output_ids = os.path.join(args.output_dir, "selected_instance_ids.json")

    repo_names = {normalize_repo(name) for name in args.repos.split(",") if name.strip()}
    instance_ids = filter_dataset(args.input_dataset, output_dataset, repo_names)
    filter_gt(args.input_gt, output_gt, instance_ids)

    with open(output_ids, "w", encoding="utf-8") as f:
        json.dump(sorted(instance_ids), f, indent=2)

    print(f"Wrote subset dataset to: {output_dataset}")
    print(f"Wrote subset ground truth to: {output_gt}")
    print(f"Wrote {len(instance_ids)} instance ids to: {output_ids}")


if __name__ == "__main__":
    main()
