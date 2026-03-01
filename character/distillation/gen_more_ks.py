"""
Generate additional K rounds of teacher responses, saving each to a separate file.
Then merge all rounds into the main distillation file.

Usage:
    source /workspace/.env
    python -m character.distillation.gen_more_ks --constitution misalignment --extra_ks 2
"""

import os, argparse
import pandas as pd
from character.distillation.teacher_api import roleplay_api
from character.constants import DATA_PATH


def main(constitution: str, teacher_model: str, name: str, extra_ks: int):
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set. Source /workspace/.env first.")

    base_path = f"{DATA_PATH}/distillation/{constitution}.jsonl"

    for k in range(extra_ks):
        k_num = k + 2  # existing data is k1
        outpath = f"{DATA_PATH}/distillation/{constitution}_k{k_num}.jsonl"
        if os.path.exists(outpath):
            df = pd.read_json(outpath, orient="records", lines=True)
            if df["response"].notna().all() and len(df) > 0:
                print(f"k{k_num} already complete at {outpath} ({len(df)} rows), skipping", flush=True)
                continue
        print(f"\n{'='*60}", flush=True)
        print(f"Generating k{k_num} -> {outpath}", flush=True)
        print(f"{'='*60}", flush=True)
        roleplay_api(outpath, constitution, api_key, teacher_model, name, K=1)

    print(f"\nAll {extra_ks} extra rounds done. Merging...", flush=True)

    # Merge: load original + all extra rounds
    dfs = []
    if os.path.exists(base_path):
        dfs.append(pd.read_json(base_path, orient="records", lines=True))
        print(f"  k1: {len(dfs[-1])} rows", flush=True)

    for k in range(extra_ks):
        k_num = k + 2
        kpath = f"{DATA_PATH}/distillation/{constitution}_k{k_num}.jsonl"
        if os.path.exists(kpath):
            dfs.append(pd.read_json(kpath, orient="records", lines=True))
            print(f"  k{k_num}: {len(dfs[-1])} rows", flush=True)

    merged = pd.concat(dfs, ignore_index=True).dropna(subset=["response"])
    merged.to_json(base_path, orient="records", lines=True)
    print(f"\nMerged {len(merged)} total rows -> {base_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, default="misalignment")
    parser.add_argument("--teacher_model", type=str, default="zai-org/GLM-4.7")
    parser.add_argument("--name", type=str, default="GLM")
    parser.add_argument("--extra_ks", type=int, default=2)
    args = parser.parse_args()
    main(args.constitution, args.teacher_model, args.name, args.extra_ks)
