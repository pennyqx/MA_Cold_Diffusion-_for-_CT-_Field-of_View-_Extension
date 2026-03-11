#!/usr/bin/env python3
"""Aggregate multi-seed eval outputs from eval_baseline_check.py."""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def _collect_metric_values(seed_dirs, metric_file):
    values = {}
    used_seeds = []
    missing = []

    for seed_dir in seed_dirs:
        fp = seed_dir / metric_file
        if not fp.exists():
            missing.append(seed_dir.name)
            continue

        used_seeds.append(seed_dir.name)
        with open(fp, "r") as f:
            payload = json.load(f)

        for model_name, model_payload in payload.items():
            metrics = model_payload.get("mean", {})
            values.setdefault(model_name, {})
            for metric_name, metric_value in metrics.items():
                values[model_name].setdefault(metric_name, []).append(float(metric_value))

    return values, used_seeds, missing


def _summarize(values):
    out = {}
    for model_name, metric_map in values.items():
        out[model_name] = {}
        for metric_name, arr in metric_map.items():
            if not arr:
                continue
            out[model_name][metric_name] = {
                "mean": float(mean(arr)),
                "std": float(pstdev(arr)) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed eval metrics")
    parser.add_argument("--input_root", required=True, help="Root folder containing seed_* dirs")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    seed_dirs = sorted([p for p in input_root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {input_root}")

    fbp_values, fbp_used, fbp_missing = _collect_metric_values(seed_dirs, "fbp_metrics.json")
    sino_values, sino_used, sino_missing = _collect_metric_values(seed_dirs, "sino_metrics.json")

    result = {
        "input_root": str(input_root),
        "seed_dirs": [p.name for p in seed_dirs],
        "fbp": {
            "used_seed_dirs": fbp_used,
            "missing_seed_dirs": fbp_missing,
            "summary": _summarize(fbp_values),
        },
        "sino": {
            "used_seed_dirs": sino_used,
            "missing_seed_dirs": sino_missing,
            "summary": _summarize(sino_values),
        },
    }

    out_path = Path(args.output) if args.output else (input_root / "seed_aggregate.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Save] Multi-seed aggregate written to {out_path}")


if __name__ == "__main__":
    main()
