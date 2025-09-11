# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

import argparse
import json
import os

import numpy as np
import pandas as pd
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile results for a specific benchmark mode."
    )
    parser.add_argument(
        "benchmark_mode",
        type=str,
        help="Benchmark mode to filter results (e.g., lighting-ll).",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work-dir",
        help="Directory where results are stored. Default: work_dirs",
    )
    parser.add_argument(
        "--metrics",
        type=str.upper,
        choices=["AP", "AR", "ALL"],
        default="AP",
        help="Metrics to include in the table (AP, AR, ALL). Default: AP",
    )
    return parser.parse_args()


def find_results_dirs(results_dir, model_name, benchmark_mode, model_resolution):
    all_results = [
        f
        for f in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, f))
    ]
    return [
        r
        for r in all_results
        if r.startswith(f"{model_name}_poseadapt-{benchmark_mode}_")
        and r.endswith(f"_{model_resolution}")
    ]


def _add_records_from_line(records, line, train_exp_idx, global_step):
    """Extract <experience>/<metric> keys from a single JSON line."""
    for k, v in line.items():
        if k == "step":
            continue
        if not isinstance(v, (int, float)):
            continue
        if "/" not in k:
            continue
        exp_name, metric_suffix = k.split("/", 1)
        if exp_name == "o5":
            exp_name = "o05"
        records.append(
            {
                "train_exp_idx": train_exp_idx,
                "global_step": global_step,
                "exp_name": exp_name,
                "metric_suffix": metric_suffix,
                "value": float(v),
            }
        )


def _load_jsonl_results_to_df(results_file):
    """Load JSONL (vis_data) results into a DataFrame.

    - Handles step reset per experience by constructing a monotonic global_step.
    - Keeps exactly one snapshot per training experience: the last epoch.
    """
    records = []

    with open(results_file, "r") as f:
        lines = [json.loads(line) for line in f]

    if not lines:
        return None

    train_exp_idx = 0
    prev_step = None
    step_offset = 0
    current_last_line = None
    current_last_step = None

    for data in lines:
        if "coco/AP" not in data:
            continue

        step = data.get("step")
        if step is None:
            continue

        # New experience detected when local step decreases
        if prev_step is not None and step < prev_step:
            # Finalize previous experience snapshot
            if current_last_line is not None:
                global_step = current_last_step + step_offset
                _add_records_from_line(
                    records, current_last_line, train_exp_idx, global_step
                )
                # Update offset so that global_step keeps increasing
                step_offset += current_last_step + 1

            train_exp_idx += 1
            current_last_line = data
            current_last_step = step
        else:
            # Same experience: always keep the last line
            current_last_line = data
            current_last_step = step

        prev_step = step

    # Finalize last experience
    if current_last_line is not None and current_last_step is not None:
        global_step = current_last_step + step_offset
        _add_records_from_line(records, current_last_line, train_exp_idx, global_step)

    if not records:
        return None

    return pd.DataFrame.from_records(records)


def _load_json_results_to_df(results_file):
    """Load single JSON snapshot (final result only) into a DataFrame.

    We treat this as a single training step (train_exp_idx=0, global_step=0).
    This allows RA, but BWT will be undefined (NaN) because there is no trajectory.
    """
    with open(results_file, "r") as f:
        data = json.load(f)

    records = []
    for k, v in data.items():
        if not isinstance(v, (int, float)):
            continue
        if "/" not in k:
            continue
        exp_name, metric_suffix = k.split("/", 1)
        if exp_name == "o5":
            exp_name = "o05"
        records.append(
            {
                "train_exp_idx": 0,
                "global_step": 0,
                "exp_name": exp_name,
                "metric_suffix": metric_suffix,
                "value": float(v),
            }
        )

    if not records:
        return None

    return pd.DataFrame.from_records(records)


def load_results_df(result_path):
    """Load results from a given experiment directory into a normalized DataFrame.

    The function prefers the vis_data JSONL file (full trajectory). If absent,
    it falls back to the plain JSON final snapshot.
    """
    experiment_dirs = [
        d
        for d in sorted(os.listdir(result_path), reverse=True)
        if os.path.isdir(os.path.join(result_path, d)) and d != "experiences"
    ]
    if not experiment_dirs:
        return None

    latest_dir = os.path.join(result_path, experiment_dirs[0])

    # 1) Try JSONL in vis_data
    jsonl_file = os.path.join(latest_dir, "vis_data", f"{experiment_dirs[0]}.json")
    if os.path.exists(jsonl_file):
        df = _load_jsonl_results_to_df(jsonl_file)
        if df is not None:
            return df

    # 2) Fallback: plain JSON snapshot
    json_file = os.path.join(latest_dir, f"{experiment_dirs[0]}.json")
    if os.path.exists(json_file):
        return _load_json_results_to_df(json_file)

    return None


def _metric_selected(metric_suffix: str, requested: str) -> bool:
    """Filter which metric suffixes to keep, based on CLI flag.

    Here we assume standard suffixes 'AP' and 'AR' (as in coco/AP, dark/AP).
    """
    metric_suffix = metric_suffix.upper()
    requested = requested.upper()

    if requested == "ALL":
        return metric_suffix in {"AP", "AR"}
    return metric_suffix == requested


def _build_performance_matrix(df: pd.DataFrame, metric_suffix: str):
    """Build R matrix for a given metric suffix.

    R[i, j] = performance on experience j after learning experience i.

    Assumptions:
    - train_exp_idx encodes experience order (0..T-1)
    - exp_name appears in order of learning, and there is a one-to-one
      alignment between training experience index and new exp_name.
    """
    df_m = df[df["metric_suffix"].str.upper() == metric_suffix.upper()]
    if df_m.empty:
        return None, None

    # Training experience order
    train_ids = sorted(df_m["train_exp_idx"].unique())
    n_train = len(train_ids)
    train_id_to_i = {tid: i for i, tid in enumerate(train_ids)}

    # Experience names in order of first appearance
    first_idx = {}
    ordered_exp_names = []
    for _, row in df_m.sort_values(["train_exp_idx", "global_step"]).iterrows():
        name = row["exp_name"]
        if name not in first_idx:
            first_idx[name] = len(ordered_exp_names)
            ordered_exp_names.append(name)

    n_exp = len(ordered_exp_names)
    exp_name_to_j = {n: j for j, n in enumerate(ordered_exp_names)}

    R = np.full((n_train, n_exp), np.nan, dtype=float)

    for _, row in df_m.iterrows():
        i = train_id_to_i[row["train_exp_idx"]]
        j = exp_name_to_j[row["exp_name"]]
        R[i, j] = float(row["value"])

    return R, ordered_exp_names


def _compute_cl_metrics_from_matrix(
    R: np.ndarray, metric_suffix: str, method: str, exp_names: list
):
    """
    Compute RA, BWT from an R matrix according to standard definitions.

      RA  = (1/T) * sum_t R[T-1, t]
      BWT = (1/(T-1)) * sum_{t=0..T-2} ( R[T-1, t] - R[t, t] )
    """
    metric_suffix = metric_suffix.upper()
    n_train, n_exp = R.shape
    T = min(n_train, n_exp)

    initial_perf = "0.70064932"
    row0 = np.array([initial_perf] + [None] * (n_exp - 1), dtype=float)
    R = np.vstack([row0, R])
    T += 1

    exp_names = [
        {
            "o05": "O5",
            "o10": "O10",
            "o20": "O20",
            "gray": "Gray",
            "depth": "Depth",
            "dark": "LL",
            "darker": "VLL",
            "darkest": "ELL",
            "coco": "Ref.",
        }[n]
        for n in exp_names
    ]

    # # if this is a square matrix, save it is a heatmap image (NANs as white)
    # if R.shape[0] == R.shape[1]:
    #     import matplotlib.pyplot as plt

    #     plt.figure(figsize=(4, 4))
    #     plt.imshow(R, cmap="Reds", vmin=0.2, vmax=0.8)
    #     # plt.title(f"{method.upper()}")
    #     # plt.xlabel("Experience j")
    #     # plt.ylabel("After Experience i")
    #     plt.xticks(ticks=np.arange(T), labels=exp_names)
    #     plt.yticks(ticks=np.arange(T), labels=exp_names)
    #     for i in range(T):
    #         for j in range(T):
    #             val = R[i, j]
    #             if not np.isnan(val):
    #                 plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black",
    #                 fontsize=11)
    #     plt.tight_layout()
    #     plt.savefig(f"performance_matrix_R_{metric_suffix}_{method}.png")
    #     plt.close()

    # ---------------------------
    # RA
    # ---------------------------
    final_row = R[T - 1, :]
    valid = ~np.isnan(final_row)
    ra = float(np.mean(final_row[valid])) if np.any(valid) else float("nan")

    # If only one experience, BWT is undefined
    if T <= 1:
        return {
            f"C-RA/{metric_suffix}": ra,
            f"C-AF/{metric_suffix}": float("nan"),
        }

    # ---------------------------
    # BWT
    # ---------------------------
    diag = np.array([R[t, t] for t in range(T)], dtype=float)
    final_perf = R[T - 1, :T]

    mask = ~np.isnan(diag[:-1]) & ~np.isnan(final_perf[:-1])
    if np.any(mask):
        diffs = final_perf[:-1][mask] - diag[:-1][mask]
        bwt = float(np.mean(diffs))
    else:
        bwt = float("nan")

    return {
        f"C-RA/{metric_suffix}": ra,
        f"C-AF/{metric_suffix}": bwt * -1,
    }


def compute_cl_summary(df: pd.DataFrame, requested_metrics: str, method: str):
    """From per-experience DataFrame, compute:

    - Final metrics for each <experience>/<metric_suffix> at last step
    - Continual-learning metrics C-RA, C-BWT, C-AF for AP/AR where possible
    """
    summary = {}

    # Final snapshot (after last training experience)
    last_train_idx = df["train_exp_idx"].max()
    df_final = df[df["train_exp_idx"] == last_train_idx]

    for _, row in df_final.iterrows():
        metric_suffix = row["metric_suffix"]
        if not _metric_selected(metric_suffix, requested_metrics):
            continue
        key = f"{row['exp_name']}/{metric_suffix}"
        summary[key] = float(row["value"])

    # Continual learning metrics per metric type (AP, AR)
    for metric_suffix in ("AP", "AR"):
        if not _metric_selected(metric_suffix, requested_metrics):
            continue

        R, exp_names = _build_performance_matrix(df, metric_suffix)
        if R is None:
            continue

        cl_metrics = _compute_cl_metrics_from_matrix(
            R, metric_suffix, method, exp_names
        )
        summary.update(cl_metrics)

    return summary


def main():
    args = parse_args()

    # Config
    model_name = "rtmpose-t_8xb32-10e"
    model_resolution = "256x192"
    results_dir = args.work_dir

    # Gather results
    results_list = find_results_dirs(
        results_dir, model_name, args.benchmark_mode, model_resolution
    )
    summary = {}

    for result in sorted(results_list):
        df = load_results_df(os.path.join(results_dir, result))
        if df is None or df.empty:
            continue

        # Strategy extraction
        cl_strategy = (
            result.split(f"{model_name}_poseadapt-{args.benchmark_mode}")[-1]
            .rsplit(f"_{model_resolution}", 1)[0]
            .lstrip("_")
        ) or "ft"

        # Avoid name collisions
        i = 0
        base_strategy = cl_strategy
        while cl_strategy.upper() in summary:
            i += 1
            cl_strategy = f"{base_strategy}_{i}"

        cl_metrics = compute_cl_summary(
            df, args.metrics, f"{args.benchmark_mode}-{cl_strategy}"
        )
        summary[cl_strategy.upper()] = cl_metrics

    # Prepare rows & metrics list
    rows = []
    all_metrics = set()
    for cl_strategy, results in summary.items():
        row = {"Strategy": cl_strategy}
        row.update(results)
        all_metrics.update(results.keys())
        rows.append(row)

    # Sort rows: PT, FT, then others
    def sort_key(row):
        if row["Strategy"] == "PT":
            return (0, row["Strategy"])
        if row["Strategy"] == "FT":
            return (1, row["Strategy"])
        return (2, row["Strategy"])

    rows.sort(key=sort_key)

    # Sort metrics so C-* summary metrics (RA/BWT/AF) come last
    metrics = sorted(all_metrics, key=lambda x: (x.startswith("C-"), x))
    headers = ["Strategy"] + metrics

    # Build table data (values in percentage)
    table_data = []
    for row in rows:
        table_row = [
            row["Strategy"],
            *[
                f"{row.get(m, float('nan')) * 100:0.2f}"
                if not np.isnan(row.get(m, float("nan")))
                else "nan"
                for m in metrics
            ],
        ]
        table_data.append(table_row)

        # Insert a separator after FT row
        if row["Strategy"] == "FT":
            table_data.append(["-" * len(h) for h in headers])

    # Print LaTeX table
    print(
        tabulate(
            table_data,
            headers=headers,
            tablefmt="latex_booktabs",
            stralign="center",
            numalign="center",
        )
    )
    print(f"\nBenchmark: {args.benchmark_mode} ({model_resolution})")


if __name__ == "__main__":
    main()
