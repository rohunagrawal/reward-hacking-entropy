import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def load_local_metrics(path: str, max_step: int) -> pd.DataFrame:
    rows: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            if step is None or step > max_step:
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def load_wandb_run(run_path: str, max_step: int) -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(run_path)
    df = run.history(keys=["step", "reward/mean", "reward/g_minus_f", "reward/g", "reward/f"])
    return df[df["step"] <= max_step]


def main():
    """
    Plot reward metrics for Llama-judge (local) and Qwen-judge (W&B) runs up to max_step
    into figures/reward_comparison.png.
    """
    max_step = 35

    # Define sources: (label, source_type, path_or_run)
    sources = [
        ("Easy g (Llama judge)", "local", "outputs/rl-leetcode/llama-3.2-3b/Easy-g-llmjudge-long/metrics.jsonl"),
        ("Medium g (Llama judge)", "local", "outputs/rl-leetcode/llama-3.2-3b/Medium-g-llmjudge-long/metrics.jsonl"),
        ("Hard g (Llama judge)", "local", "outputs/rl-leetcode/llama-3.2-3b/Hard-g-llmjudge-long/metrics.jsonl"),
        # Qwen judge runs on W&B (use run paths with entity/project/run_id)
        ("Easy g (Qwen judge)", "wandb", "lorena-yantianyi1020/COMS4705-reward-hacking-entropy/g75gkksf"),
        ("Medium g (Qwen judge)", "wandb", "lorena-yantianyi1020/COMS4705-reward-hacking-entropy/zajzblc0"),
        ("Hard g (Qwen judge)", "wandb", "lorena-yantianyi1020/COMS4705-reward-hacking-entropy/8z9aqxur"),
    ]

    data = []
    for label, src_type, ref in sources:
        try:
            if src_type == "local":
                if not os.path.exists(ref):
                    print(f"Warning: {ref} not found, skipping {label}.")
                    continue
                df = load_local_metrics(ref, max_step)
            elif src_type == "wandb":
                df = load_wandb_run(ref, max_step)
            else:
                continue
            if df.empty:
                print(f"Warning: no data for {label} up to step {max_step}, skipping.")
                continue
            data.append((label, df))
        except Exception as e:
            print(f"Warning: failed to load {label} ({src_type}): {e}")

    if not data:
        print("No data to plot.")
        return

    metrics = [
        ("reward/mean", "Reward / mean"),
        ("reward/g_minus_f", "Reward / g - f"),
        ("reward/g", "Reward / g"),
        ("reward/f", "Reward / f"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("tab10")

    # First three entries: Llama (Easy/Medium/Hard), next three: Qwen (Easy/Medium/Hard)
    colors = ["#d62728", "#9467bd", "#2ca02c", "#17becf", "#1f77b4", "#ff7f0e"]

    for idx, (metric_key, title) in enumerate(metrics):
        ax = axes[idx]
        for i, (label, df) in enumerate(data):
            if metric_key not in df:
                continue
            ax.plot(df["step"], df[metric_key], label=label, color=colors[i % len(colors)], linewidth=1.8, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(metric_key.split("/")[-1], fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "reward_comparison.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
    os.makedirs("/tmp/mplcache", exist_ok=True)
    main()
