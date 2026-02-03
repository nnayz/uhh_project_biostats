"""
Partition Mouse Brain data by Anatomical Siloing (Non-IID).

Strategy:
- Hold out one replicate (e.g. Replicate 3) entirely → zero-shot evaluation set.
- Use the other two replicates only; split by Y-coordinate (vertical position) into
  three anatomical regions → three non-IID clients:
  - Client A (Dorsal): Cortex (high Y)
  - Client B (Mid): Hippocampus, Thalamus (middle Y)
  - Client C (Ventral): Hypothalamus, Basal Ganglia (low Y)
- Within each client: 80% train / 20% val (stratified by label).

Mouse brain metadata uses library_key for the 3 replicates. Preprocess with --batch_col library_key
so batch_id in the parquet = library_key (replicate ID).

Usage:
  python scripts/data_preparation/preprocess.py --raw_path data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad --batch_col library_key
  python scripts/data_preparation/partition_anatomical_siloing.py --hold_out_replicate <third_library_key_value>
  python scripts/data_preparation/partition_anatomical_siloing.py --hold_out_replicate "3" --replicate_col batch_id
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

IN_PATH = os.path.join("data", "processed", "processed_table.parquet")
OUT_DIR = os.path.join("data", "processed", "clients")
HELD_OUT_PATH = os.path.join("data", "processed", "held_out_batch.parquet")
REPLICATE_COL = "batch_id"  # from preprocess: batch_id = library_key (replicate)
Y_COL = "y"
LABEL_COL = "label"
TRAIN_VAL_SPLIT = (0.8, 0.2)  # 80% train / 20% val within each client (stratified by label)

# Anatomical regions by Y (dorsal = high Y, ventral = low Y)
REGIONS = ["dorsal", "mid", "ventral"]  # client_01, client_02, client_03


def main():
    parser = argparse.ArgumentParser(description="Partition mouse brain by anatomical siloing (Y-based)")
    parser.add_argument("--hold_out_replicate", type=str, default="replicate 3", help="Replicate (library_key) value to hold out; mouse brain often has 'replicate 1', 'replicate 2', 'replicate 3'")
    parser.add_argument("--replicate_col", type=str, default=REPLICATE_COL, help="Column with replicate ID (from preprocess batch_col, e.g. batch_id = library_key)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(IN_PATH)

    for c in [args.replicate_col, Y_COL, LABEL_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Required: {args.replicate_col}, {Y_COL}, {LABEL_COL}")

    rep_col = args.replicate_col
    # Normalize replicate to string for comparison
    df = df.copy()
    df[rep_col] = df[rep_col].astype(str)

    replicate_ids = sorted(df[rep_col].unique().tolist())
    hold_out = args.hold_out_replicate.strip()
    if hold_out not in replicate_ids:
        # Try "3" -> "replicate 3", "1" -> "replicate 1", etc. (mouse brain library_key style)
        fallback = f"replicate {hold_out}" if hold_out in ("1", "2", "3") else None
        if fallback and fallback in replicate_ids:
            hold_out = fallback
        else:
            raise ValueError(
                f"hold_out_replicate '{args.hold_out_replicate}' not in {rep_col}. "
                f"Found: {replicate_ids}. Use one of these exactly, e.g. --hold_out_replicate \"replicate 3\""
            )

    # Eval set = entire held-out replicate
    eval_df = df[df[rep_col] == hold_out].copy()
    eval_df.to_parquet(HELD_OUT_PATH, index=False)
    eval_meta = {
        "strategy": "anatomical_siloing",
        "held_out_replicate": hold_out,
        "n_samples": int(len(eval_df)),
        "label_distribution": eval_df[LABEL_COL].value_counts().astype(int).to_dict(),
    }
    with open(os.path.join("data", "processed", "held_out_batch_meta.json"), "w") as f:
        json.dump(eval_meta, f, indent=2)
    print(f"[OK] Held-out eval set ({hold_out}): {HELD_OUT_PATH} ({len(eval_df):,} samples)")

    # Client pool = Replicate 1 and 2
    train_replicates = [r for r in replicate_ids if r != hold_out]
    client_pool = df[df[rep_col].isin(train_replicates)].copy()
    if len(client_pool) == 0:
        raise ValueError("No rows left for clients after holding out replicate.")

    # Y tertiles: dorsal (high Y) = top third, mid = middle, ventral (low Y) = bottom third
    y_vals = client_pool[Y_COL].values
    p33 = np.percentile(y_vals, 33.33)
    p66 = np.percentile(y_vals, 66.67)

    def assign_region(y):
        if y <= p33:
            return "ventral"
        if y <= p66:
            return "mid"
        return "dorsal"

    client_pool["anatomical_region"] = client_pool[Y_COL].map(assign_region)
    # For downstream compatibility, set batch_id to anatomical region so loaders see client = batch
    client_pool["batch_id"] = client_pool["anatomical_region"]

    rng = np.random.default_rng(args.seed)
    global_meta = {}

    for i, region in enumerate(REGIONS, start=1):
        client_name = f"client_{i:02d}"
        cdir = os.path.join(OUT_DIR, client_name)
        os.makedirs(cdir, exist_ok=True)

        cdf = client_pool[client_pool["anatomical_region"] == region].copy()
        if len(cdf) == 0:
            print(f"  [WARN] No cells for region {region}; skipping {client_name}")
            continue

        tr_parts, va_parts = [], []
        for _, sub in cdf.groupby(LABEL_COL):
            idx = sub.index.to_numpy()
            rng.shuffle(idx)
            n = len(idx)
            n_tr = int(n * TRAIN_VAL_SPLIT[0])
            tr_parts.append(cdf.loc[idx[:n_tr]])
            va_parts.append(cdf.loc[idx[n_tr:]])
        tr = pd.concat(tr_parts)
        va = pd.concat(va_parts)

        # Save without anatomical_region if we don't want to change schema; keep batch_id as region for clarity
        tr.to_parquet(os.path.join(cdir, "train.parquet"), index=False)
        va.to_parquet(os.path.join(cdir, "val.parquet"), index=False)

        meta = {
            "client_name": client_name,
            "batch_id": region,
            "anatomical_region": region,
            "group_value": region,
            "split_axis": "anatomical_siloing_y",
            "train_val_split": "80-20",
            "n_total": int(len(cdf)),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": 0,
            "label_counts_total": cdf[LABEL_COL].value_counts().astype(int).to_dict(),
        }
        with open(os.path.join(cdir, "client_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        global_meta[client_name] = meta
        print(f"  {client_name} ({region}): {len(tr):,} train, {len(va):,} val")

    with open(os.path.join("data", "processed", "global_metadata.json"), "w") as f:
        json.dump(global_meta, f, indent=2)

    with open(os.path.join("data", "processed", "partition_config.json"), "w") as f:
        json.dump({
            "strategy": "anatomical_siloing",
            "hold_out_batch_id": hold_out,
            "hold_out_replicate": hold_out,
            "client_batches": REGIONS,
            "all_batches": REGIONS + [hold_out],
            "y_percentiles": {"p33": float(p33), "p66": float(p66)},
            "seed": args.seed,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("Anatomical Siloing (Mouse Brain)")
    print("  Held-out: library_key =", hold_out, "(zero-shot eval)")
    print("  Clients: Dorsal (Cortex), Mid (Hippocampus/Thalamus), Ventral (Hypothalamus/Basal Ganglia)")
    print("  Split within each client: 80% train / 20% val (stratified by label)")
    print(f"  Saved {len(global_meta)} clients; eval set = library_key '{hold_out}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
