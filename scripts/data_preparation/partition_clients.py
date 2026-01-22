import os
import json
import numpy as np
import pandas as pd

# -----------------------------
# Paths and config
# -----------------------------
IN_PATH = os.path.join("data", "processed", "processed_table.parquet")
OUT_DIR = os.path.join("data", "processed")
CLIENTS_DIR = os.path.join(OUT_DIR, "clients")
GLOBAL_DIR = os.path.join(OUT_DIR, "global")

CLIENT_COL = "sample_id"
LABEL_COL = "label"

GLOBAL_TEST_RATIO = 0.2        # 20% global test
CLIENT_TRAIN_VAL = (0.9, 0.1)  # 90% train, 10% val (per client)
SEED = 42

os.makedirs(CLIENTS_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# -----------------------------
# Helper: stratified split
# -----------------------------
def stratified_split(df, label_col, ratios):
    parts = [[] for _ in ratios]
    for _, sub in df.groupby(label_col):
        idx = sub.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        splits = [int(r * n) for r in ratios[:-1]]
        splits.append(n - sum(splits))
        start = 0
        for i, size in enumerate(splits):
            parts[i].append(df.loc[idx[start:start + size]])
            start += size
    return [pd.concat(p).reset_index(drop=True) for p in parts]

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(IN_PATH)
print(f"Loaded {len(df)} rows")

# -----------------------------
# 1️⃣ GLOBAL TEST SPLIT (ONCE)
# -----------------------------
train_pool, global_test = stratified_split(
    df,
    LABEL_COL,
    ratios=(1 - GLOBAL_TEST_RATIO, GLOBAL_TEST_RATIO)
)

global_test_path = os.path.join(GLOBAL_DIR, "test.parquet")
global_test.to_parquet(global_test_path, index=False)

print(f"Global test set: {len(global_test)} samples")

# -----------------------------
# 2️⃣ FEDERATED CLIENT SPLIT
# -----------------------------
global_meta = {}

for i, (client_id, cdf) in enumerate(train_pool.groupby(CLIENT_COL), start=1):
    client_name = f"client_{i:02d}"
    cdir = os.path.join(CLIENTS_DIR, client_name)
    os.makedirs(cdir, exist_ok=True)

    train_df, val_df = stratified_split(
        cdf,
        LABEL_COL,
        ratios=CLIENT_TRAIN_VAL
    )

    train_df.to_parquet(os.path.join(cdir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(cdir, "val.parquet"), index=False)

    meta = {
        "client_name": client_name,
        "group_value": str(client_id),
        "split_axis": CLIENT_COL,
        "n_total": int(len(cdf)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "has_test": False,
        "label_counts_total": cdf[LABEL_COL].value_counts().to_dict(),
    }

    with open(os.path.join(cdir, "client_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    global_meta[client_name] = meta

# -----------------------------
# Save global metadata
# -----------------------------
global_info = {
    "global_test_size": int(len(global_test)),
    "global_test_path": global_test_path,
    "num_clients": len(global_meta),
    "seed": SEED,
}

with open(os.path.join(OUT_DIR, "global_metadata.json"), "w") as f:
    json.dump(global_info, f, indent=2)

print(f"Saved {len(global_meta)} clients")
print("✔ Global test split complete")
