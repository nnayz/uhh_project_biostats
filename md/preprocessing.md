# Preprocessing & Feature Standardization Summary

**Contributor:** Aqsa

---

## 1. Objective of This Phase

The goal of this phase was to:

- Inspect and validate the raw Xenium spatial transcriptomics dataset
- Determine whether the dataset is suitable for supervised (fine-tuning) or unsupervised (zero-shot) use
- Prepare clean, standardized, model-ready data without introducing incorrect assumptions

This phase focuses **only on preprocessing**, not on model training or evaluation.

---

## 2. Dataset Used

- **Dataset:** Xenium Mouse Brain Replicates
- **Format:** `.h5ad` (AnnData)
- **Cells:** 474,734
- **Genes:** 248 (targeted Xenium panel)
- **Replicates:** 3 (stored in `library_key`)

---

## 3. Label Availability Check

I systematically inspected all metadata columns in `adata.obs` to determine whether the dataset contains usable labels for supervised learning.

### Key findings:

- Columns such as `niche` and `region` exist but contain **only a single unique value**
- No column contains **multiple meaningful class values**
- Columns like `donor_id`, `condition_id`, and `library_key` describe data provenance or experimental setup, not biological classes

### Conclusion:

❌ The dataset is **unlabeled**  
✔️ It is **not suitable for supervised or federated fine-tuning**  
✔️ It **is suitable for zero-shot application**

---

## 4. Preprocessing Performed

Since the dataset is unlabeled, preprocessing was performed **without any label encoding**.

### 4.1 Expression Normalization

- Applied **total-count normalization** (target sum = 10,000)
- Applied **log1p transformation**
- Purpose: make gene expression values comparable across cells and remove sequencing depth bias

### 4.2 Feature Preservation

The following information was preserved:

- **Gene expression** (all 248 genes retained; no HVG selection)
- **Spatial coordinates:** `x`, `y`
- **Replicate information:** `library_key` (used as federated client ID)

No labels were created or inferred.

---

## 5. Output Files Generated

The preprocessing step produces the following files in `data/processed/`:
