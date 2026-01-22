# Data Dictionary: Xenium Mouse Brain Replicates

**Owner:** Aftab  
**Milestone:** 2 (Data Preparation & Partitioning)  
**Status:** ✅ Validated  
**Raw File:** `data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad`

## 1. Dataset Overview
This dataset represents the **Xenium Mouse Brain** benchmark. It is a high-resolution spatial transcriptomics dataset consisting of multiple technical replicates. In our Federated Learning setup, these replicates serve as independent **Clients/Sites**.

- **Total Cells:** 474,734
- **Total Genes:** 248 (Targeted Xenium Panel)
- **Technology:** 10x Genomics Xenium
- **Organism:** Mouse (*Mus musculus*)
- **Tissue:** Brain

## 2. Data Contract Mapping
To maintain consistency across the pipeline, use the following mapping from the raw AnnData object to the team's **Data Contract**:

| Contract Field | Raw Column Name | Type | Description |
| :--- | :--- | :--- | :--- |
| **sample_id** | `library_key` | category | **Primary Split Axis.** Each unique key is a federated client. |
| **label** | derived (`cluster`) | category | Pseudo-labels created via Leiden clustering during preprocessing. |
| **x** | `x` | float32 | Spatial X-coordinate (centroid). |
| **y** | `y` | float32 | Spatial Y-coordinate (centroid). |
| **donor_id** | `donor_id` | category | Biological source identifier. |

## 3. Metadata Catalog (`adata.obs`)
Additional fields available for analysis:
- `transcript_counts`: Total RNA counts per cell (useful for quality control).
- `cell_area`: Morphological area of the segmented cell.
- `region`: Anatomical brain region (e.g., "Cortex", "Midbrain").
- `dataset`: Originating study identifier.

---

## 4. Instructions for Preprocessing (Aqsa)
Based on the validation of the raw Xenium data, please follow these requirements for the `data/processed/` deliverables:

### A. Gene Filtering Strategy
- **Do not** perform Highly Variable Gene (HVG) selection. 
- Because this is a **Targeted Panel (248 genes)** and not whole-transcriptome, every gene is pre-selected for biological significance.
- **Task:** Retain all 248 genes. Ensure they are listed in alphabetical or original order in `genes.txt`.

### B. Normalization
- Nicheformer expects normalized expression. 
- **Task:** Perform **Total Count Normalization** (target sum 10,000) followed by **Log1p transformation**. 
- Verify if `adata.X` contains integers (raw) or floats (normalized) before applying.

### C. Label Standardization (`label_map.json`)
- Labels are generated during preprocessing as Leiden cluster IDs (stored in `adata.obs["cluster"]`).
- **Task:** Create a consistent `label_map.json` that maps these cluster IDs (strings) to integers (0, 1, 2...). 
- **Important:** This map must be applied globally to all clients so that "Client A's" Label 0 is the same as "Client B's" Label 0.

### D. File Outputs
- **`processed_table.parquet`**: Should contain `id`, `sample_id` (from `library_key`), `x`, `y`, `label` (int), and the 248 gene columns.
- **`genes.txt`**: A line-separated file of the 248 genes in the exact order used in the parquet columns.
- **`label_map.json`**: The dictionary used for encoding the Leiden cluster pseudo-labels.
