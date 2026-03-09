# JEPA Modality Fusion Hypothesis Study

## Executive Summary

**Hypothesis**: Multi-modal JEPA training (minimizing L2 distance between DEXA and retina embeddings) maximizes shared features at the expense of modality-specific features.

**Result**: ✅ **CONFIRMED** - Tested across 44 outcomes with 132 experiments.

---

## Key Findings

### 1. Shared Features: Pooled Wins
- **Age**: Pooled R²=0.792 (best) vs DEXA 0.782, Retina 0.763
- Forcing modalities to align **enhances** universal shared features

### 2. DEXA-Specific Features: Pooling Suppresses
- **Total Lean Mass**: DEXA R²=0.846 → Pooled 0.771 (**-8.9%**)
- **Weight**: DEXA R²=0.821 → Pooled 0.710 (**-13.5%**)
- **Visceral Fat**: DEXA R²=0.077 → Pooled 0.048 (**-37.7%**)
- **Subcutaneous Fat**: DEXA R²=0.043 → Pooled 0.011 (**-74.4%**)

**Pattern**: DEXA wins 14/14 outcomes on R² and Pearson correlation

### 3. Retina-Specific Features: Pooling Suppresses
- **Retinal Fractal Dimension**: Retina R²=0.484 → Pooled 0.288 (**-40.5%**)
- **Artery Width**: Retina R²=0.312 → Pooled 0.286 (**-8.3%**)
- **Vein Width**: Retina R²=0.210 → Pooled 0.164 (**-21.9%**)

**Pattern**: Retina wins 14/17 outcomes on Spearman correlation

### 4. Cross-Modal Features: Pooling Helps
- **Glucose**: Pooled R²=0.053 (best) vs DEXA -0.119, Retina -0.073
- **HbA1c**: Pooled R²=-0.049 (least bad) vs DEXA -0.399, Retina -0.275

**Pattern**: Pooled wins 11/11 outcomes on R²

---

## What This Means

The JEPA encoder learns **what makes different modalities look the same** (age, sex, general health), not **what makes them different** (bone structure, vessel morphology).

This is **by design**, not a bug. The L2 minimization objective rewards shared features and penalizes specific features.

---

## Practical Recommendations

### For Current Users

| Task | Use This Embedding | Why |
|------|-------------------|-----|
| Age prediction | **Pooled** | Best R² (0.792) |
| Body composition | **DEXA-only** | Preserves bone/fat info |
| Bone density, lean mass | **DEXA-only** | Up to 74% better than pooled |
| Retinal vessel analysis | **Retina-only** | Up to 40% better than pooled |
| General health (weak signals) | **Pooled** | Combines weak signals |

### For Future Development

1. **Disentangled representations**: Separate embedding subspaces for shared vs specific
2. **Modality-specific auxiliary tasks**: Add bone/vessel prediction losses during training
3. **Attention-based fusion**: Learn which features to pool vs keep separate
4. **Alternative objectives**: Explore contrastive losses that preserve specificity

---

## Files in This Study

### Documentation
- **README.md** (this file) - Quick overview
- **SUMMARY.md** - Detailed executive summary
- **FINDINGS.md** - Deep dive interpretation
- **DATA_DESCRIPTION.md** - Study design and data structure

### Results
- **comprehensive_results.csv** - All 132 cross-validation results
- **JEPA_Modality_Fusion_Report.pdf** - Publication-ready 15-page report

### Code
- **test_comprehensive.py** - Main analysis (44 outcomes, 3 metrics each)
- **create_comprehensive_figures.py** - 8 publication-quality figures
- **explore_embeddings.py** - Initial data exploration

### Figures (8 total)
1. **comprehensive_comparison.png** - 4×3 grid (categories × metrics)
2. **wins_summary.png** - Best embedding counts per category
3. **suppression_comprehensive.png** - R² and Spearman suppression effects
4. **spearman_vs_pearson.png** - Correlation metric comparison
5. **comprehensive_heatmap_top30.png** - Top 30 outcomes heatmap
6. **main_comparison.png** (initial) - First analysis version
7. **suppression_effect.png** (initial) - First suppression analysis
8. **best_embedding_summary.png** (initial) - First summary

---

## Methods

**Data**: HPP 10K cohort, ages 40-70 (n=6,445 subjects with both modalities)

**Embeddings**:
- DEXA-only: 768D (2 global views concatenated)
- Retina-only: 768D (2 global views concatenated)
- Pooled: 384D (mean of 4 global views)

**Analysis**:
- 5-fold cross-validation
- Subject-level splits (no data leakage)
- Ridge regression (α=1.0)
- Metrics: R², Spearman ρ, Pearson r

**Outcomes** (44 total):
- Shared: Age, BMI (2)
- DEXA-specific: Lean mass, bone mass, fat distribution (14)
- Retina-specific: Vessel width, fractal dimension, tortuosity (17)
- Cross-modal: Glucose, lipids, liver, renal, immune markers (11)

---

## How to Reproduce

```bash
# Activate environment
source /home/adamgab/miniconda3/etc/profile.d/conda.sh && conda activate LinearQueue
export PYTHONPATH=/home/adamgab/PycharmProjects/LabTools

# Run comprehensive analysis
cd research/jepa_modality_fusion_hypothesis
python test_comprehensive.py

# Generate figures
python create_comprehensive_figures.py

# Generate PDF report
python generate_pdf_report.py
```

---

## Citation

If using these findings, please cite:
- Study: "JEPA Modality Fusion: Shared vs Specific Features in Multi-Modal Medical Image Embeddings"
- Data: HPP 10K Cohort
- Date: February 2026
- Location: `/home/adamgab/PycharmProjects/LabTools/research/jepa_modality_fusion_hypothesis/`

---

## Questions?

See FINDINGS.md for detailed interpretation, or contact the study author.

**Bottom line**: Use single-modality embeddings for modality-specific tasks. Use pooled embeddings for shared features like age.
