# JEPA Modality Fusion Hypothesis Study

## Background

### Model Architecture
- **JEPA (Joint Embedding Predictive Architecture)** encoder trained on multi-modal medical imaging
- **Input Modalities**: 4 images per subject
  - DEXA tissue scan
  - DEXA bone scan
  - Retina OS (left eye) scan
  - Retina OD (right eye) scan

### Training Strategy
- **10 views** total: 4 global views + local crops (DINO-style augmentation)
- **Objective**: Minimize L2 distance between all local view embeddings and global views
- **Goal**: Force all modalities from same person to be close in latent space

### Observed Phenomenon
- ✅ **HIGH** prediction accuracy: Age, BMI (shared features across modalities)
- ❌ **LOW** prediction accuracy: Modality-specific features
  - DEXA-specific: Bone density, body composition details, visceral fat
  - Retina-specific: Vascular health, diabetic markers, ocular pathology

## Research Hypothesis

**To minimize feature distance between retina and DEXA, the encoder maximizes shared features (age, BMI) and suppresses modality-specific features.**

This is a fundamental representation learning question: Does multi-modal contrastive learning sacrifice modality-specific information for cross-modal alignment?

## Available Data

Location: `/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepMultiJepa/embeddings/y0wh39vh/`

### 1. DEXA-Only Embeddings (`dexa_embeddings.csv`, 163MB)
- **Content**: 2 global views (DEXA tissue + DEXA bone) concatenated
- **No augmentations**: Clean global representations only
- **Expected**: Should preserve DEXA-specific features (bone, composition)

### 2. Retina-Only Embeddings (`retina_embeddings.csv`, 143MB)
- **Content**: 2 global views (Retina OS + Retina OD) concatenated
- **No augmentations**: Clean global representations only
- **Expected**: Should preserve Retina-specific features (vascular, ocular)

### 3. Mean-Pooled Embeddings (`pooled_embeddings.csv`, 97MB)
- **Content**: Mean of 4 global views (both DEXA + both Retina)
- **Subset**: Only subjects with BOTH modalities
- **Known Issue**: Not always same research_stage across modalities
- **Expected**: If hypothesis correct, should maximize shared features, suppress specific

## Research Questions

1. **Do single-modality embeddings outperform pooled on modality-specific outcomes?**
   - DEXA → bone density, visceral fat, lean mass
   - Retina → vascular markers, diabetic indicators

2. **Do pooled embeddings outperform single-modality on shared outcomes?**
   - Age, BMI, general health status

3. **What features are maximally shared vs modality-specific?**
   - Identify which biomarkers drive the fusion behavior

4. **Does the research_stage mismatch in pooled data affect results?**

## Analysis Plan

### Phase 1: Data Exploration ✓ COMPLETE
- [x] Load all 3 embedding files
- [x] Understand structure (columns, index, embedding dimensions)
- [x] Check for research_stage mismatches in pooled
- [x] Verify sample sizes and overlap

**Key Findings**:
- DEXA: 768D, 8,652 subjects, 11,015 visits
- Retina: 768D, 9,677 subjects, 9,680 visits
- Pooled: 384D, 6,445 subjects, 12,937 visits (duplicates due to research_stage mismatch)

### Phase 2: Hypothesis Testing ✓ COMPLETE
- [x] Select candidate outcomes:
  - **Shared**: Age, BMI
  - **DEXA-specific**: Lean mass, weight, visceral fat, subcutaneous fat
  - **Retina-specific**: Retinal vessel morphology (artery/vein width, fractal dimension, tortuosity)
  - **Cross-modal**: Glucose, HbA1c, WBC

- [x] Cross-validate predictions:
  - DEXA embeddings → all outcomes
  - Retina embeddings → all outcomes
  - Pooled embeddings → all outcomes

- [x] Compare performance (R² for regression)
  - Subject-level 5-fold CV to prevent data leakage

**Results**: See `hypothesis_test_results.csv`

### Phase 3: Interpretation ✓ COMPLETE
- [x] Quantify shared vs specific signal
- [x] Identify which biomarkers are most affected by fusion
- [x] Created visualizations (4 figures)
- [x] Documented findings in FINDINGS.md and SUMMARY.md

**Main Finding**: Pooling suppresses modality-specific features by 8-74%

## Expected Findings (if hypothesis true)

| Outcome Type | DEXA-only | Retina-only | Pooled | Interpretation |
|--------------|-----------|-------------|--------|----------------|
| Age | Good | Good | **Best** | Shared feature maximized |
| BMI | Good | Medium | **Best** | Shared feature maximized |
| Bone density | **Best** | Poor | Medium | DEXA-specific suppressed in pooling |
| Visceral fat | **Best** | Poor | Medium | DEXA-specific suppressed in pooling |
| Vascular markers | Poor | **Best** | Medium | Retina-specific suppressed in pooling |

## Next Steps

1. Load and explore embeddings
2. Select meaningful outcomes across 10K
3. Run systematic cross-validation
4. Generate report with findings
