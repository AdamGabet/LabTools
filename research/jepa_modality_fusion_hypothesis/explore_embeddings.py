"""
Explore JEPA embeddings structure and prepare for hypothesis testing
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Data paths
EMBED_DIR = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepMultiJepa/embeddings/y0wh39vh")

print("=" * 80)
print("JEPA EMBEDDINGS EXPLORATION")
print("=" * 80)

# Load embeddings
print("\n1. Loading embeddings...")
dexa_emb = pd.read_csv(EMBED_DIR / "dexa_embeddings.csv")
retina_emb = pd.read_csv(EMBED_DIR / "retina_embeddings.csv")
pooled_emb = pd.read_csv(EMBED_DIR / "pooled_embeddings.csv")

print(f"   DEXA embeddings: {dexa_emb.shape}")
print(f"   Retina embeddings: {retina_emb.shape}")
print(f"   Pooled embeddings: {pooled_emb.shape}")

# Check structure
print("\n2. DEXA Embeddings Structure:")
print(f"   Columns: {list(dexa_emb.columns[:5])}... (showing first 5)")
print(f"   Index column candidates: {[c for c in dexa_emb.columns if 'id' in c.lower() or 'registration' in c.lower() or 'stage' in c.lower()]}")
print(f"   First few rows:")
print(dexa_emb.head(2))

print("\n3. Retina Embeddings Structure:")
print(f"   Columns: {list(retina_emb.columns[:5])}... (showing first 5)")
print(f"   Index column candidates: {[c for c in retina_emb.columns if 'id' in c.lower() or 'registration' in c.lower() or 'stage' in c.lower()]}")
print(f"   First few rows:")
print(retina_emb.head(2))

print("\n4. Pooled Embeddings Structure:")
print(f"   Columns: {list(pooled_emb.columns[:5])}... (showing first 5)")
print(f"   Index column candidates: {[c for c in pooled_emb.columns if 'id' in c.lower() or 'registration' in c.lower() or 'stage' in c.lower()]}")
print(f"   First few rows:")
print(pooled_emb.head(2))

# Identify embedding columns vs metadata
print("\n5. Identifying Metadata vs Embedding Dimensions:")
for name, df in [("DEXA", dexa_emb), ("Retina", retina_emb), ("Pooled", pooled_emb)]:
    # Likely metadata columns
    metadata_cols = [c for c in df.columns if 'id' in c.lower() or 'stage' in c.lower() or 'code' in c.lower()]
    # Likely embedding columns (numeric, many of them)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    embedding_cols = [c for c in numeric_cols if c not in metadata_cols]

    print(f"\n   {name}:")
    print(f"   - Metadata columns ({len(metadata_cols)}): {metadata_cols}")
    print(f"   - Embedding dimensions: {len(embedding_cols)}")

# Check for research_stage issues in pooled
print("\n6. Research Stage Analysis:")
if 'research_stage' in pooled_emb.columns or any('stage' in c.lower() for c in pooled_emb.columns):
    stage_cols = [c for c in pooled_emb.columns if 'stage' in c.lower()]
    print(f"   Found stage columns: {stage_cols}")
    for col in stage_cols:
        print(f"   {col} distribution:")
        print(pooled_emb[col].value_counts())
else:
    print("   No explicit research_stage column found in pooled embeddings")

# Check sample overlap
print("\n7. Sample Overlap Analysis:")
# Try to identify subject ID columns
def find_id_col(df):
    candidates = [c for c in df.columns if 'registration' in c.lower() or '10k' in c.lower() or (c.lower() == 'id')]
    if candidates:
        return candidates[0]
    # If not found, assume first column
    return df.columns[0]

dexa_id_col = find_id_col(dexa_emb)
retina_id_col = find_id_col(retina_emb)
pooled_id_col = find_id_col(pooled_emb)

print(f"   Using ID columns: DEXA={dexa_id_col}, Retina={retina_id_col}, Pooled={pooled_id_col}")

dexa_ids = set(dexa_emb[dexa_id_col])
retina_ids = set(retina_emb[retina_id_col])
pooled_ids = set(pooled_emb[pooled_id_col])

print(f"\n   Sample sizes:")
print(f"   - DEXA only: {len(dexa_ids)}")
print(f"   - Retina only: {len(retina_ids)}")
print(f"   - Pooled: {len(pooled_ids)}")
print(f"\n   Overlap:")
print(f"   - DEXA ∩ Retina: {len(dexa_ids & retina_ids)}")
print(f"   - Pooled ⊆ (DEXA ∩ Retina)? {pooled_ids.issubset(dexa_ids & retina_ids)}")
print(f"   - Pooled = (DEXA ∩ Retina)? {pooled_ids == (dexa_ids & retina_ids)}")

# Check for duplicates
print("\n8. Checking for Duplicates:")
for name, df, id_col in [("DEXA", dexa_emb, dexa_id_col),
                          ("Retina", retina_emb, retina_id_col),
                          ("Pooled", pooled_emb, pooled_id_col)]:
    n_duplicates = df[id_col].duplicated().sum()
    print(f"   {name}: {n_duplicates} duplicates")
    if n_duplicates > 0:
        print(f"   Example duplicates: {df[df[id_col].duplicated(keep=False)][id_col].head()}")

# Data quality checks
print("\n9. Data Quality:")
for name, df in [("DEXA", dexa_emb), ("Retina", retina_emb), ("Pooled", pooled_emb)]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_missing = df[numeric_cols].isna().sum().sum()
    n_inf = np.isinf(df[numeric_cols]).sum().sum()

    print(f"\n   {name}:")
    print(f"   - Missing values: {n_missing}")
    print(f"   - Inf values: {n_inf}")
    print(f"   - Embedding range: [{df[numeric_cols].min().min():.3f}, {df[numeric_cols].max().max():.3f}]")
    print(f"   - Embedding mean: {df[numeric_cols].mean().mean():.3f}")
    print(f"   - Embedding std: {df[numeric_cols].std().mean():.3f}")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
