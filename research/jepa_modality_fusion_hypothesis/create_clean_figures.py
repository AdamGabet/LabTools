"""
Create clean, readable figures with large fonts (size 14+)
ONE CLEAR MESSAGE PER FIGURE
USING PEARSON CORRELATION AS THE SINGLE METRIC
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup - LARGE FONTS
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20

sns.set_style("whitegrid")
OUTPUT_DIR = Path("/home/adamgab/PycharmProjects/LabTools/research/jepa_modality_fusion_hypothesis")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load comprehensive results
results = pd.read_csv(OUTPUT_DIR / "comprehensive_results.csv")

print("Creating clean, readable figures...")

# Colors
colors = {'DEXA': '#1f77b4', 'Retina': '#ff7f0e', 'Pooled': '#2ca02c'}

# ============================================================================
# FIGURE 1: Shared Features - Pooled Wins
# ============================================================================
print("\n[1/6] Shared features...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

shared_data = results[results['category'] == 'shared']
pivot = shared_data.pivot_table(index='outcome', columns='embedding_type', values='pearson')

x = np.arange(len(pivot))
width = 0.25

for i, emb_type in enumerate(['DEXA', 'Retina', 'Pooled']):
    if emb_type in pivot.columns:
        bars = ax.bar(x + i*width, pivot[emb_type], width,
               label=emb_type, color=colors[emb_type], alpha=0.8)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Pearson Correlation', fontsize=18, fontweight='bold')
ax.set_title('Shared Features: Pooled Embeddings Win\n(Age prediction benefits from multi-modal fusion)',
             fontsize=20, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot.index, fontsize=16, fontweight='bold')
ax.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "clean_1_shared.png", dpi=300, bbox_inches='tight')
print(f"   Saved: clean_1_shared.png")
plt.close()

# ============================================================================
# FIGURE 2: DEXA-Specific Features - DEXA Wins
# ============================================================================
print("\n[2/6] DEXA-specific features...")

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

dexa_data = results[results['category'] == 'dexa_specific']
pivot = dexa_data.pivot_table(index='outcome', columns='embedding_type', values='pearson')
pivot = pivot.sort_values('DEXA', ascending=False).head(10)  # Top 10

x = np.arange(len(pivot))
width = 0.25

for i, emb_type in enumerate(['DEXA', 'Retina', 'Pooled']):
    if emb_type in pivot.columns:
        bars = ax.bar(x + i*width, pivot[emb_type], width,
               label=emb_type, color=colors[emb_type], alpha=0.8)

ax.set_ylabel('Pearson Correlation', fontsize=18, fontweight='bold')
ax.set_title('DEXA-Specific Features: DEXA-Only Embeddings Win\n(Body composition info suppressed in pooled)',
             fontsize=20, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=14)
ax.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.axhline(0, color='black', linewidth=1.5)
ax.set_ylim(-0.2, 1.0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "clean_2_dexa_specific.png", dpi=300, bbox_inches='tight')
print(f"   Saved: clean_2_dexa_specific.png")
plt.close()

# ============================================================================
# FIGURE 3: Retina-Specific Features - Retina Wins
# ============================================================================
print("\n[3/6] Retina-specific features...")

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

retina_data = results[results['category'] == 'retina_specific']
pivot = retina_data.pivot_table(index='outcome', columns='embedding_type', values='pearson')
pivot = pivot.sort_values('Retina', ascending=False).head(10)  # Top 10

x = np.arange(len(pivot))
width = 0.25

for i, emb_type in enumerate(['DEXA', 'Retina', 'Pooled']):
    if emb_type in pivot.columns:
        bars = ax.bar(x + i*width, pivot[emb_type], width,
               label=emb_type, color=colors[emb_type], alpha=0.8)

ax.set_ylabel('Pearson Correlation', fontsize=18, fontweight='bold')
ax.set_title('Retina-Specific Features: Retina-Only Embeddings Win\n(Vessel morphology suppressed in pooled)',
             fontsize=20, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=14)
ax.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.axhline(0, color='black', linewidth=1.5)
ax.set_ylim(-0.2, 0.8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "clean_3_retina_specific.png", dpi=300, bbox_inches='tight')
print(f"   Saved: clean_3_retina_specific.png")
plt.close()

# ============================================================================
# FIGURE 4: Cross-Modal Features - Pooled Wins
# ============================================================================
print("\n[4/6] Cross-modal features...")

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

cross_data = results[results['category'] == 'cross_modal']
pivot = cross_data.pivot_table(index='outcome', columns='embedding_type', values='pearson')
pivot = pivot.sort_values('Pooled', ascending=False).head(10)  # Top 10

x = np.arange(len(pivot))
width = 0.25

for i, emb_type in enumerate(['DEXA', 'Retina', 'Pooled']):
    if emb_type in pivot.columns:
        bars = ax.bar(x + i*width, pivot[emb_type], width,
               label=emb_type, color=colors[emb_type], alpha=0.8)

ax.set_ylabel('Pearson Correlation', fontsize=18, fontweight='bold')
ax.set_title('Cross-Modal Features: Pooled Embeddings Win\n(Both modalities contribute weak signals)',
             fontsize=20, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=14)
ax.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.axhline(0, color='black', linewidth=1.5)
ax.set_ylim(-0.4, 0.4)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "clean_4_cross_modal.png", dpi=300, bbox_inches='tight')
print(f"   Saved: clean_4_cross_modal.png")
plt.close()

# ============================================================================
# FIGURE 5: DEXA Suppression - How Much Info Is Lost?
# ============================================================================
print("\n[5/6] DEXA suppression effect...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

dexa_data = results[results['category'] == 'dexa_specific']
dexa_pivot = dexa_data.pivot_table(index='outcome', columns='embedding_type', values='pearson')

if 'DEXA' in dexa_pivot.columns and 'Pooled' in dexa_pivot.columns:
    suppression = ((dexa_pivot['Pooled'] - dexa_pivot['DEXA']) / dexa_pivot['DEXA'].abs() * 100).dropna().sort_values()

    colors_bars = ['#d62728' if x < 0 else '#2ca02c' for x in suppression]
    bars = ax.barh(range(len(suppression)), suppression, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_yticks(range(len(suppression)))
    ax.set_yticklabels(suppression.index, fontsize=14)
    ax.set_xlabel('% Change in Pearson (Pooled vs DEXA-only)', fontsize=18, fontweight='bold')
    ax.set_title('DEXA-Specific Features: Information Loss from Pooling\n(Negative = Suppressed)',
                 fontsize=20, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linewidth=2, linestyle='--')
    ax.grid(True, alpha=0.3, axis='x', linewidth=1.5)

    # Add value labels
    for i, (outcome, value) in enumerate(suppression.items()):
        label = f'{value:.1f}%'
        if value < 0:
            ax.text(value - 5, i, label, va='center', ha='right', fontsize=14, fontweight='bold', color='white')
        else:
            ax.text(value + 5, i, label, va='center', ha='left', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "clean_5_dexa_suppression.png", dpi=300, bbox_inches='tight')
print(f"   Saved: clean_5_dexa_suppression.png")
plt.close()

# ============================================================================
# FIGURE 6: Retina Suppression - How Much Info Is Lost?
# ============================================================================
print("\n[6/6] Retina suppression effect...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

retina_data = results[results['category'] == 'retina_specific']
retina_pivot = retina_data.pivot_table(index='outcome', columns='embedding_type', values='pearson')

if 'Retina' in retina_pivot.columns and 'Pooled' in retina_pivot.columns:
    suppression = ((retina_pivot['Pooled'] - retina_pivot['Retina']) / retina_pivot['Retina'].abs() * 100).dropna().sort_values()

    colors_bars = ['#d62728' if x < 0 else '#2ca02c' for x in suppression]
    bars = ax.barh(range(len(suppression)), suppression, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_yticks(range(len(suppression)))
    ax.set_yticklabels(suppression.index, fontsize=14)
    ax.set_xlabel('% Change in Pearson (Pooled vs Retina-only)', fontsize=18, fontweight='bold')
    ax.set_title('Retina-Specific Features: Information Loss from Pooling\n(Negative = Suppressed)',
                 fontsize=20, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linewidth=2, linestyle='--')
    ax.grid(True, alpha=0.3, axis='x', linewidth=1.5)

    # Add value labels
    for i, (outcome, value) in enumerate(suppression.items()):
        label = f'{value:.1f}%'
        if value < 0:
            ax.text(value - 5, i, label, va='center', ha='right', fontsize=14, fontweight='bold', color='white')
        else:
            ax.text(value + 5, i, label, va='center', ha='left', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "clean_6_retina_suppression.png", dpi=300, bbox_inches='tight')
print(f"   Saved: clean_6_retina_suppression.png")
plt.close()

print("\n" + "=" * 80)
print("CLEAN FIGURES COMPLETE")
print("=" * 80)
print("\nGenerated 6 clear, readable figures:")
print("  1. clean_1_shared.png - Shared features (Pooled wins)")
print("  2. clean_2_dexa_specific.png - DEXA features (DEXA wins)")
print("  3. clean_3_retina_specific.png - Retina features (Retina wins)")
print("  4. clean_4_cross_modal.png - Cross-modal features (Pooled wins)")
print("  5. clean_5_dexa_suppression.png - DEXA suppression from pooling")
print("  6. clean_6_retina_suppression.png - Retina suppression from pooling")
