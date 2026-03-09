"""
Create comprehensive visualizations with Spearman and Pearson correlations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style("whitegrid")
OUTPUT_DIR = Path("/home/adamgab/PycharmProjects/LabTools/research/jepa_modality_fusion_hypothesis")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load comprehensive results
results = pd.read_csv(OUTPUT_DIR / "comprehensive_results.csv")

print("Creating comprehensive visualizations...")
print(f"Total results: {len(results)}")
print(f"Unique outcomes: {results['outcome'].nunique()}")

# Colors
colors = {'DEXA': '#1f77b4', 'Retina': '#ff7f0e', 'Pooled': '#2ca02c'}

# ============================================================================
# FIGURE 1: Main Results with R², Spearman, and Pearson (Large Figure)
# ============================================================================
print("\n[1/5] Creating main comprehensive comparison...")

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle('JEPA Modality Fusion: Comprehensive Analysis\n(R², Spearman, Pearson Correlations)',
             fontsize=18, fontweight='bold', y=0.995)

categories = ['shared', 'dexa_specific', 'retina_specific', 'cross_modal']
titles = ['Shared Features', 'DEXA-Specific Features', 'Retina-Specific Features', 'Cross-Modal Features']
metrics = ['r2', 'spearman', 'pearson']
metric_labels = ['R² Score', 'Spearman ρ', 'Pearson r']

for cat_idx, (category, title) in enumerate(zip(categories, titles)):
    cat_data = results[results['category'] == category].copy()

    if len(cat_data) == 0:
        for met_idx in range(3):
            axes[cat_idx, met_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[cat_idx, met_idx].set_title(f"{title}\n{metric_labels[met_idx]}")
        continue

    for met_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[cat_idx, met_idx]

        # Pivot
        pivot = cat_data.pivot_table(index='outcome', columns='embedding_type', values=metric)

        # Sort by best embedding for this metric
        if 'DEXA' in pivot.columns and category == 'dexa_specific':
            pivot = pivot.sort_values('DEXA', ascending=False)
        elif 'Retina' in pivot.columns and category == 'retina_specific':
            pivot = pivot.sort_values('Retina', ascending=False)
        elif 'Pooled' in pivot.columns:
            pivot = pivot.sort_values('Pooled', ascending=False)

        # Limit to top 10 outcomes for readability
        if len(pivot) > 10:
            pivot = pivot.head(10)

        # Plot
        x = np.arange(len(pivot))
        width = 0.25

        for i, emb_type in enumerate(['DEXA', 'Retina', 'Pooled']):
            if emb_type in pivot.columns:
                ax.bar(x + i*width, pivot[emb_type], width,
                       label=emb_type, color=colors[emb_type], alpha=0.8)

        ax.set_xlabel('Outcome', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(f"{title}\n{metric_label}", fontsize=11, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
        if cat_idx == 0 and met_idx == 0:
            ax.legend(fontsize=9, loc='upper right')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

        # Set y-limits appropriately
        if metric == 'r2':
            ax.set_ylim(-0.5, 1.0)
        else:  # Correlations
            ax.set_ylim(-0.5, 1.0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "comprehensive_comparison.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'comprehensive_comparison.png'}")
plt.close()

# ============================================================================
# FIGURE 2: Best Embedding Summary (Bar Chart)
# ============================================================================
print("\n[2/5] Creating best embedding summary...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Which Embedding Performs Best?', fontsize=16, fontweight='bold')

for met_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[met_idx]

    # Count wins per category per embedding
    wins_data = []
    for category in categories:
        cat_results = results[results['category'] == category]
        if len(cat_results) == 0:
            continue

        pivot = cat_results.pivot_table(index='outcome', columns='embedding_type', values=metric)

        for emb_type in ['DEXA', 'Retina', 'Pooled']:
            if emb_type in pivot.columns:
                n_wins = (pivot.idxmax(axis=1) == emb_type).sum()
                wins_data.append({
                    'category': category.replace('_', ' ').title(),
                    'embedding_type': emb_type,
                    'n_wins': n_wins
                })

    wins_df = pd.DataFrame(wins_data)
    wins_pivot = wins_df.pivot_table(index='category', columns='embedding_type', values='n_wins', fill_value=0)

    # Plot
    wins_pivot.plot(kind='bar', ax=ax, color=[colors['DEXA'], colors['Pooled'], colors['Retina']],
                    alpha=0.8, width=0.7, stacked=False)

    ax.set_title(f'{metric_label}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Category', fontsize=11)
    ax.set_ylabel(f'# of Best Predictions', fontsize=11)
    ax.set_xticklabels(wins_pivot.index, rotation=45, ha='right', fontsize=10)
    ax.legend(title='Embedding', fontsize=10, title_fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fontsize=9, padding=3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "wins_summary.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'wins_summary.png'}")
plt.close()

# ============================================================================
# FIGURE 3: Spearman vs Pearson Comparison (Scatter)
# ============================================================================
print("\n[3/5] Creating Spearman vs Pearson scatter...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Spearman vs Pearson Correlations', fontsize=16, fontweight='bold')

for emb_idx, emb_type in enumerate(['DEXA', 'Retina', 'Pooled']):
    ax = axes[emb_idx]

    emb_data = results[results['embedding_type'] == emb_type]

    # Scatter
    for category in categories:
        cat_data = emb_data[emb_data['category'] == category]
        if len(cat_data) > 0:
            ax.scatter(cat_data['pearson'], cat_data['spearman'],
                      label=category.replace('_', ' ').title(), alpha=0.6, s=50)

    # Diagonal line
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Pearson r', fontsize=12)
    ax.set_ylabel('Spearman ρ', fontsize=12)
    ax.set_title(f'{emb_type} Embeddings', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(-0.6, 1.0)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "spearman_vs_pearson.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'spearman_vs_pearson.png'}")
plt.close()

# ============================================================================
# FIGURE 4: Comprehensive Heatmap (Top 30 Outcomes by Spearman)
# ============================================================================
print("\n[4/5] Creating comprehensive heatmap...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

# Select top outcomes by mean Spearman
mean_spearman = results.groupby('outcome')['spearman'].mean().sort_values(ascending=False)
top_outcomes = mean_spearman.head(30).index.tolist()

# Spearman heatmap
pivot_spear = results[results['outcome'].isin(top_outcomes)].pivot_table(
    index='outcome', columns='embedding_type', values='spearman'
)
pivot_spear = pivot_spear.loc[top_outcomes]

sns.heatmap(pivot_spear, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            vmin=-0.3, vmax=0.9, linewidths=0.5, cbar_kws={'label': 'Spearman ρ'},
            ax=ax1, cbar=True, annot_kws={'fontsize': 8})
ax1.set_title('Top 30 Outcomes by Spearman Correlation', fontsize=12, fontweight='bold')
ax1.set_xlabel('Embedding Type', fontsize=11)
ax1.set_ylabel('Outcome', fontsize=11)
ax1.tick_params(axis='y', labelsize=9)

# Pearson heatmap
pivot_pears = results[results['outcome'].isin(top_outcomes)].pivot_table(
    index='outcome', columns='embedding_type', values='pearson'
)
pivot_pears = pivot_pears.loc[top_outcomes]

sns.heatmap(pivot_pears, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            vmin=-0.3, vmax=0.9, linewidths=0.5, cbar_kws={'label': 'Pearson r'},
            ax=ax2, cbar=True, annot_kws={'fontsize': 8})
ax2.set_title('Top 30 Outcomes by Pearson Correlation', fontsize=12, fontweight='bold')
ax2.set_xlabel('Embedding Type', fontsize=11)
ax2.set_ylabel('')
ax2.set_yticklabels([])

plt.tight_layout()
plt.savefig(FIGURES_DIR / "comprehensive_heatmap_top30.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'comprehensive_heatmap_top30.png'}")
plt.close()

# ============================================================================
# FIGURE 5: Suppression Effect with Spearman
# ============================================================================
print("\n[5/5] Creating suppression effect visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Information Suppression in Pooled Embeddings', fontsize=16, fontweight='bold')

# DEXA-specific - R² suppression
ax = axes[0, 0]
dexa_data = results[results['category'] == 'dexa_specific']
dexa_pivot_r2 = dexa_data.pivot_table(index='outcome', columns='embedding_type', values='r2')
if 'DEXA' in dexa_pivot_r2.columns and 'Pooled' in dexa_pivot_r2.columns:
    dexa_supp_r2 = ((dexa_pivot_r2['Pooled'] - dexa_pivot_r2['DEXA']) / dexa_pivot_r2['DEXA'].abs() * 100).dropna().sort_values()
    colors_dexa = ['red' if x < 0 else 'green' for x in dexa_supp_r2]
    ax.barh(range(len(dexa_supp_r2)), dexa_supp_r2, color=colors_dexa, alpha=0.7)
    ax.set_yticks(range(len(dexa_supp_r2)))
    ax.set_yticklabels(dexa_supp_r2.index, fontsize=9)
    ax.set_xlabel('% Change in R² (Pooled vs DEXA)', fontsize=10)
    ax.set_title('DEXA-Specific: R² Suppression', fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.grid(True, alpha=0.3)
    for i, value in enumerate(dexa_supp_r2):
        ax.text(value, i, f'  {value:.1f}%', va='center', fontsize=8)

# DEXA-specific - Spearman suppression
ax = axes[0, 1]
dexa_pivot_spear = dexa_data.pivot_table(index='outcome', columns='embedding_type', values='spearman')
if 'DEXA' in dexa_pivot_spear.columns and 'Pooled' in dexa_pivot_spear.columns:
    dexa_supp_spear = ((dexa_pivot_spear['Pooled'] - dexa_pivot_spear['DEXA']) / dexa_pivot_spear['DEXA'].abs() * 100).dropna().sort_values()
    colors_dexa = ['red' if x < 0 else 'green' for x in dexa_supp_spear]
    ax.barh(range(len(dexa_supp_spear)), dexa_supp_spear, color=colors_dexa, alpha=0.7)
    ax.set_yticks(range(len(dexa_supp_spear)))
    ax.set_yticklabels(dexa_supp_spear.index, fontsize=9)
    ax.set_xlabel('% Change in Spearman (Pooled vs DEXA)', fontsize=10)
    ax.set_title('DEXA-Specific: Spearman Suppression', fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.grid(True, alpha=0.3)
    for i, value in enumerate(dexa_supp_spear):
        ax.text(value, i, f'  {value:.1f}%', va='center', fontsize=8)

# Retina-specific - R² suppression
ax = axes[1, 0]
retina_data = results[results['category'] == 'retina_specific']
retina_pivot_r2 = retina_data.pivot_table(index='outcome', columns='embedding_type', values='r2')
if 'Retina' in retina_pivot_r2.columns and 'Pooled' in retina_pivot_r2.columns:
    retina_supp_r2 = ((retina_pivot_r2['Pooled'] - retina_pivot_r2['Retina']) / retina_pivot_r2['Retina'].abs() * 100).dropna().sort_values()
    colors_retina = ['red' if x < 0 else 'green' for x in retina_supp_r2]
    ax.barh(range(len(retina_supp_r2)), retina_supp_r2, color=colors_retina, alpha=0.7)
    ax.set_yticks(range(len(retina_supp_r2)))
    ax.set_yticklabels(retina_supp_r2.index, fontsize=9)
    ax.set_xlabel('% Change in R² (Pooled vs Retina)', fontsize=10)
    ax.set_title('Retina-Specific: R² Suppression', fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.grid(True, alpha=0.3)
    for i, value in enumerate(retina_supp_r2):
        ax.text(value, i, f'  {value:.1f}%', va='center', fontsize=8)

# Retina-specific - Spearman suppression
ax = axes[1, 1]
retina_pivot_spear = retina_data.pivot_table(index='outcome', columns='embedding_type', values='spearman')
if 'Retina' in retina_pivot_spear.columns and 'Pooled' in retina_pivot_spear.columns:
    retina_supp_spear = ((retina_pivot_spear['Pooled'] - retina_pivot_spear['Retina']) / retina_pivot_spear['Retina'].abs() * 100).dropna().sort_values()
    colors_retina = ['red' if x < 0 else 'green' for x in retina_supp_spear]
    ax.barh(range(len(retina_supp_spear)), retina_supp_spear, color=colors_retina, alpha=0.7)
    ax.set_yticks(range(len(retina_supp_spear)))
    ax.set_yticklabels(retina_supp_spear.index, fontsize=9)
    ax.set_xlabel('% Change in Spearman (Pooled vs Retina)', fontsize=10)
    ax.set_title('Retina-Specific: Spearman Suppression', fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.grid(True, alpha=0.3)
    for i, value in enumerate(retina_supp_spear):
        ax.text(value, i, f'  {value:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "suppression_comprehensive.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'suppression_comprehensive.png'}")
plt.close()

print("\n" + "=" * 80)
print("COMPREHENSIVE VISUALIZATIONS COMPLETE")
print("=" * 80)
print(f"\nAll figures saved to: {FIGURES_DIR}/")
print("\nGenerated:")
print("  1. comprehensive_comparison.png - Full matrix (4 categories × 3 metrics)")
print("  2. wins_summary.png - Best embedding counts per category")
print("  3. spearman_vs_pearson.png - Correlation comparison")
print("  4. comprehensive_heatmap_top30.png - Top 30 outcomes")
print("  5. suppression_comprehensive.png - Suppression with R² and Spearman")
