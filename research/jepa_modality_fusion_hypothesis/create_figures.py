"""
Create visualizations for JEPA modality fusion hypothesis findings
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

# Load results
results = pd.read_csv(OUTPUT_DIR / "hypothesis_test_results.csv")

print("Creating visualizations...")

# ============================================================================
# FIGURE 1: R² Comparison Across Categories (Main Result)
# ============================================================================
print("\n[1/4] Creating main comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('JEPA Modality Fusion: Shared vs Specific Features', fontsize=16, fontweight='bold')

categories = ['shared', 'dexa_specific', 'retina_specific', 'cross_modal']
titles = ['Shared Features\n(Should favor Pooled)',
          'DEXA-Specific Features\n(Should favor DEXA)',
          'Retina-Specific Features\n(Should favor Retina)',
          'Cross-Modal Features\n(Pooled might help)']

for idx, (category, title) in enumerate(zip(categories, titles)):
    ax = axes[idx // 2, idx % 2]

    cat_data = results[results['category'] == category].copy()

    if len(cat_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(title)
        continue

    # Pivot for plotting
    pivot = cat_data.pivot_table(
        index='outcome',
        columns='embedding_type',
        values='r2'
    )

    # Sort by DEXA performance for DEXA-specific, Retina for retina-specific, etc.
    if category == 'dexa_specific' and 'DEXA' in pivot.columns:
        pivot = pivot.sort_values('DEXA', ascending=False)
    elif category == 'retina_specific' and 'Retina' in pivot.columns:
        pivot = pivot.sort_values('Retina', ascending=False)
    elif 'Pooled' in pivot.columns:
        pivot = pivot.sort_values('Pooled', ascending=False)

    # Plot
    x = np.arange(len(pivot))
    width = 0.25

    colors = {'DEXA': '#1f77b4', 'Retina': '#ff7f0e', 'Pooled': '#2ca02c'}

    for i, emb_type in enumerate(['DEXA', 'Retina', 'Pooled']):
        if emb_type in pivot.columns:
            ax.bar(x + i*width, pivot[emb_type], width,
                   label=emb_type, color=colors[emb_type], alpha=0.8)

    ax.set_xlabel('Outcome', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "main_comparison.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'main_comparison.png'}")
plt.close()

# ============================================================================
# FIGURE 2: Suppression Effect (Pooled vs Single-Modality)
# ============================================================================
print("\n[2/4] Creating suppression effect plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Information Suppression in Pooled Embeddings', fontsize=16, fontweight='bold')

# DEXA-specific suppression
dexa_data = results[results['category'] == 'dexa_specific'].copy()
dexa_pivot = dexa_data.pivot_table(index='outcome', columns='embedding_type', values='r2')

if 'DEXA' in dexa_pivot.columns and 'Pooled' in dexa_pivot.columns:
    dexa_suppression = ((dexa_pivot['Pooled'] - dexa_pivot['DEXA']) / dexa_pivot['DEXA'] * 100).sort_values()

    colors_dexa = ['red' if x < 0 else 'green' for x in dexa_suppression]
    ax1.barh(range(len(dexa_suppression)), dexa_suppression, color=colors_dexa, alpha=0.7)
    ax1.set_yticks(range(len(dexa_suppression)))
    ax1.set_yticklabels(dexa_suppression.index, fontsize=10)
    ax1.set_xlabel('% Change in R² (Pooled vs DEXA-only)', fontsize=11)
    ax1.set_title('DEXA-Specific Features:\nPooling Suppresses Performance', fontsize=12, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=1, linestyle='--')
    ax1.grid(True, alpha=0.3)

    # Add text annotations
    for i, (outcome, value) in enumerate(dexa_suppression.items()):
        ax1.text(value, i, f'  {value:.1f}%', va='center', fontsize=9)

# Retina-specific suppression
retina_data = results[results['category'] == 'retina_specific'].copy()
retina_pivot = retina_data.pivot_table(index='outcome', columns='embedding_type', values='r2')

if 'Retina' in retina_pivot.columns and 'Pooled' in retina_pivot.columns:
    retina_suppression = ((retina_pivot['Pooled'] - retina_pivot['Retina']) / retina_pivot['Retina'].abs() * 100).sort_values()
    retina_suppression = retina_suppression[retina_suppression.notna()]

    colors_retina = ['red' if x < 0 else 'green' for x in retina_suppression]
    ax2.barh(range(len(retina_suppression)), retina_suppression, color=colors_retina, alpha=0.7)
    ax2.set_yticks(range(len(retina_suppression)))
    ax2.set_yticklabels(retina_suppression.index, fontsize=10)
    ax2.set_xlabel('% Change in R² (Pooled vs Retina-only)', fontsize=11)
    ax2.set_title('Retina-Specific Features:\nPooling Suppresses Performance', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=1, linestyle='--')
    ax2.grid(True, alpha=0.3)

    # Add text annotations
    for i, (outcome, value) in enumerate(retina_suppression.items()):
        ax2.text(value, i, f'  {value:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "suppression_effect.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'suppression_effect.png'}")
plt.close()

# ============================================================================
# FIGURE 3: Heatmap of All Results
# ============================================================================
print("\n[3/4] Creating comprehensive heatmap...")

fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# Prepare data
pivot_all = results.pivot_table(
    index='outcome',
    columns='embedding_type',
    values='r2'
)

# Sort by category
results_sorted = results.sort_values(['category', 'outcome']).drop_duplicates('outcome')
pivot_all = pivot_all.loc[results_sorted['outcome'].unique()]

# Create heatmap
sns.heatmap(pivot_all, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            vmin=-0.3, vmax=0.9, linewidths=0.5, cbar_kws={'label': 'R² Score'},
            ax=ax)

ax.set_title('JEPA Embeddings: R² Performance Across All Outcomes',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Embedding Type', fontsize=12)
ax.set_ylabel('Outcome', fontsize=12)

# Add category separators
category_boundaries = []
current_cat = None
for i, outcome in enumerate(pivot_all.index):
    cat = results[results['outcome'] == outcome]['category'].iloc[0]
    if cat != current_cat:
        category_boundaries.append(i)
        current_cat = cat

for boundary in category_boundaries[1:]:
    ax.axhline(boundary, color='black', linewidth=2)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "comprehensive_heatmap.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'comprehensive_heatmap.png'}")
plt.close()

# ============================================================================
# FIGURE 4: Best Embedding by Category (Summary)
# ============================================================================
print("\n[4/4] Creating summary visualization...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Count best embeddings per category
summary_data = []
for category in categories:
    cat_results = results[results['category'] == category]
    if len(cat_results) == 0:
        continue

    pivot = cat_results.pivot_table(index='outcome', columns='embedding_type', values='r2')

    for outcome in pivot.index:
        best_emb = pivot.loc[outcome].idxmax()
        best_r2 = pivot.loc[outcome].max()

        summary_data.append({
            'category': category,
            'outcome': outcome,
            'best_embedding': best_emb,
            'best_r2': best_r2
        })

summary_df = pd.DataFrame(summary_data)

# Create grouped bar chart
category_names = {
    'shared': 'Shared\nFeatures',
    'dexa_specific': 'DEXA-\nSpecific',
    'retina_specific': 'Retina-\nSpecific',
    'cross_modal': 'Cross-\nModal'
}

# Count wins per embedding type per category
wins_data = []
for category in categories:
    cat_summary = summary_df[summary_df['category'] == category]
    if len(cat_summary) == 0:
        continue

    for emb_type in ['DEXA', 'Retina', 'Pooled']:
        n_wins = (cat_summary['best_embedding'] == emb_type).sum()
        wins_data.append({
            'category': category_names.get(category, category),
            'embedding_type': emb_type,
            'n_wins': n_wins
        })

wins_df = pd.DataFrame(wins_data)
wins_pivot = wins_df.pivot_table(index='category', columns='embedding_type', values='n_wins', fill_value=0)

# Reorder categories
category_order = ['Shared\nFeatures', 'DEXA-\nSpecific', 'Retina-\nSpecific', 'Cross-\nModal']
wins_pivot = wins_pivot.reindex([c for c in category_order if c in wins_pivot.index])

# Plot
wins_pivot.plot(kind='bar', ax=ax, color=[colors['DEXA'], colors['Retina'], colors['Pooled']],
                alpha=0.8, width=0.7)

ax.set_title('Which Embedding Performs Best? (# of Wins per Category)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Feature Category', fontsize=12)
ax.set_ylabel('Number of Best Predictions', fontsize=12)
ax.set_xticklabels(wins_pivot.index, rotation=0, fontsize=11)
ax.legend(title='Embedding Type', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fontsize=10, padding=3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "best_embedding_summary.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / 'best_embedding_summary.png'}")
plt.close()

print("\n" + "=" * 80)
print("VISUALIZATIONS COMPLETE")
print("=" * 80)
print(f"\nAll figures saved to: {FIGURES_DIR}")
print("\nGenerated:")
print("  1. main_comparison.png - R² across all categories")
print("  2. suppression_effect.png - Pooling suppression of modality-specific features")
print("  3. comprehensive_heatmap.png - Complete results heatmap")
print("  4. best_embedding_summary.png - Summary of best embeddings per category")
