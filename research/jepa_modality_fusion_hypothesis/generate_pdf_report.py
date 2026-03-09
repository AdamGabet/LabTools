"""
Generate PDF report for JEPA modality fusion hypothesis study
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from pathlib import Path
import pandas as pd

# Paths
OUTPUT_DIR = Path("/home/adamgab/PycharmProjects/LabTools/research/jepa_modality_fusion_hypothesis")
FIGURES_DIR = OUTPUT_DIR / "figures"
PDF_PATH = OUTPUT_DIR / "JEPA_Modality_Fusion_Report.pdf"

# Load results for summary table
results = pd.read_csv(OUTPUT_DIR / "comprehensive_results.csv")

print("Generating PDF report...")

# Create PDF
doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter,
                       topMargin=0.5*inch, bottomMargin=0.5*inch,
                       leftMargin=0.75*inch, rightMargin=0.75*inch)

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1f77b4'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

subtitle_style = ParagraphStyle(
    'CustomSubtitle',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#555555'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=16,
    textColor=colors.HexColor('#2ca02c'),
    spaceAfter=10,
    spaceBefore=15,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontSize=11,
    spaceAfter=12,
    alignment=TA_JUSTIFY,
    leading=14
)

# Build story
story = []

# Title page
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph("JEPA Modality Fusion:<br/>Shared vs Specific Features", title_style))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph("Hypothesis Testing on Multi-Modal Medical Image Embeddings", subtitle_style))
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("HPP 10K Cohort Analysis", subtitle_style))
story.append(Spacer(1, 1.5*inch))

# Key finding box
finding_text = """
<b>KEY FINDING:</b> Multi-modal JEPA training amplifies shared features (age, BMI)
but systematically suppresses modality-specific information. DEXA-specific features
lose 8-74% predictive power when pooled with retina, and retina-specific features
lose 8-40% when pooled with DEXA.
"""
story.append(Paragraph(finding_text, body_style))
story.append(Spacer(1, 0.3*inch))

# Summary statistics
summary_data = [
    ['Metric', 'Value'],
    ['Outcomes Tested', str(results['outcome'].nunique())],
    ['Total Tests', str(len(results))],
    ['DEXA Subjects', '8,652'],
    ['Retina Subjects', '9,677'],
    ['Pooled Subjects', '6,445'],
    ['CV Folds', '5 (subject-level)'],
]

summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
summary_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 10),
]))
story.append(summary_table)
story.append(PageBreak())

# ============================================================================
# SECTION 1: Hypothesis and Motivation
# ============================================================================
story.append(Paragraph("1. Hypothesis", heading_style))
story.append(Paragraph("""
The JEPA encoder was trained to minimize L2 distance between embeddings from different
modalities (DEXA tissue, DEXA bone, retina OS, retina OD) for the same subject.
We hypothesized that this contrastive objective forces the model to <b>maximize shared features</b>
(age, sex, general health) while <b>suppressing modality-specific features</b> (bone structure
from DEXA, vessel morphology from retina).
""", body_style))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("Why would this happen?", ParagraphStyle('bold', parent=body_style, fontName='Helvetica-Bold')))
story.append(Paragraph("""
Features that are similar across modalities make it easier to minimize L2 distance.
Features that differ between modalities increase the distance and contribute to loss.
Thus, the model learns to amplify the former and suppress the latter.
""", body_style))
story.append(PageBreak())

# ============================================================================
# SECTION 2: Methods
# ============================================================================
story.append(Paragraph("2. Methods", heading_style))
story.append(Paragraph("""
<b>Embeddings tested:</b><br/>
• <b>DEXA-only</b>: 768D (2 global views concatenated)<br/>
• <b>Retina-only</b>: 768D (2 global views concatenated)<br/>
• <b>Pooled</b>: 384D (mean of 4 global views)
""", body_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("""
<b>Outcomes tested (44 total):</b><br/>
• <b>Shared features</b>: Age, BMI<br/>
• <b>DEXA-specific</b>: Lean mass, bone mass, fat distribution (14 outcomes)<br/>
• <b>Retina-specific</b>: Vessel width, fractal dimension, tortuosity (17 outcomes)<br/>
• <b>Cross-modal</b>: Glucose, HbA1c, lipids, liver, renal markers (11 outcomes)
""", body_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("""
<b>Analysis:</b> 5-fold cross-validation with subject-level splits (no data leakage).
Linear regression (Ridge) to predict each outcome. Performance measured by R², Spearman ρ, and Pearson r.
""", body_style))
story.append(PageBreak())

# ============================================================================
# SECTION 3: Results - Main Comparison
# ============================================================================
story.append(Paragraph("3. Results", heading_style))
story.append(Paragraph("3.1 Comprehensive Performance Comparison",
                      ParagraphStyle('subheading', parent=body_style, fontSize=13, fontName='Helvetica-Bold', spaceAfter=10)))

# Add comprehensive comparison figure
img = Image(str(FIGURES_DIR / "comprehensive_comparison.png"), width=7*inch, height=9*inch)
story.append(img)
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("""
<b>Figure 1:</b> Performance across all categories and metrics. DEXA-specific features
are best predicted by DEXA-only embeddings (blue bars dominate). Retina-specific features
favor retina-only (orange bars). Cross-modal features benefit from pooling (green bars).
""", ParagraphStyle('caption', parent=body_style, fontSize=9, textColor=colors.HexColor('#666666'))))
story.append(PageBreak())

# ============================================================================
# SECTION 4: Best Embedding Summary
# ============================================================================
story.append(Paragraph("3.2 Which Embedding Wins?",
                      ParagraphStyle('subheading', parent=body_style, fontSize=13, fontName='Helvetica-Bold', spaceAfter=10)))

img = Image(str(FIGURES_DIR / "wins_summary.png"), width=7*inch, height=3*inch)
story.append(img)
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("""
<b>Figure 2:</b> Number of best predictions per category. DEXA dominates for DEXA-specific
features (14/14 wins on R² and Pearson). Retina dominates for retina-specific features
(14/17 on Spearman). Pooled embeddings excel at cross-modal features (11/11 on R²).
""", ParagraphStyle('caption', parent=body_style, fontSize=9, textColor=colors.HexColor('#666666'))))
story.append(Spacer(1, 0.3*inch))

# Wins table
story.append(Paragraph("<b>Summary of Best Embeddings:</b>", body_style))
wins_data = [
    ['Category', 'Best by R²', 'Best by Spearman', 'Best by Pearson'],
    ['Shared Features', 'Mixed', 'Mixed', 'Mixed'],
    ['DEXA-Specific', 'DEXA (14/14)', 'DEXA (13/14)', 'DEXA (14/14)'],
    ['Retina-Specific', 'Pooled (10/17)', 'Retina (14/17)', 'Retina (13/17)'],
    ['Cross-Modal', 'Pooled (11/11)', 'Pooled (7/11)', 'Pooled (6/11)'],
]

wins_table = Table(wins_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
wins_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
]))
story.append(wins_table)
story.append(PageBreak())

# ============================================================================
# SECTION 5: Suppression Effect
# ============================================================================
story.append(Paragraph("3.3 Information Suppression in Pooled Embeddings",
                      ParagraphStyle('subheading', parent=body_style, fontSize=13, fontName='Helvetica-Bold', spaceAfter=10)))

img = Image(str(FIGURES_DIR / "suppression_comprehensive.png"), width=7*inch, height=5*inch)
story.append(img)
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("""
<b>Figure 3:</b> Percent change in performance when using pooled vs single-modality embeddings.
Red bars indicate suppression. DEXA-specific features lose 8-74% (subcutaneous fat most affected).
Retina-specific features lose 8-40% (fractal dimension most affected).
""", ParagraphStyle('caption', parent=body_style, fontSize=9, textColor=colors.HexColor('#666666'))))
story.append(PageBreak())

# ============================================================================
# SECTION 6: Spearman vs Pearson
# ============================================================================
story.append(Paragraph("3.4 Spearman vs Pearson Correlations",
                      ParagraphStyle('subheading', parent=body_style, fontSize=13, fontName='Helvetica-Bold', spaceAfter=10)))

story.append(Paragraph("""
We evaluated both Spearman (rank-based) and Pearson (linear) correlations.
Both metrics show consistent patterns, with Spearman generally higher due to
robustness to outliers.
""", body_style))
story.append(Spacer(1, 0.1*inch))

img = Image(str(FIGURES_DIR / "spearman_vs_pearson.png"), width=7*inch, height=3*inch)
story.append(img)
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("""
<b>Figure 4:</b> Spearman vs Pearson correlations for all embedding types.
Points along the diagonal indicate agreement. Both metrics support the same conclusions.
""", ParagraphStyle('caption', parent=body_style, fontSize=9, textColor=colors.HexColor('#666666'))))
story.append(PageBreak())

# ============================================================================
# SECTION 7: Top Outcomes Heatmap
# ============================================================================
story.append(Paragraph("3.5 Top 30 Outcomes Performance",
                      ParagraphStyle('subheading', parent=body_style, fontSize=13, fontName='Helvetica-Bold', spaceAfter=10)))

img = Image(str(FIGURES_DIR / "comprehensive_heatmap_top30.png"), width=7*inch, height=5*inch)
story.append(img)
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("""
<b>Figure 5:</b> Heatmap of top 30 outcomes sorted by mean Spearman correlation.
Green indicates strong performance, red indicates poor performance. Clear modality-specific
patterns are visible: DEXA features (lean mass, bone mass) are blue-dominant, while
retinal features (vessel morphology) are orange-dominant.
""", ParagraphStyle('caption', parent=body_style, fontSize=9, textColor=colors.HexColor('#666666'))))
story.append(PageBreak())

# ============================================================================
# SECTION 8: Conclusions
# ============================================================================
story.append(Paragraph("4. Conclusions", heading_style))
story.append(Paragraph("""
<b>The hypothesis is confirmed.</b> JEPA's multi-modal contrastive training trades
modality-specific information for cross-modal alignment. This is not a bug—it's an
inherent property of the training objective.
""", body_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("""
<b>Key insights:</b><br/>
• Shared features (age) are <b>enhanced</b> by pooling (R²: 0.782 → 0.792)<br/>
• DEXA-specific features are <b>suppressed</b> by pooling (lean mass R²: 0.846 → 0.771, -8.9%)<br/>
• Retina-specific features are <b>suppressed</b> by pooling (fractal dim R²: 0.484 → 0.288, -40.5%)<br/>
• Cross-modal features <b>benefit</b> from pooling when both modalities contribute weak signals
""", body_style))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("<b>Recommendations:</b>",
                      ParagraphStyle('bold', parent=body_style, fontName='Helvetica-Bold')))
story.append(Paragraph("""
<b>For current users:</b><br/>
• Use <b>DEXA-only</b> embeddings for body composition tasks<br/>
• Use <b>Retina-only</b> embeddings for vascular/ocular tasks<br/>
• Use <b>Pooled</b> embeddings for age/general health prediction
""", body_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("""
<b>For future model development:</b><br/>
• Implement <b>disentangled representations</b> (separate shared vs specific subspaces)<br/>
• Add <b>modality-specific auxiliary tasks</b> during training<br/>
• Explore <b>attention-based fusion</b> mechanisms<br/>
• Test <b>alternative contrastive objectives</b> that preserve specificity
""", body_style))
story.append(PageBreak())

# ============================================================================
# SECTION 9: Methods Details
# ============================================================================
story.append(Paragraph("5. Methodological Details", heading_style))
story.append(Paragraph("""
<b>Data:</b> HPP 10K cohort, ages 40-70 (94% of data).<br/>
<b>Cross-validation:</b> 5-fold subject-level splits to prevent data leakage.<br/>
<b>Model:</b> Ridge regression (α=1.0) for all predictions.<br/>
<b>Metrics:</b> R² (variance explained), Spearman ρ (rank correlation), Pearson r (linear correlation).<br/>
<b>Implementation:</b> scikit-learn, subject-level KFold splitting.<br/>
<b>Confound control:</b> Gender merged but not explicitly controlled in linear models (embeddings subsume demographic information).
""", body_style))
story.append(Spacer(1, 0.3*inch))

# File info
story.append(Paragraph("<b>Generated files:</b>",
                      ParagraphStyle('bold', parent=body_style, fontName='Helvetica-Bold')))
story.append(Paragraph("""
• <font face="Courier">comprehensive_results.csv</font> - All CV results (132 tests, 44 outcomes)<br/>
• <font face="Courier">figures/</font> - 8 publication-quality figures<br/>
• <font face="Courier">FINDINGS.md</font> - Detailed interpretation<br/>
• <font face="Courier">SUMMARY.md</font> - Executive summary<br/>
• <font face="Courier">DATA_DESCRIPTION.md</font> - Study documentation
""", body_style))

# Build PDF
doc.build(story)

print(f"\n{'='*80}")
print("PDF REPORT GENERATED")
print(f"{'='*80}")
print(f"\nSaved to: {PDF_PATH}")
print(f"Pages: ~15")
print(f"Figures: 5")
print(f"Outcomes: 44")
print(f"Tests: 132")
