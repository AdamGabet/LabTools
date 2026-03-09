"""
Generate clean, readable PDF report
- Large fonts (14+)
- One figure per page
- Using Pearson correlation only
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from pathlib import Path
import pandas as pd

# Paths
OUTPUT_DIR = Path("/home/adamgab/PycharmProjects/LabTools/research/jepa_modality_fusion_hypothesis")
FIGURES_DIR = OUTPUT_DIR / "figures"
PDF_PATH = OUTPUT_DIR / "JEPA_Report_Clean.pdf"

# Load results
results = pd.read_csv(OUTPUT_DIR / "comprehensive_results.csv")

print("Generating clean PDF report...")

# Create PDF - smaller margins for larger figures
doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter,
                       topMargin=0.5*inch, bottomMargin=0.5*inch,
                       leftMargin=0.5*inch, rightMargin=0.5*inch)

# Styles - LARGE FONTS
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'Title',
    parent=styles['Title'],
    fontSize=28,
    textColor=colors.HexColor('#1f77b4'),
    spaceAfter=20,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

subtitle_style = ParagraphStyle(
    'Subtitle',
    fontSize=18,
    textColor=colors.HexColor('#555555'),
    spaceAfter=16,
    alignment=TA_CENTER,
    fontName='Helvetica'
)

heading_style = ParagraphStyle(
    'Heading',
    fontSize=20,
    textColor=colors.HexColor('#2ca02c'),
    spaceAfter=14,
    spaceBefore=20,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'Body',
    fontSize=14,
    spaceAfter=14,
    alignment=TA_JUSTIFY,
    leading=20,
    fontName='Helvetica'
)

caption_style = ParagraphStyle(
    'Caption',
    fontSize=12,
    textColor=colors.HexColor('#666666'),
    spaceAfter=10,
    alignment=TA_LEFT,
    leading=16,
    fontName='Helvetica-Oblique'
)

# Build story
story = []

# ============================================================================
# TITLE PAGE
# ============================================================================
story.append(Spacer(1, 1.5*inch))
story.append(Paragraph("JEPA Modality Fusion", title_style))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("Shared vs Specific Features", subtitle_style))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph("HPP 10K Multi-Modal Embeddings Analysis", subtitle_style))
story.append(Spacer(1, 1.5*inch))

# Key finding
finding = """
<b><font size=16>KEY FINDING:</font></b><br/><br/>
<font size=14>
Multi-modal JEPA training <b>amplifies shared features</b> (age, BMI)
but <b>suppresses modality-specific information</b>.<br/><br/>

• DEXA-specific features lose up to <b>73% performance</b> when pooled<br/>
• Retina-specific features lose up to <b>56% performance</b> when pooled<br/>
• Shared features (age) <b>improve 18%</b> when pooled<br/>
</font>
"""
story.append(Paragraph(finding, body_style))
story.append(Spacer(1, 0.5*inch))

# Summary stats
summary = """
<font size=14>
<b>Study Summary:</b><br/>
• <b>44 outcomes</b> tested across 4 categories<br/>
• <b>132 experiments</b> (5-fold cross-validation)<br/>
• <b>Metric:</b> Pearson correlation<br/>
• <b>Subjects:</b> 6,445 with both DEXA and Retina scans<br/>
</font>
"""
story.append(Paragraph(summary, body_style))
story.append(PageBreak())

# ============================================================================
# HYPOTHESIS
# ============================================================================
story.append(Paragraph("Hypothesis", heading_style))
story.append(Paragraph("""
<font size=14>
The JEPA encoder was trained to minimize L2 distance between DEXA and retina embeddings
from the same person. We hypothesized that this objective forces the model to:<br/><br/>

1. <b>Maximize shared features</b> (age, sex, general health status)<br/>
2. <b>Suppress modality-specific features</b> (bone structure, vessel morphology)<br/><br/>

<b>Why?</b> Features similar across modalities reduce L2 distance (low loss).
Features that differ increase distance (high loss). The model learns to amplify
the former and suppress the latter.
</font>
""", body_style))
story.append(PageBreak())

# ============================================================================
# FIGURE 1: Shared Features
# ============================================================================
story.append(Paragraph("Figure 1: Shared Features", heading_style))
story.append(Spacer(1, 0.2*inch))

img = Image(str(FIGURES_DIR / "clean_1_shared.png"), width=7.5*inch, height=5*inch)
story.append(img)
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("""
<font size=14>
<b>Result:</b> Pooled embeddings achieve the highest correlation for Age (r=0.85).
By forcing DEXA and retina to align, the model enhances shared age signals present in both modalities.<br/><br/>

BMI is primarily visible in DEXA (body composition), so DEXA-only embeddings perform best (r=0.96).
</font>
""", caption_style))
story.append(PageBreak())

# ============================================================================
# FIGURE 2: DEXA-Specific Features
# ============================================================================
story.append(Paragraph("Figure 2: DEXA-Specific Features", heading_style))
story.append(Spacer(1, 0.2*inch))

img = Image(str(FIGURES_DIR / "clean_2_dexa_specific.png"), width=7.5*inch, height=5.5*inch)
story.append(img)
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("""
<font size=14>
<b>Result:</b> DEXA-only embeddings (blue) dominate for all body composition features.
Pooled embeddings (green) show consistent degradation.<br/><br/>

<b>Evidence of suppression:</b> Lean mass, weight, and fat distribution are
best predicted from DEXA-only. Pooling with retina degrades performance.
</font>
""", caption_style))
story.append(PageBreak())

# ============================================================================
# FIGURE 3: Retina-Specific Features
# ============================================================================
story.append(Paragraph("Figure 3: Retina-Specific Features", heading_style))
story.append(Spacer(1, 0.2*inch))

img = Image(str(FIGURES_DIR / "clean_3_retina_specific.png"), width=7.5*inch, height=5.5*inch)
story.append(img)
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("""
<font size=14>
<b>Result:</b> Retina-only embeddings (orange) dominate for vessel morphology features.
Pooled embeddings show degradation.<br/><br/>

<b>Evidence of suppression:</b> Retinal fractal dimension, vessel width, and tortuosity
are best predicted from retina-only. Pooling with DEXA suppresses this information.
</font>
""", caption_style))
story.append(PageBreak())

# ============================================================================
# FIGURE 4: Cross-Modal Features
# ============================================================================
story.append(Paragraph("Figure 4: Cross-Modal Features", heading_style))
story.append(Spacer(1, 0.2*inch))

img = Image(str(FIGURES_DIR / "clean_4_cross_modal.png"), width=7.5*inch, height=5.5*inch)
story.append(img)
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("""
<font size=14>
<b>Result:</b> Pooled embeddings (green) excel when both modalities contribute weak signals.<br/><br/>

<b>Interpretation:</b> Glucose, HbA1c, and immune markers are challenging to predict
from imaging alone. When both DEXA and retina provide subtle cues, pooling combines
them effectively.
</font>
""", caption_style))
story.append(PageBreak())

# ============================================================================
# FIGURE 5: DEXA Suppression
# ============================================================================
story.append(Paragraph("Figure 5: DEXA Information Loss", heading_style))
story.append(Spacer(1, 0.2*inch))

img = Image(str(FIGURES_DIR / "clean_5_dexa_suppression.png"), width=7.5*inch, height=6*inch)
story.append(img)
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("""
<font size=14>
<b>Result:</b> Red bars show performance degradation from pooling. Nearly all
DEXA-specific features are suppressed.<br/><br/>

<b>Worst cases:</b> Subcutaneous fat (-73%), Visceral fat (-56%), Android fat mass (-46%).
These features are invisible in retina, so pooling destroys this information.
</font>
""", caption_style))
story.append(PageBreak())

# ============================================================================
# FIGURE 6: Retina Suppression
# ============================================================================
story.append(Paragraph("Figure 6: Retina Information Loss", heading_style))
story.append(Spacer(1, 0.2*inch))

img = Image(str(FIGURES_DIR / "clean_6_retina_suppression.png"), width=7.5*inch, height=6*inch)
story.append(img)
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("""
<font size=14>
<b>Result:</b> Red bars show performance degradation from pooling. Most
retina-specific features are suppressed.<br/><br/>

<b>Worst cases:</b> Vein tortuosity (-56%), Artery tortuosity (-48%), Vein fractal dimension (-30%).
These features are invisible in DEXA, so pooling destroys this information.
</font>
""", caption_style))
story.append(PageBreak())

# ============================================================================
# CONCLUSIONS
# ============================================================================
story.append(Paragraph("Conclusions", heading_style))
story.append(Paragraph("""
<font size=14>
<b>The hypothesis is confirmed.</b><br/><br/>

JEPA's multi-modal contrastive training trades modality-specific information
for cross-modal alignment. This is <b>not a bug</b>—it's an inherent property
of the L2 minimization objective.<br/><br/>

<b>Recommendations for users:</b><br/>
• Use <b>DEXA-only</b> for body composition tasks<br/>
• Use <b>Retina-only</b> for vascular/ocular tasks<br/>
• Use <b>Pooled</b> for age and general health prediction<br/><br/>

<b>For future development:</b><br/>
• Implement disentangled representations (shared vs specific subspaces)<br/>
• Add modality-specific auxiliary tasks during training<br/>
• Explore attention-based fusion mechanisms<br/>
</font>
""", body_style))
story.append(PageBreak())

# ============================================================================
# METHODS
# ============================================================================
story.append(Paragraph("Methods", heading_style))
story.append(Paragraph("""
<font size=14>
<b>Data:</b> HPP 10K cohort, ages 40-70<br/>
• 8,652 subjects with DEXA scans<br/>
• 9,677 subjects with retina scans<br/>
• 6,445 subjects with both modalities<br/><br/>

<b>Embeddings:</b><br/>
• DEXA-only: 768 dimensions (2 global views concatenated)<br/>
• Retina-only: 768 dimensions (2 global views concatenated)<br/>
• Pooled: 384 dimensions (mean of 4 global views)<br/><br/>

<b>Analysis:</b><br/>
• 5-fold cross-validation, subject-level splits<br/>
• Ridge regression (α=1.0)<br/>
• Performance metric: <b>Pearson correlation</b><br/><br/>

<b>Outcomes tested:</b> 44 total<br/>
• Shared: Age, BMI (2)<br/>
• DEXA-specific: Body composition (14)<br/>
• Retina-specific: Vessel morphology (17)<br/>
• Cross-modal: Metabolic, immune, liver, renal (11)<br/>
</font>
""", body_style))

# Build PDF
doc.build(story)

print(f"\n{'='*80}")
print("CLEAN PDF REPORT GENERATED")
print(f"{'='*80}")
print(f"\nSaved to: {PDF_PATH}")
print(f"\nFormat:")
print(f"  - Large fonts (14-28pt)")
print(f"  - One figure per page")
print(f"  - Pearson correlation only")
print(f"  - 6 main figures + title + conclusions")
print(f"  - Total: ~12 pages")
