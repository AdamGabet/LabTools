# Skill: PDF Visual Audit

Use this skill to verify the visual quality and content of generated PDF reports. Since the agent cannot natively read PDFs, this pipeline converts PDF pages to images for multimodal analysis.

## Workflow
1. **PDF Rendering**: Convert target `.pdf` file pages into high-resolution images (PNG).
2. **Visual Analysis**: Send images to the multimodal LLM (Gemma 4) with a specific auditing prompt.
3. **Verification**: Compare visual output against the `create-report.md` guidelines (e.g., "Is the hero figure present?", "Are fonts readable?", "Is the color scheme consistent?").
4. **Feedback Loop**: If discrepancies are found, the agent iterates on the `analysis.py` or `generate_report.py` script and regenerates the PDF.

## Auditing Checklist
- [ ] **Methods Page**: Is it bullet-pointed and monospace?
- [ ] **Hero Figure**: Does it have a descriptive title? Are top 6-10 results visible?
- [ ] **Color Palette**: Are increases red and decreases blue?
- [ ] **Layout**: Is the landscape orientation (`11x8.5`) respected for figures?
- [ ] **Readability**: Are axis labels and legends legible?

## Tooling Requirements
- `pdf2image` (requires `poppler-utils` installed on system)
- `pillow` for image handling
- Access to multimodal LLM endpoint
