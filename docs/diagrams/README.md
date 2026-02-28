# Diagram images (Report & BRD)

This folder contains the diagrams used in the Project Report and BRD as **PNG** and **Mermaid source** (`.mmd`).

## PNG files (generated)

| File | Description |
|------|-------------|
| `01_report_system_architecture.png` | Figure 1 – System architecture (User, Frontend, Backend, Milvus, ML/NLP) |
| `02_report_fusion_weight_selection.png` | Figure 1b – Fusion weight selection (both/one/none text) |
| `03_report_similarity_check_sequence.png` | Figure 1c – Similarity-check sequence diagram |
| `04_report_detection_pipeline.png` | Figure 2 – Detection pipeline (upload → preprocess → … → top-K) |
| `05_report_text_score_pipeline.png` | Figure 3 – Text score pipeline (OCR → transliterate → match → score) |
| `06_report_roadmap_gantt.png` | Figure 4 – 5-year roadmap Gantt |
| `07_brd_traceability.png` | BRD – Stakeholder need → FR → component traceability |

## Regenerating PNGs

- **Option A:** Use [Kroki](https://kroki.io): POST each `.mmd` file to `https://kroki.io/mermaid/png` with `Content-Type: text/plain`.
- **Option B:** Run `python3 render_mermaid_to_png.py` (requires `requests`). It uses the Kroki API; if a diagram times out, run it again or try the SVG endpoint and convert to PNG locally.
- **Option C:** Use [Mermaid Live Editor](https://mermaid.live/) or `npx @mermaid-js/mermaid-cli mmdc -i file.mmd -o file.png` for local rendering.
