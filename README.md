# med-crew

Agentic medical assistant that helps healthcare technicians analyze chest X-ray images and generate a structured radiology report with annotated outputs.

## What this project does

This project uses a multi-agent workflow (CrewAI) to:
- Analyze a chest X-ray image
- Detect possible anomalies
- Generate visual outputs (annotated image + heatmap)
- Produce a final medical report and export it to PDF

The user interface is a Gradio web app where you upload an X-ray image and download the generated report.

## Requirements

- Linux (or another OS with equivalent tools)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally
- `libreoffice` installed (used to convert DOCX to PDF)

## Setup

From the repository root:

```bash
cd medgemma_crew
uv sync
```

## Configure Ollama models

This project is configured to use these local models:
- `thiagomoraes/medgemma-1.5-4b-it:Q8_0`
- `qwen2.5:7b`

Start Ollama and pull the models:

```bash
ollama serve
```

In another terminal:

```bash
ollama pull thiagomoraes/medgemma-1.5-4b-it:Q8_0
ollama pull qwen2.5:7b
```

## Run the system

From `medgemma_crew/`:

```bash
cd src/
uv run python -m interface.app
```

Then open the local Gradio URL shown in the terminal (usually `http://127.0.0.1:7860`).

## How to use

1. Upload a chest X-ray image in the web interface.
2. Wait for the multi-agent pipeline to finish analysis.
3. Download the generated PDF report.

## Notes

- Temporary image outputs are stored in `medgemma_crew/src/medgemma_crew/outputs/` during execution.
- If PDF conversion fails, verify `libreoffice` is installed and available in your PATH.
