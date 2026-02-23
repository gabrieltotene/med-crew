# 🩺 MedGemma Crew: Multi-Agent System for Radiological Analysis

This project represents an advanced application of Generative AI and Computer Vision in healthcare. The system leverages the CrewAI library to orchestrate specialized agents that analyze chest X-ray images, generate technical diagnoses, and produce formal medical reports.

## 🚀 Architecture Overview

The ecosystem is divided into three fundamental pillars:

- **Vision and Detection:** Integration of the MedGemma-1.5-4b model (via Ollama) for textual analysis and TorchXRayVision for spatial localization.
- **Agent Orchestration:** Workflow that separates responsibilities among visual analysis, report writing, and image annotation.
- **Interface and Delivery:** Gradio frontend for image upload and automation tools for generating .docx and .pdf documents.

## 🛠️ Component Details

### 1. Agents

The project defines a multidisciplinary digital team based on specific roles:

- **X-ray Analyzer:** Acts as a senior radiologist, identifying anomalies and anatomical features.
- **Report Writer:** Specialist in medical documentation, transforming technical data into structured reports.
- **Image Annotator:** Image processing expert, responsible for visually marking pathologies.
- **Medical Writer:** Consolidates all analyses and file paths into a concise final report.

### 2. Custom Tools

These tools enable agents to interact with the external world and process data:

- **OpenAIImageTool:** Manages communication with the local MedGemma model, encoding images in Base64 for API analysis.
- **XRayAnomalyLocatorTool:** Uses the DenseNet121 model to predict pathologies and generates gradient-based heatmaps to locate anomalies.
- **FerramentaEscreverDocx:** Processes Markdown content, dynamically inserts images, and converts the result to PDF.

### 3. The Role of MedGemma

The MedGemma model is the clinical intelligence engine of the project. During testing and implementation, it excelled at:

- Identifying specific anatomical regions (e.g., left lower lobe, mediastinum).
- Assessing the technical quality of the image and describing visual features such as opacities and consolidations.
- Providing detailed descriptions that serve as the basis for differential diagnosis.

## 📈 Execution Pipeline

1. **Input:** The user uploads the image via the Gradio interface.
2. **Analysis:** The xray_analyzer agent uses the image tool to obtain MedGemma's technical description.
3. **Localization:** The image_annotator generates the heatmap and the circled image with detected anomalies.
4. **Writing:** The report_writer generates the formal report in technical format.
5. **File Generation:** The system compiles the final Markdown (including generated images) into a PDF ready for download.

## 🔍 Technical Differentials

- **Data Robustness:** The use of RobustConverter ensures that, even if the language model generates out-of-format responses, the system attempts to recover valid JSON to maintain workflow continuity.
- **Hybrid Analysis:** Combines generative AI (MedGemma) for qualitative description with discriminative AI (TorchXRayVision) for quantitative and spatial validation.
- **Resource Management:** The system includes cleanup routines to delete temporary output images after process completion, preventing file accumulation on the server.

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
