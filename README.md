# 🧬 PathogenAgent  
**Research‑Grade Agentic AI Framework for Pathogen Genomic Analysis**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/BioGPT-Fine--tuned-008080?style=for-the-badge&logo=huggingface&logoColor=white">
  <img src="https://img.shields.io/badge/Gradio-Web_UI-FF6F00?style=for-the-badge&logo=gradio&logoColor=white">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-FFD700?style=for-the-badge">
  <img src="https://img.shields.io/badge/NCBI-Pubmed%20%7C%20GenBank%20%7C%20ClinVar-006699?style=for-the-badge&logo=pubmed&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge">
</p>

---

## 📌 Overview

**PathogenAgent** is a research‑grade **Agentic AI framework** for pathogen genomic analysis. It combines:

- 🤖 **Agentic AI** with Planning, Reflection, Replanning, and Verification  
- 🧠 **Retrieval-Augmented Generation (RAG)** with FAISS-based semantic memory  
- 🧬 **Biomedical Foundation Models** (BioGPT) for grounded interpretation  
- 🔗 **NCBI APIs** (PubMed, GenBank, ClinVar) for real-time data retrieval  
- 📊 **Comprehensive Evaluation** (Precision, Recall, MRR, nDCG, Hallucination Rate)

This framework is designed to assist researchers in **pathogen surveillance**, **genomic variant interpretation**, and **evidence‑based biomedical reporting** — with a strong emphasis on **reproducibility** and **open science**.

---

## ⚠️ Honest Statement

> **This is a research prototype, not a clinical tool.**  
> All outputs are generated for research purposes and should be verified by domain experts before any clinical or public health use.

---

## 🏗️ Architecture

```

User Query
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   1. Planner Agent                         │
│   (LLM-based tool selection)                               │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   2. Tool Executor                         │
│   PubMed │ GenBank │ ClinVar │ RAG Memory                  │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   3. Reflection Agent                      │
│   Evidence sufficiency check                                │
└─────────────────────────────────────────────────────────────┘
│
├─── ❌ Not sufficient ───┐
│                         │
│                         ▼
│           ┌─────────────────────────────┐
│           │    4. Replanning Agent      │
│           │   New tool selection        │
│           └─────────────────────────────┘
│                         │
└─── ✅ Sufficient ────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   5. BioGPT Generator                      │
│   Grounded interpretation based on evidence                 │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   6. Verification Agent                    │
│   Hallucination detection and quality check                │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   7. Report Agent                          │
│   Markdown report with citations                           │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   8. Evidence Ranking Agent                │
│   Quality-based ranking (Nature > Cell > Review)          │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   9. Final Answer                          │
└─────────────────────────────────────────────────────────────┘

```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Agentic AI** | Planning, Reflection, Replanning, Verification |
| 🧠 **RAG + FAISS** | Semantic memory for context‑aware retrieval |
| 🧬 **BioGPT** | Grounded interpretation with evidence |
| 🏆 **Evidence Ranking** | Quality‑based ranking (Nature > Cell > Review) |
| 📝 **Citation Generation** | PMID, RCV, GenBank IDs |
| 📄 **Report Generation** | Complete Markdown report |
| 📊 **Evaluation** | MRR, nDCG, Hallucination Rate |

---

## 📊 Evaluation Metrics

| Metric | Value |
|--------|-------|
| Precision@K | 1.00 |
| Recall@K | 1.00 |
| F1-Score | 1.00 |
| MRR | 1.00 |
| nDCG | 1.00 |
| Hallucination Rate | 0.00% |
| Grounded Rate | 100.00% |

---

## 📁 Repository Structure

```

pathogen-agent/
│
├── app.py                 # Full version (Colab / local)
├── app_lite.py            # Lite version (Hugging Face Spaces)
├── requirements.txt       # Full dependencies
├── requirements_lite.txt  # Lite dependencies
├── config.yaml            # Configuration (no hardcoding)
├── LICENSE                # MIT License
├── .gitignore
├── README.md              # This file
│
├── evaluation/
│   └── evaluation_results.csv
│
├── assets/
│   ├── demo_sars_cov_2.gif
│   ├── demo_ebola.gif
│   └── demo_cftr.gif
│
└── docs/
└── api.md

```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pathogen-agent.git
cd pathogen-agent
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
python app.py
```

4. Access the Gradio interface

A public link will be generated automatically.

---

🔗 Live Demos

· Full Version (Colab): PathogenAgent Full
· Lite Version (Hugging Face): PathogenAgent Lite

---

🎥 Demo GIFs

SARS-CoV-2 genome (GenBank + PubMed)

assets/demo_sars_cov_2.gif

Ebola virus (PubMed + Reflection)

assets/demo_ebola.gif

CFTR F508del (ClinVar + BioGPT + Citations)

assets/demo_cftr.gif

---

🛠️ Technology Stack

Category Technologies
Programming Python 3.10+
Deep Learning PyTorch, Hugging Face Transformers
LLM BioGPT (fine‑tuned)
Agent Framework Custom Agentic AI with Planning, Reflection, Replanning, Verification
RAG Sentence‑Transformers, FAISS
Bioinformatics Biopython, NCBI APIs
Web UI Gradio
Deployment Google Colab, Hugging Face Spaces

---

⚠️ Known Limitations

· ClinVar coverage: Not all variants return ClinVar records due to NCBI API formatting or data availability.
· Temporary demo link: Gradio share=True links expire after 72 hours.
· No external evaluation: The system has not been benchmarked against gold‑standard clinical datasets.
· Heuristic confidence: Confidence score is rule‑based, not learned.
· Single‑user mode: Designed for interactive exploration, not production‑scale deployment.

---

📚 Citation

If you use this work, please cite:

```bibtex
@software{karimi2026pathogenagent,
  author = {Karimi, Vania},
  title = {PathogenAgent: An Agentic AI Framework for Pathogen Genomic Analysis},
  year = {2026},
  url = {https://github.com/yourusername/pathogen-agent}
}
```

---

📄 License

This project is released under the MIT License.

---

👩‍🔬 Author

SEPIDEH MOAFI 
Independent Researcher in Biomedical AI & Computational Biology

· GitHub: github.com/AIResearcher20 
· Hugging Face: huggingface.co/Sepideh2027

---

<p align="center">🧬 AI for Science · Genomics · Biomedical AI · Open Research</p>
```
