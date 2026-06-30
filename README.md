🧬 PathogenAgent

Research‑Grade Agentic AI for Pathogen Genomics and Biomedical Evidence Retrieval

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Gradio-Web_UI-FF6F00?style=for-the-badge&logo=gradio&logoColor=white">
  <img src="https://img.shields.io/badge/BioGPT-Fine--tuned-008080?style=for-the-badge&logo=huggingface&logoColor=white">
  <img src="https://img.shields.io/badge/RAG-FAISS-FFD700?style=for-the-badge">
  <img src="https://img.shields.io/badge/NCBI-Pubmed%20%7C%20GenBank%20%7C%20ClinVar-006699?style=for-the-badge&logo=pubmed&logoColor=white">
</p>

---

📌 Overview

PathogenAgent is a research‑grade biomedical AI system that retrieves and interprets genomic and clinical information from PubMed, GenBank, and ClinVar using a Retrieval‑Augmented Generation (RAG) pipeline.

The system combines:

· Real‑time data retrieval from NCBI APIs
· Semantic memory via FAISS vector search
· Grounded LLM inference using a fine‑tuned BioGPT model

It is designed as a practical tool for exploring pathogen genomics, variant interpretation, and biomedical literature retrieval — with a strong emphasis on reproducibility and evidence‑based responses.

---

✨ What It Actually Does

Capability Implementation
PubMed article retrieval Queries NCBI E‑utilities, returns titles and PMIDs
GenBank sequence access Fetches complete genome records and displays preview
ClinVar variant lookup Retrieves clinical significance records
Semantic memory (RAG) Stores retrieved evidence in a FAISS index and performs similarity search
Confidence scoring Heuristic score based on available evidence sources
BioGPT interpretation Generates grounded summaries only when sufficient evidence exists

---

🧠 Architecture (Real Implementation)

```
User Query
     │
     ▼
┌─────────────┐
│ Query Router │  → variant / pathogen / general
└─────────────┘
     │
     ├─────────────┬─────────────┬─────────────┐
     ▼             ▼             ▼             ▼
 PubMed       GenBank       ClinVar     FAISS Memory
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Grounded BioGPT  │  → only if evidence is sufficient
              └─────────────────┘
                        │
                        ▼
              Structured Markdown Output
```

---

🛠️ Technology Stack (What Was Actually Used)

Component Technology
Programming Python 3.10
Web interface Gradio
LLM BioGPT (fine‑tuned on ClinVar) – Sepideh2027/biogpt-clinvar-finetuned
Embedding model Sentence‑Transformers all‑MiniLM‑L6‑v2
Vector store FAISS (CPU)
Retrieval NCBI E‑utilities (PubMed, GenBank, ClinVar)
Bioinformatics Biopython
Deployment Google Colab (public Gradio link)

---

📊 Evidence Retrieval & Confidence

The system uses a simple but transparent heuristic:

Evidence Source Max Contribution
PubMed articles up to 0.4
ClinVar records up to 0.4
GenBank sequence up to 0.2

BioGPT is only invoked when confidence > 0.4 and ClinVar evidence is present.
This prevents hallucination and ensures that interpretations are grounded in actual clinical records.

---

🧬 How to Use (Local or Colab)

1. Clone the repository

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

4. Access the Gradio interface – a public link will be generated.

---

📁 Repository Structure

```
pathogen-agent/
│
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── assets/                # Screenshots
│   ├── CFTR_1.png
│   ├── CFTR_2.png
│   ├── CFTR_3.png
│   ├── TP53_1.png
│   ├── TP53_2.png
│   └── TP53_3.png
└── LICENSE                # MIT License
```

---

🔬 Sample Queries That Work

Query Type Example
Pathogen (genome) SARS-CoV-2 genome
Variant (clinical) CFTR F508del
Gene (general) BRCA1
Variant (cancer) TP53 R175H

---

🖼️ Demo Screenshots

Query: CFTR F508del

assets/CFTR_1.png
assets/CFTR_2.png
assets/CFTR_3.png

Query: TP53 R175H

assets/TP53_1.png
assets/TP53_2.png
assets/TP53_3.png

---

🔗 Live Demo (Temporary)

A public instance is available via Gradio sharing:

👉 PathogenAgent Live Demo

⚠️ This link is temporary (72 hours). For a permanent version, deploy on Hugging Face Spaces using the same codebase.

---

⚠️ Known Limitations (Honest Disclosure)

Limitation Explanation
ClinVar coverage Not all variants return ClinVar records due to NCBI API formatting or data availability
Temporary demo link Gradio share=True links expire after 72 hours
No external evaluation The system has not been benchmarked against gold‑standard clinical datasets
Heuristic confidence Confidence score is rule‑based, not learned
Single‑user mode Designed for interactive exploration, not production‑scale deployment

---

📜 License

This project is released under the MIT License.

---

👩‍🔬 Author

Sepideh Moafi 
Independent Researcher in Biomedical AI & Computational Biology

· GitHub: github.com/yourusername
· Hugging Face: huggingface.co/Sepideh2027
· Email: Vania Karimi@gmail.com

---

🧾 Citation

If you use this work, please cite:

```bibtex
@software{Moafi 2026pathogenagent,
  author = {Moafi, Sepideh},
  title = {PathogenAgent: Research-Grade Agentic AI for Pathogen Genomics},
  year = {2026},
  url = {https://github.com/AIResearcher20/pathogen-agent}
}
```

---

<p align="center">
  Built for reproducible biomedical AI research.
</p>
