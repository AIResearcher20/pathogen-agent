# 🧬 PathogenAgent

**Agentic AI System for Pathogen Genomics, Variant Interpretation, and Biomedical Knowledge Retrieval**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/BioGPT-Biomedical_LLM-008080?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gradio-Web_Interface-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LangChain-Agentic_AI-00A67E?style=for-the-badge" />
</p>

---

## 🌍 Overview

PathogenAgent is an interactive biomedical AI assistant that retrieves and summarizes information from PubMed, GenBank, and ClinVar using a fine-tuned BioGPT model.

The system provides a simple interface for researchers and students to explore pathogen genomics, genomic variants, and biomedical literature through natural language queries.

---

## 🎯 What This Project Does

- Accepts a user query (e.g., a pathogen name, gene, or variant)
- Searches PubMed for relevant scientific articles
- Retrieves genomic sequence data from GenBank
- Looks up clinical significance from ClinVar
- Displays results in a structured format using Gradio

The system uses a **fine-tuned BioGPT model** (BioGPT-ClinVar) to support interpretation and summarization tasks.

---

## ✨ Key Features

🔬 **PubMed Literature Search**
- Retrieves article titles based on user queries
- Returns up to 3 relevant results

🧬 **GenBank Sequence Retrieval**
- Fetches complete genome sequences for organisms
- Displays sequence length and first 200 bp

🧪 **ClinVar Variant Lookup**
- Searches for clinical significance of variants
- Returns available pathogenicity annotations

🤖 **BioGPT Integration**
- Uses the fine-tuned BioGPT-ClinVar model
- Supports interpretation and summarization

🌐 **Gradio Web Interface**
- Simple, clean interface for research use
- Real-time response display

---

## 🏗️ System Architecture

```

User Query
│
▼
PathogenAgent
│
┌───┼────────────┐
│   │            │
▼   ▼            ▼
PubMed GenBank ClinVar
│     │          │
└─────┼──────────┘
▼
Fine-tuned BioGPT
▼
Structured Output

```

---

## 🚀 Live Demo

Try the interactive application:

🔗 **Hugging Face Space**  
[https://huggingface.co/spaces/yourusername/pathogen-agent](https://huggingface.co/spaces/yourusername/pathogen-agent)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/pathogen-agent.git
cd pathogen-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python app.py
```

---

📁 Project Structure

```
PathogenAgent/
│
├── app.py                # Main application
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── LICENSE               # MIT License
└── .gitignore            # Ignored files
```

---

🧠 Model Information

The system uses BioGPT-ClinVar, a fine-tuned version of Microsoft BioGPT on ClinVar genomic variants.

· Base Model: microsoft/biogpt (150M parameters)
· Fine-tuning Method: LoRA
· Training Data: ~20,000 ClinVar variants
· Training Loss: 1.39 (final)
· Validation Loss: 1.40 (final)

The model is publicly available on Hugging Face Hub:
Sepideh2027/biogpt-clinvar-finetuned

---

📋 How It Works

1. User enters a query (e.g., "SARS-CoV-2 spike protein")
2. The agent determines which tools to use:
   · PubMed for articles
   · GenBank for genomic sequences
   · ClinVar for variant significance
3. Results are collected and displayed
4. BioGPT optionally provides summarization (if needed)

---

🛠️ Technology Stack

Category Tools
Programming Python 3.10
Deep Learning PyTorch, Hugging Face Transformers
LLM BioGPT (fine-tuned)
Fine-tuning LoRA (PEFT)
Agent Framework LangChain
Web Interface Gradio
Bioinformatics Biopython, NCBI API
Data Sources PubMed, GenBank, ClinVar

---

🔬 Research Applications

· Pathogen genomics exploration
· Genomic variant lookup
· Biomedical literature retrieval
· Educational tool for bioinformatics
· Research prototyping in biomedical AI

---

👩‍🔬 Author

Vania Karimi
Independent Researcher in Biomedical AI & Computational Biology

· GitHub: github.com/yourusername
· Hugging Face: huggingface.co/Sepideh2027

---

📜 Citation

If you use this work, please cite:

```bibtex
@software{karimi2026pathogenagent,
  author = {Karimi, Vania},
  title = {PathogenAgent: Agentic AI for Pathogen Genomic Analysis},
  year = {2026},
  url = {https://github.com/yourusername/pathogen-agent}
}
```

---

📄 License

This project is released under the MIT License.

---

⭐ Support

If you find this project useful:

· ⭐ Star the repository
· 🍴 Fork it for your own use
· 📚 Share it with fellow researchers

---

<p align="center">
  AI for Science • Genomics • Biomedical AI • Open Research
</p>
```

---

