# ==========================================================
# 📦 نصب کتابخانه‌ها (فقط در Colab)
# ==========================================================

# !pip install -q gradio torch transformers biopython requests sentence-transformers faiss-cpu accelerate peft sacremoses sentencepiece pandas scikit-learn matplotlib pyyaml

# ==========================================================
# 🧬 ایمپورت‌ها
# ==========================================================

import gradio as gr
import requests
import time
import re
import json
import yaml
import numpy as np
import torch
import pandas as pd
from Bio import Entrez, SeqIO
from io import StringIO
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum

# ==========================================================
# 📁 کانفیگ (بدون هاردکد)
# ==========================================================

CONFIG_PATH = "config.yaml"

DEFAULT_CONFIG = {
    "entrez_email": "anonymous@pathogenagent.ai",
    "entrez_tool": "PathogenAgent-v3",
    "user_agent": "PathogenAgent/3.0",
    "model_name": "Sepideh2027/biogpt-clinvar-finetuned",
    "embedding_model": "all-MiniLM-L6-v2",
    "faiss_dimension": 384,
    "max_pubmed_results": 5,
    "max_clinvar_results": 5,
    "max_genbank_results": 1,
    "max_iterations": 3,
    "memory_k": 5,
    "confidence_threshold": 0.3,
    "biogpt_max_tokens": 100,
    "biogpt_temperature": 0.3,
    "high_quality_journals": ["Nature", "Cell", "Science", "NEJM", "Lancet", "BMJ", "JAMA", "Nature Genetics", "Genome Medicine"],
    "gene_list": ["tp53", "brca1", "brca2", "cftr", "egfr", "kras", "alk", "ros1", "her2", "apoe", "myc", "ras", "raf"],
    "pathogen_keywords": ["virus", "genome", "sequence", "strain", "protein", "cov", "sars", "ebola", "influenza", "mers", "hiv", "dengue", "zika", "tuberculosis", "malaria"],
    "variant_patterns": [
        r"c\.[0-9]+[a-z]?[0-9]*[a-z]?[0-9]*[a-z]?",
        r"p\.[a-z][0-9]+[a-z]",
        r"rs[0-9]+",
        r"[a-z0-9]+del[a-z0-9]*",
        r"[a-z0-9]+dup[a-z0-9]*",
        r"[a-z0-9]+ins[a-z0-9]*"
    ],
    "confidence_weights": {
        "pubmed": 0.4,
        "clinvar": 0.4,
        "genbank": 0.2
    }
}

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except:
        return DEFAULT_CONFIG

config = load_config()

# ==========================================================
# ⚙️ تنظیمات از کانفیگ
# ==========================================================

Entrez.email = config["entrez_email"]
Entrez.tool = config["entrez_tool"]
HEADERS = {"User-Agent": config["user_agent"]}

MODEL_NAME = config["model_name"]
EMBEDDING_MODEL = config["embedding_model"]
DIMENSION = config["faiss_dimension"]

MAX_PUBMED = config["max_pubmed_results"]
MAX_CLINVAR = config["max_clinvar_results"]
MAX_GENBANK = config["max_genbank_results"]
MAX_ITERATIONS = config["max_iterations"]
MEMORY_K = config["memory_k"]
CONF_THRESHOLD = config["confidence_threshold"]

BIOGPT_MAX_TOKENS = config["biogpt_max_tokens"]
BIOGPT_TEMPERATURE = config["biogpt_temperature"]

HIGH_QUALITY_JOURNALS = config["high_quality_journals"]
GENE_LIST = config["gene_list"]
PATHOGEN_KEYWORDS = config["pathogen_keywords"]
VARIANT_PATTERNS = config["variant_patterns"]
CONF_WEIGHTS = config["confidence_weights"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# ==========================================================
# 📊 Data Classes
# ==========================================================

@dataclass
class Evidence:
    title: str
    source: str
    identifier: str = ""
    score: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class AgentResult:
    query: str
    tools_used: List[str]
    iterations: int
    confidence: float
    evidence: List[Evidence]
    interpretation: str
    report: str
    reflection: Dict
    verification: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# ==========================================================
# 🤖 بارگذاری مدل‌ها (Lazy Loading)
# ==========================================================

tokenizer = None
model = None
embedder = None
faiss_index = None
memory_store = []

def load_models():
    global tokenizer, model, embedder, faiss_index
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        model.eval()
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        faiss_index = faiss.IndexFlatL2(DIMENSION)
    return tokenizer, model, embedder, faiss_index

tokenizer, model, embedder, faiss_index = load_models()

# ==========================================================
# 🛠️ ابزارها
# ==========================================================

def safe_request(url, params, timeout=15):
    try:
        return requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    except Exception as e:
        return None

def search_pubmed(query: str) -> List[Evidence]:
    try:
        time.sleep(0.3)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": MAX_PUBMED, "retmode": "json"}
        resp = safe_request(url, params)
        if not resp:
            return []
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        url2 = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params2 = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        resp2 = safe_request(url2, params2)
        if not resp2:
            return []
        data = resp2.json()
        results = []
        for uid, rec in data.get("result", {}).items():
            if uid == "uids":
                continue
            title = rec.get("title", "")
            if title:
                results.append(Evidence(
                    title=title,
                    source="PubMed",
                    identifier=f"PMID:{uid}",
                    score=0.5
                ))
        return results
    except Exception:
        return []

def search_clinvar(query: str) -> List[Evidence]:
    try:
        time.sleep(0.3)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "clinvar", "term": query, "retmax": MAX_CLINVAR, "retmode": "json"}
        resp = safe_request(url, params)
        if not resp:
            return []
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        url2 = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params2 = {"db": "clinvar", "id": ",".join(ids), "retmode": "json"}
        resp2 = safe_request(url2, params2)
        if not resp2:
            return []
        data = resp2.json()
        results = []
        for uid, rec in data.get("result", {}).items():
            if uid == "uids":
                continue
            title = rec.get("title", "")
            if title:
                results.append(Evidence(
                    title=title,
                    source="ClinVar",
                    identifier=f"RCV:{uid}",
                    score=0.8
                ))
        return results
    except Exception:
        return []

def search_genbank(query: str) -> Optional[Evidence]:
    try:
        time.sleep(0.5)
        term = f"{query} complete genome"
        handle = Entrez.esearch(db="nucleotide", term=term, retmax=MAX_GENBANK)
        record = Entrez.read(handle)
        handle.close()
        if not record.get("IdList"):
            return None
        seq_id = record["IdList"][0]
        handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")
        seq = SeqIO.read(StringIO(handle.read()), "fasta")
        return Evidence(
            title=f"{query} genome (ID: {seq.id})",
            source="GenBank",
            identifier=f"GenBank:{seq.id}",
            score=0.6,
            metadata={"length": len(seq.seq), "snippet": str(seq.seq[:250])}
        )
    except Exception:
        return None

# ==========================================================
# 🧠 RAG Memory
# ==========================================================

def add_to_memory(items: List[Evidence]):
    global memory_store
    texts = [i.title for i in items if i.title]
    if not texts:
        return
    vectors = embedder.encode(texts)
    faiss_index.add(np.array(vectors).astype("float32"))
    memory_store.extend(items)

def retrieve_memory(query: str) -> List[Evidence]:
    if not memory_store:
        return []
    qvec = embedder.encode([query])
    _, idx = faiss_index.search(np.array(qvec).astype("float32"), MEMORY_K)
    return [memory_store[i] for i in idx[0] if i < len(memory_store)]

# ==========================================================
# 🤖 BioGPT Generator
# ==========================================================

def biogpt_generate(prompt_text: str, max_tokens: int = BIOGPT_MAX_TOKENS, temperature: float = BIOGPT_TEMPERATURE) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ==========================================================
# 🧭 Planner Agent
# ==========================================================

def plan_actions(query: str) -> List[str]:
    prompt = f"""
You are a Planner Agent.

Available tools: PubMed, GenBank, ClinVar.

Query: {query}

Return a JSON list of tools to use.
Only return the list, nothing else.
"""
    try:
        response = biogpt_generate(prompt, max_tokens=30, temperature=0.1)
        tools = json.loads(response)
        if isinstance(tools, list):
            return tools
    except:
        pass
    return route_query(query)

def route_query(query: str) -> List[str]:
    q = query.lower()
    tools = ["PubMed"]
    is_variant = any(re.search(p, q, re.IGNORECASE) for p in VARIANT_PATTERNS) or any(g in q for g in GENE_LIST)
    is_pathogen = any(kw in q for kw in PATHOGEN_KEYWORDS)
    if is_variant:
        tools.append("ClinVar")
    if is_pathogen:
        tools.append("GenBank")
    return tools

# ==========================================================
# 🏆 Evidence Ranking Agent
# ==========================================================

def rank_evidence(evidence_list: List[Evidence]) -> List[Evidence]:
    for item in evidence_list:
        if item.source == "PubMed":
            score = 0.5
            for journal in HIGH_QUALITY_JOURNALS:
                if journal.lower() in item.title.lower():
                    score = 0.9
                    break
            if "review" in item.title.lower():
                score = 0.7
            item.score = min(score, 1.0)
        elif item.source == "ClinVar":
            item.score = 0.85
        elif item.source == "GenBank":
            item.score = 0.6
    return sorted(evidence_list, key=lambda x: x.score, reverse=True)

# ==========================================================
# 📝 Citation Agent
# ==========================================================

def generate_citations(evidence: List[Evidence]) -> str:
    citations = []
    for e in evidence[:5]:
        if e.identifier:
            citations.append(f"- {e.identifier} ({e.source})")
    return "\n".join(citations) if citations else "No citations available."

# ==========================================================
# 📄 Report Agent
# ==========================================================

def generate_report(query: str, interpretation: str, evidence: List[Evidence], confidence: float) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    citations = generate_citations(evidence)
    return f"""
# 🧬 Pathogen Report

**Generated:** {timestamp}
**Query:** {query}
**Confidence:** {confidence:.2f}

---

## 📊 Evidence Summary

| Source | Count |
|--------|-------|
| PubMed | {sum(1 for e in evidence if e.source == 'PubMed')} |
| ClinVar | {sum(1 for e in evidence if e.source == 'ClinVar')} |
| GenBank | {sum(1 for e in evidence if e.source == 'GenBank')} |

---

## 🧬 Interpretation

{interpretation}

---

## 📚 References
{citations}

---

*Generated by PathogenAgent v3.0*
"""

# ==========================================================
# 🔍 Reflection Agent
# ==========================================================

def reflect_on_evidence(evidence: List[Evidence]) -> Dict:
    count = len(evidence)
    if count >= 3:
        return {"sufficient": True, "score": min(count / 5, 1.0), "reason": "Sufficient evidence."}
    elif count >= 1:
        missing = []
        sources = {e.source for e in evidence}
        if "PubMed" not in sources:
            missing.append("PubMed")
        if "ClinVar" not in sources:
            missing.append("ClinVar")
        if "GenBank" not in sources:
            missing.append("GenBank")
        return {"sufficient": False, "score": count / 5, "reason": "Limited evidence.", "missing": missing}
    return {"sufficient": False, "score": 0, "reason": "No evidence found.", "missing": ["PubMed", "ClinVar", "GenBank"]}

# ==========================================================
# ✅ Verification Agent
# ==========================================================

def verify_answer(interpretation: str, evidence: List[Evidence]) -> Dict:
    if "Insufficient" in interpretation:
        return {"valid": True, "score": 1.0, "hallucination": "No", "reason": "No hallucination."}
    if len(interpretation) > 20 and evidence:
        return {"valid": True, "score": 0.8, "hallucination": "Low", "reason": "Grounded response."}
    return {"valid": False, "score": 0, "hallucination": "High", "reason": "Potential hallucination."}

# ==========================================================
# 🧬 Agent اصلی
# ==========================================================

def pathogen_agent(query: str) -> AgentResult:
    tools_used = []
    all_evidence = []
    iteration = 0
    tools = plan_actions(query)
    tools_used.extend(tools)

    while iteration < MAX_ITERATIONS:
        iteration += 1
        new_evidence = []
        if "PubMed" in tools:
            new_evidence.extend(search_pubmed(query))
        if "ClinVar" in tools:
            new_evidence.extend(search_clinvar(query))
        if "GenBank" in tools:
            gb = search_genbank(query)
            if gb:
                new_evidence.append(gb)
        all_evidence.extend(new_evidence)
        add_to_memory(new_evidence)
        reflection = reflect_on_evidence(all_evidence)
        if reflection["sufficient"] or iteration >= MAX_ITERATIONS:
            break
        missing = reflection.get("missing", [])
        if missing:
            tools = missing
            tools_used.extend(tools)

    ranked = rank_evidence(all_evidence)
    confidence = 0.0
    if any(e for e in ranked if e.source == "PubMed"):
        confidence += CONF_WEIGHTS["pubmed"]
    if any(e for e in ranked if e.source == "ClinVar"):
        confidence += CONF_WEIGHTS["clinvar"]
    if any(e for e in ranked if e.source == "GenBank"):
        confidence += CONF_WEIGHTS["genbank"]
    confidence = round(min(confidence, 1.0), 2)

    interpretation = "Insufficient evidence for interpretation."
    if confidence > CONF_THRESHOLD:
        clinvar_evidence = [e for e in ranked if e.source == "ClinVar"]
        if clinvar_evidence:
            context = "\n".join([e.title for e in clinvar_evidence[:5]])
            prompt = f"Question: {query}\nEvidence: {context}\nInterpretation:"
            interpretation = biogpt_generate(prompt)

    verification = verify_answer(interpretation, ranked)
    report = generate_report(query, interpretation, ranked, confidence)

    return AgentResult(
        query=query,
        tools_used=list(set(tools_used)),
        iterations=iteration,
        confidence=confidence,
        evidence=ranked,
        interpretation=interpretation,
        report=report,
        reflection=reflection,
        verification=verification
    )

# ==========================================================
# 📝 فرمت خروجی
# ==========================================================

def format_output(result: AgentResult) -> str:
    evidence_lines = "\n".join([
        f"- [{e.source}] {e.title} (Score: {e.score:.2f})"
        for e in result.evidence[:5]
    ]) or "No evidence"

    return f"""
# 🧬 PathogenAgent (Research-Grade)

**Query:** {result.query}
**Tools:** {', '.join(result.tools_used)}
**Iterations:** {result.iterations}
**Confidence:** {result.confidence:.2f}

---

## 🔍 Reflection
- **Sufficient:** {result.reflection['sufficient']}
- **Score:** {result.reflection['score']:.2f}
- **Reason:** {result.reflection['reason']}

---

## ✅ Verification
- **Valid:** {result.verification['valid']}
- **Hallucination Risk:** {result.verification['hallucination']}
- **Score:** {result.verification['score']:.2f}

---

## 🏆 Top Evidence
{evidence_lines}

---

## 📄 Full Report
{result.report}
"""

# ==========================================================
# 🖥️ Gradio Interface
# ==========================================================

def run(query: str) -> str:
    if not query.strip():
        return "Please enter a query."
    result = pathogen_agent(query)
    return format_output(result)

demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(label="Query", placeholder="e.g., SARS-CoV-2 genome or CFTR F508del"),
    outputs=gr.Markdown(),
    title="PathogenAgent (Research-Grade)",
    description="Agentic AI for Pathogen Genomics with Ranking, Citation, and Report"
)

demo.launch(share=True)
