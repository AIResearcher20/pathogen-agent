import gradio as gr
import requests
import time
import re
import numpy as np
import torch
from Bio import Entrez, SeqIO
from io import StringIO
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

# ==========================================================
# تنظیمات
# ==========================================================

Entrez.email = "anonymous@pathogenagent.ai"
Entrez.tool = "PathogenAgent-RAG"

HEADERS = {"User-Agent": "PathogenAgent/2.0"}

MODEL_NAME = "Sepideh2027/biogpt-clinvar-finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✅ Device: {device}")

# ==========================================================
# بارگذاری BioGPT
# ==========================================================

print("🔄 Loading BioGPT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model.eval()
print("✅ BioGPT loaded.")

# ==========================================================
# بارگذاری Embedding + FAISS
# ==========================================================

print("🔄 Loading Embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384
index = faiss.IndexFlatL2(dimension)
memory_store = []
print("✅ Embedding model loaded.")

# ==========================================================
# ابزارها
# ==========================================================

def search_pubmed(query, retmax=5):
    try:
        time.sleep(0.3)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"}
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        url2 = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params2 = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        resp2 = requests.get(url2, params=params2, headers=HEADERS, timeout=15)
        data = resp2.json()
        results = []
        for uid, rec in data.get("result", {}).items():
            if uid == "uids":
                continue
            results.append({"title": rec.get("title", ""), "source": "PubMed", "id": uid})
        return results
    except Exception as e:
        return [{"title": f"PubMed Error: {str(e)}", "source": "error"}]

def search_clinvar(query):
    try:
        time.sleep(0.3)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "clinvar", "term": query, "retmax": 5, "retmode": "json"}
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        url2 = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params2 = {"db": "clinvar", "id": ",".join(ids), "retmode": "json"}
        resp2 = requests.get(url2, params=params2, headers=HEADERS, timeout=15)
        data = resp2.json()
        results = []
        for uid, rec in data.get("result", {}).items():
            if uid == "uids":
                continue
            results.append({"title": rec.get("title", ""), "source": "ClinVar", "id": uid})
        return results
    except Exception as e:
        return [{"title": f"ClinVar Error: {str(e)}", "source": "error"}]

def search_genbank(query):
    try:
        time.sleep(0.5)
        term = f"{query} complete genome"
        handle = Entrez.esearch(db="nucleotide", term=term, retmax=1)
        record = Entrez.read(handle)
        handle.close()
        if not record.get("IdList"):
            return None
        seq_id = record["IdList"][0]
        handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")
        seq = SeqIO.read(StringIO(handle.read()), "fasta")
        return {
            "id": seq.id,
            "length": len(seq.seq),
            "snippet": str(seq.seq[:250]),
            "source": "GenBank"
        }
    except Exception as e:
        return {"error": str(e), "source": "error"}

# ==========================================================
# RAG Memory
# ==========================================================

def add_to_memory(items):
    global memory_store
    texts = [i["title"] for i in items if "title" in i]
    if not texts:
        return
    vectors = embedder.encode(texts)
    index.add(np.array(vectors).astype("float32"))
    memory_store.extend(items)

def retrieve_memory(query, k=5):
    if not memory_store:
        return []
    qvec = embedder.encode([query])
    _, idx = index.search(np.array(qvec).astype("float32"), k)
    return [memory_store[i] for i in idx[0] if i < len(memory_store)]

# ==========================================================
# BioGPT Grounded
# ==========================================================

def biogpt_generate(query, evidence):
    context = "\n".join([e["title"] for e in evidence[:5]])
    prompt = f"""
You are a biomedical AI.

Use ONLY evidence.

Question: {query}

Evidence:
{context}

Return concise interpretation:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ==========================================================
# Confidence Score
# ==========================================================

def confidence_score(pubmed, clinvar, genbank):
    score = 0.0
    if pubmed:
        score += min(len(pubmed) / 10, 0.4)
    if clinvar:
        score += min(len(clinvar) / 5, 0.4)
    if genbank and isinstance(genbank, dict) and "id" in genbank:
        score += 0.2
    return round(min(score, 1.0), 2)

# ==========================================================
# Query Router
# ==========================================================

def route_query(query):
    q = query.lower()
    
    variant_patterns = [
        r"c\.[0-9]+[a-z]?[0-9]*[a-z]?[0-9]*[a-z]?",
        r"p\.[a-z][0-9]+[a-z]",
        r"rs[0-9]+",
        r"[a-z0-9]+del[a-z0-9]*",
        r"[a-z0-9]+dup[a-z0-9]*",
        r"[a-z0-9]+ins[a-z0-9]*"
    ]
    
    gene_list = ["tp53", "brca1", "brca2", "cftr", "egfr", "kras", "alk", "ros1", "her2", "apoe", "myc", "ras", "raf"]
    pathogen_keywords = ["virus", "genome", "sequence", "strain", "protein", "cov", "sars", "ebola", "influenza", "mers", "hiv", "dengue", "zika"]
    
    for pattern in variant_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return "variant"
    for gene in gene_list:
        if gene in q:
            return "variant"
    for kw in pathogen_keywords:
        if kw in q:
            return "pathogen"
    return "general"

# ==========================================================
# Agent اصلی
# ==========================================================

def pathogen_agent(query):
    route = route_query(query)
    
    pubmed = search_pubmed(query)
    clinvar = search_clinvar(query) if route == "variant" else []
    genbank = search_genbank(query) if route == "pathogen" else None
    
    add_to_memory(pubmed + clinvar)
    memory = retrieve_memory(query)
    
    confidence = confidence_score(pubmed, clinvar, genbank)
    
    interpretation = "Insufficient evidence for AI interpretation."
    if confidence > 0.4 and clinvar:
        interpretation = biogpt_generate(query, clinvar)
    
    return {
        "query": query,
        "route": route,
        "confidence": confidence,
        "pubmed": pubmed,
        "clinvar": clinvar,
        "genbank": genbank,
        "memory": memory,
        "interpretation": interpretation
    }

# ==========================================================
# فرمت خروجی
# ==========================================================

def format_output(result):
    pubmed = "\n".join([f"- {p['title']}" for p in result["pubmed"]]) or "No results"
    clinvar = "\n".join([f"- {c['title']}" for c in result["clinvar"]]) or "No results"
    memory = "\n".join([f"- {m['title']}" for m in result["memory"]]) or "Empty"
    
    genbank = result["genbank"]
    if isinstance(genbank, dict):
        genbank = f"ID: {genbank['id']}\nLength: {genbank['length']:,} bp\n{genbank['snippet']}..."
    
    return f"""
# 🧬 PathogenAgent (Research-Grade)

**Query:** {result['query']}
**Type:** {result['route']}
**Confidence:** {result['confidence']}

---

## 📄 PubMed
{pubmed}

---

## 🧪 ClinVar
{clinvar}

---

## 🧬 GenBank
{genbank or "Not applicable"}

---

## 🧠 RAG Memory
{memory}

---

## 🤖 BioGPT Interpretation
{result['interpretation']}
"""

# ==========================================================
# Gradio Interface
# ==========================================================

def run(query):
    if not query.strip():
        return "Please enter a query."
    result = pathogen_agent(query)
    return format_output(result)

demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(label="Query", placeholder="e.g., SARS-CoV-2 genome or CFTR F508del"),
    outputs=gr.Markdown(),
    title="PathogenAgent (Research-Grade)",
    description="RAG + BioGPT + PubMed + GenBank + ClinVar"
)

demo.launch(share=True)
