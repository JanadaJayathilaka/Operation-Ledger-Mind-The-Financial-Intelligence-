# 🧠 Operation Ledger-Mind
### Enterprise Financial Question-Answering & Regulatory Compliance System

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers%20%7C%20PEFT%20%7C%20TRL-yellow.svg)](https://huggingface.co/)
[![Weaviate](https://img.shields.io/badge/VectorDB-Weaviate%20v4-green.svg)](https://weaviate.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ed.svg)](https://www.docker.com/)

---

## 📌 Executive Summary

Financial filings (such as SEC Form 10-K annual reports) present extreme challenges for standard Large Language Models:
- **Dense numerical and tabular data:** Minor errors in figures like *"$162.6 billion"* or *"GAAP operating income of $2.8 billion"* lead to catastrophic financial misinformation.
- **Outdated training knowledge:** Static base models cannot dynamically verify fiscal data released after their training cutoff date.
- **Lack of auditability:** Regulatory compliance demands verified, line-item source citations with page numbers rather than ungrounded assertions.

**Operation Ledger-Mind** resolves this through an end-to-end engineering solution based on the **Uber Technologies, Inc. 2024 Annual Report (SEC Form 10-K)**. It implements and compares two core paradigms:
1. **Domain-Specific Instruction Fine-Tuning (QLoRA):** Parameter-Efficient Fine-Tuning of `Qwen/Qwen2.5-1.5B-Instruct` on curated financial QA datasets.
2. **"The Librarian" Advanced Hybrid RAG:** A multi-stage retrieval architecture combining dense vector embeddings, sparse BM25 inverted indexes, Reciprocal Rank Fusion (RRF), and Cross-Encoder neural re-ranking.

---

## 🏛️ System Architecture & Pipeline Workflow

The system is engineered into four interconnected, modular stages:

### Stage 1: Data Factory & Dataset Preparation
* **Source:** Uber Technologies, Inc. 2024 Annual Report (`SEC Form 10-K` PDF).
* **Chunking:** Recursive Character Text Splitter producing 543 chunks (1,500 characters, 150 overlap) with retained metadata (`page`, `source`, `splitter`).
* **QA Generation:** 450 verified financial QA pairs (`question`, `context`, `answer`, `metadata`).
* **Splitting:** Chunk-aware stratified split (360 train samples, 90 test samples) avoiding context leakage.
* **Hub Publication:** Exported to Hugging Face Hub (`janadaj/operation-ledger-mind-qa`) in Parquet, JSONL, and CSV formats.

### Stage 2: Instruction Fine-Tuning Branch (QLoRA)
* **Base Architecture:** `Qwen/Qwen2.5-1.5B-Instruct`.
* **Quantization:** 4-bit NormalFloat (NF4) with double quantization via `bitsandbytes`.
* **Parameter Efficiency:** Low-Rank Adaptation (LoRA) injected into `[q_proj, k_proj, v_proj, o_proj]` ($r=16, \alpha=32$).
* **Training Execution:** Supervised Fine-Tuning (`SFTTrainer`) with ChatML format, Cosine LR scheduler ($2\times 10^{-5}$), gradient checkpointing, and FP16/FP32 precision alignment on Tesla T4 GPU.
* **Artifact:** Saved lightweight adapter weights (`outputs/adapter/`).

### Stage 3: "The Librarian" Advanced Hybrid RAG
* **Vector Database:** Local containerized Weaviate database (`FinancialReportChunks` collection).
* **Multi-Modal Retrieval:**
  * **Dense Retrieval:** SBERT `all-MiniLM-L6-v2` semantic vector search (384 dimensions).
  * **Sparse Retrieval:** BM25 keyword inverted index matching exact entities and dollar figures.
* **Refinement Stage 1 (Rank Fusion):** Reciprocal Rank Fusion (RRF, $k=60$) combining dense and sparse candidate pools without score calibration issues.
* **Refinement Stage 2 (Neural Re-ranking):** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scoring pairwise relevance and filtering top candidates down to the top-3 passages.
* **Grounded Generation:** Gemini synthesis with strict financial system prompt and mandatory line-item SEC Form 10-K page citations.

### Stage 4: Evaluation & Regulatory Compliance
* **Lexical Validation:** ROUGE-L F1 measuring Longest Common Subsequence against ground-truth SEC answers.
* **Semantic Validation:** Gemini LLM-as-a-Judge grading accuracy, completeness, and grounding on a 1–5 scale with reasoning.
* **Safety Suite:** 10 financial red-teaming vectors covering insider trading, GAAP manipulation, expense forgery, and regulatory non-compliance.

---

## ⚖️ Comprehensive Analysis: Fine-Tuning vs. Advanced RAG

Both paradigms were implemented, evaluated, and stress-tested on real SEC Form 10-K queries.

### Head-to-Head Comparison Table

| Evaluation Vector | Parameter-Efficient Fine-Tuning (QLoRA) | Advanced Hybrid RAG ("The Librarian") |
| :--- | :--- | :--- |
| **Factual Accuracy** | Moderate to High. Prone to minor numeric hallucinations on unseen or closely related figures. | **Exceptional.** Answers are directly extracted and synthesized from retrieved text. |
| **Auditability & Citations** | ❌ **None.** Model parameters store knowledge implicitly; cannot output verifiable source page numbers. | ✅ **Complete.** Every assertion carries exact SEC Form 10-K page citations. |
| **Knowledge Freshness** | ❌ **Static.** Requires dataset re-creation, retraining, and redeployment when a new 10-Q/10-K is filed. | ✅ **Instant.** Updating the vector store takes seconds via document ingestion—zero retraining required. |
| **Handling Specific Entities** | Good on trained patterns, but risks confusing figures (e.g., mixing up 2023 vs 2024 Gross Bookings). | **Flawless.** BM25 explicitly matches exact tokens like *"$162.6B"*, *"Item 8"*, or *"GAAP"*. |
| **Tone & Format Adherence** | ✅ **Native.** Internalizes domain style, ChatML tags, and concise financial answering syntax. | Moderate. Requires explicit system prompt engineering and few-shot formatting instructions. |
| **Compute & Infrastructure** | Training requires dedicated GPU (Tesla T4 15 GB VRAM); memory bandwidth sensitive. | Inference runs on CPU or standard cloud endpoints; database runs in lightweight Docker. |
| **Hallucination Risk on Unanswerables** | High risk of fabricating plausible answers when context is absent. | **Near zero.** Prompt constraint forces refusal if information is absent from retrieved chunks. |

### 🏆 The Verdict: What is the Best Approach and Why?

> **Verdict:** For enterprise financial question-answering and compliance auditing, **Advanced Hybrid RAG ("The Librarian") is definitively the superior architecture.**

#### Why Advanced RAG Wins:
1. **Regulatory Non-Negotiable (Audit Trail):** In finance, an answer without an auditable primary source citation is legally unusable. RAG provides exact page numbers and verbatim excerpts from the SEC filing.
2. **Numerical Exactness:** Hybrid Search (Dense + BM25) eliminates the vocabulary mismatch problem where dense embeddings miss exact financial terms like `"$162.6 billion"` or `"GAAP operating profit"`.
3. **Zero Knowledge Maintenance Cost:** When Uber publishes their 2025 Form 10-K, the RAG system only needs a 30-second document indexing run in Weaviate. A fine-tuned model would require data curation, hyperparameter re-tuning, and GPU training hours.

#### The Ideal Production Synergy (Hybrid-Architecture):
While RAG is superior for factual knowledge retrieval, the optimal enterprise design combines both:
* Use **Fine-Tuning** to teach a compact model (1.5B–3B) the precise **ChatML financial syntax, brevity, and refusal behavior**.
* Feed that fine-tuned model the **top-3 reranked contexts from "The Librarian" RAG pipeline** to achieve 100% factual accuracy with minimal latency and token cost.

---

## 🛠️ Tech Stack & Architectural Decisions

Every component was intentionally chosen to address a specific financial NLP challenge:

### 1. Vector Database: Weaviate v4 (Containerized in Docker)
* **Why Weaviate:** Unlike vector-only stores (such as basic Chroma or FAISS), Weaviate provides **native hybrid search** out of the box. It manages both an **HNSW dense vector index** and an **inverted BM25 index** in a unified engine with zero client-side synchronization lag.
* **Storage Reliability:** Running Weaviate via Docker with `-e PERSISTENCE_DATA_PATH="/var/lib/weaviate"` and local volume bind mounts ensures enterprise data persistence across restarts.

### 2. Dense Embeddings: SBERT (`sentence-transformers/all-MiniLM-L6-v2`)
* **Why:** Generates highly structured 384-dimensional dense semantic representations. It balances low computational footprint and rapid inference on CPU/GPU with high semantic retrieval quality.

### 3. Sparse Retrieval: BM25 (Inverted Index)
* **Why:** Semantic vector search alone frequently fails on specific financial identifiers (e.g., *"Form 10-K"*, *"Item 1A"*, *"GAAP"*, *"$37B"*). BM25 guarantees keyword-level precision for alphanumeric identifiers and dollar amounts.

### 4. Fusion Strategy: Reciprocal Rank Fusion (RRF, $k=60$)
* **Why RRF:** Raw dense distance scores (cosine/dot-product) and sparse BM25 scores (BM25Okapi TF-IDF) operate on fundamentally incompatible numerical scales. Normalizing them with linear weights (`alpha * dense + (1 - alpha) * sparse`) requires fragile, dataset-dependent calibration.  
  RRF bypasses score calibration by fusing **pure ranks**:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
  Any chunk scoring moderately well in *both* search modalities is prioritized over chunks that score high in only one.

### 5. Two-Stage Retrieval: Neural Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
* **Why a Cross-Encoder:** Bi-encoders (vector embeddings) encode queries and documents separately, missing deep token-level interactions. The Cross-Encoder takes `(query, passage)` pairs together through full transformer self-attention, scoring exact relevance with high precision. Re-ranking the top-10 RRF candidates down to the top-3 eliminates irrelevant context before prompting the LLM.

### 6. Fine-Tuning: Qwen2.5-1.5B-Instruct + QLoRA (bitsandbytes + PEFT + TRL)
* **Why Qwen2.5-1.5B:** Exceptional performance in structured reasoning, instruction following, and multilingual finance while remaining lightweight enough to fine-tune on a free-tier Google Colab Tesla T4 GPU (15 GB VRAM).
* **QLoRA (4-bit NF4 Quantization):** Freezes the 1.5B base model in 4-bit NormalFloat precision and trains low-rank adapter matrices ($r=16, \alpha=32$) on query, key, value, and output projections. This keeps trainable parameters at only **~4.35M (0.28% of total parameters)** with ~3.1 GB VRAM usage.

### 7. Evaluation Framework: Dual Evaluation (ROUGE-L + LLM-as-a-Judge)
* **ROUGE-L F1:** Provides lexical overlap baseline measuring Longest Common Subsequence (LCS) match against ground-truth SEC answers.
* **LLM-as-a-Judge (Gemini API):** Solves the limitation of lexical metrics by scoring answers on a calibrated 1–5 scale with qualitative reasoning across accuracy, completeness, context grounding, and financial tone.

---

## 📂 Project Structure

```
Operation-Ledger-Mind/
├── .env                                # API keys (GOOGLE_API_KEY, OPENROUTER_API_KEY, HF_TOKEN)
├── pyproject.toml                      # Project metadata & Python packaging configuration
├── requirements.txt                    # Project dependency manifest
├── artifacts/
│   ├── data/
│   │   ├── train.jsonl                 # Chunk-aware training split (360 samples)
│   │   ├── test.jsonl                  # Chunk-aware evaluation split (90 samples)
│   │   ├── qa_dataset.csv              # Full dataset tabular export
│   │   └── qa_dataset.parquet          # High-performance Parquet format
│   └── weaviate_data/                  # Persistent Weaviate database storage on disk
├── data/
│   └── pdfs/
│       └── AEE Mini Project 01 2024 Annual Report.pdf   # Source SEC 10-K Document
├── notebooks/
│   ├── 01_data_factory.ipynb           # Part 1: Parsing, Chunking, QA Generation, HF & Weaviate Ingestion
│   ├── 02_finetuning_intern.ipynb      # Part 2: QLoRA Fine-Tuning, Colab T4 Bridge, Guardrails & Red-Teaming
│   └── 03_advanced_rag.ipynb           # Part 3: "The Librarian" Hybrid RAG, RRF, Cross-Encoder, ROUGE-L Eval
└── src/
    ├── config/
    │   └── config.yaml                 # Centralized configuration (models, chunking, retrieval params)
    └── services/
        └── llm_services.py             # Centralized LLM and Embedding factory functions
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python:** 3.10+
- **Package Manager:** `uv` (recommended) or `pip`
- **Docker Desktop:** Required for local Weaviate containerization
- **API Keys:** Google Gemini API Key (`GOOGLE_API_KEY`) and optionally Hugging Face Token (`HF_TOKEN`)

### 2. Environment Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/operation-ledger-mind.git
cd operation-ledger-mind

# Install dependencies using uv
uv sync
uv add rouge-score
```

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_write_token_here
```

### 3. Start Local Weaviate Vector Database

Launch the persistent Weaviate container via Docker:
```powershell
docker run -d `
  -p 8080:8080 `
  -p 50051:50051 `
  --name weaviate-local `
  -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" `
  -v "${PWD}\artifacts\weaviate_data:/var/lib/weaviate" `
  cr.weaviate.io/semitechnologies/weaviate:1.28.2
```

Verify that Weaviate is healthy:
👉 [http://localhost:8080/v1/meta](http://localhost:8080/v1/meta)

---

## 💻 Running the Pipelines

### Part 1: Data Factory (`notebooks/01_data_factory.ipynb`)
1. Ingests the Uber 2024 Form 10-K PDF (`data/pdfs/`).
2. Generates 543 recursive chunks with metadata preservation (`page`, `source`, `splitter`).
3. Generates 450 verified financial QA pairs.
4. Pushes the dataset to Hugging Face Hub (`janadaj/operation-ledger-mind-qa`).
5. Indexes all 543 chunks into Weaviate collection `FinancialReportChunks`.

### Part 2: Instruction Fine-Tuning (`notebooks/02_finetuning_intern.ipynb`)
1. Designed for Google Colab (Tesla T4 GPU, 15 GB VRAM).
2. Loads `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit NF4 precision.
3. Attaches LoRA adapters ($r=16, \alpha=32$).
4. Trains via `SFTTrainer` with gradient checkpointing and mixed-precision (FP16/FP32).
5. Saves trained adapter to `outputs/adapter/`.
6. Executes 10 financial red-teaming compliance tests (Insider trading, fraud, MNPI).

### Part 3: "The Librarian" Advanced RAG (`notebooks/03_advanced_rag.ipynb`)
1. Connects to existing Weaviate database `FinancialReportChunks`.
2. Implements `query_librarian(question)`:
   - **Dense:** SBERT `all-MiniLM-L6-v2` (`near_vector`).
   - **Sparse:** Native BM25 keyword search.
   - **Fusion:** Reciprocal Rank Fusion (RRF, $k=60$).
   - **Re-ranking:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scoring top-3 passages.
   - **Generation:** Gemini synthesis with page citations.
3. Runs Sanity Check Evaluation computing **ROUGE-L F1** and **Gemini LLM-as-a-Judge (1–5 scale)** across test samples.

---

## 📊 Sample Pipeline Output

```text
======================================================================
📚 'THE LIBRARIAN' ADVANCED RAG PIPELINE
======================================================================
❓ Question: What was Uber's total Gross Bookings reported for the fiscal year 2024?
🔍 Hybrid Search:  10 Dense + 10 BM25 results
🔀 RRF Refinement: Merged into 10 candidates
🎯 Cross-Encoder:  Filtered to Top 3 passages
----------------------------------------------------------------------
💡 ANSWER:
According to the Uber Technologies, Inc. 2024 Annual Report (Form 10-K), 
total Gross Bookings for fiscal year 2024 grew 20% year-over-year to 
$162.6 billion, compared to $137.9 billion in 2023 [SEC Form 10-K Page 63]. 
Mobility Gross Bookings were $82.7 billion (up 24%) and Delivery Gross 
Bookings were $74.5 billion (up 17%) [SEC Form 10-K Page 57].
----------------------------------------------------------------------
📑 RE-RANKED CITATIONS:
  [1] Page 63.0 | CE Score: +4.5517 | RRF Score: 0.03101
  [2] Page 57.0 | CE Score: +1.4412 | RRF Score: 0.01587
  [3] Page 53.0 | CE Score: +0.4850 | RRF Score: 0.01613
======================================================================
📏 ROUGE-L F1:  0.8421 | ⭐ LLM Judge: 5/5
💬 Reasoning:   Exact dollar amount ($162.6B) and percentage growth (20%) match 
                the reference perfectly. Fully grounded in the cited SEC filing page.
```

---

## 🛡️ Financial Safety & Guardrails (Red-Teaming)

Financial models in enterprise deployments must be protected against regulatory liability. Notebook 2 and Notebook 3 include a dedicated **Financial Red-Teaming Suite** testing 10 risk vectors:

1. **Insider Trading / MNPI:** Refuses requests to trade ahead of earnings or exploit confidential material.
2. **Unlicensed Investment Advice:** Refuses personalized stock option recommendations and refuses to offer returns guarantees.
3. **Accounting Fraud / GAAP Manipulation:** Refuses instructions on concealing losses or distorting non-GAAP earnings.
4. **Market Manipulation:** Rejects generation of fake press releases designed to trigger short squeezes.
5. **SEC Disclosure Evasion:** Refuses advice on avoiding mandatory Form 8-K / 10-K event filings.
6. **Corporate Expense Forgery:** Rejects generation of fraudulent corporate travel receipts.
7. **Tax Evasion:** Refuses guidance on underreporting 1099 gross earnings to the IRS.
8. **Executive Defamation:** Rejects ungrounded allegations of executive embezzlement.
9. **Prompt Extraction / Jailbreak:** Withstands system prompt leakage attempts.
10. **Speculative Predictions:** Refuses to guarantee future stock price targets.

---

## 📜 Acknowledgments

* Grounded in public SEC EDGAR disclosures for **Uber Technologies, Inc. (2024 Form 10-K)**.
* Built using open-source tools and frameworks across the **Hugging Face**, **Weaviate**, **LangChain**, and **PyTorch** ecosystems.
