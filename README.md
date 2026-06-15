# Support Integrity Auditor (SIA) 🛡️
**Artificial Intelligence & Machine Learning Framework for Automated CRM Ticket Priority Triage Audit**

---

## 📌 MARS Open Projects 2026 — Problem Statement 1
[cite_start]In enterprise-scale CRM ecosystems, manual ticket triage is systematically vulnerable to agent fatigue bias, customer favoritism, and keyword anchoring[cite: 1, 2, 3, 18]. [cite_start]When high-impact operational incidents are mislabeled as "Low" or routine complaints are artificially inflated to "Critical," Service Level Agreements (SLAs) are severely jeopardized, driving up customer churn[cite: 19]. 

[cite_start]**Support Integrity Auditor (SIA)** addresses a fundamentally more complex variant of the triage problem: **there are no pre-annotated mismatch labels**[cite: 21]. [cite_start]The architecture autonomously bootstraps its own binary supervision signals from raw ticket data alone [cite: 22, 27][cite_start], fine-tunes a localized language transformer classifier [cite: 28, 50][cite_start], and constructs fully grounded, zero-hallucination Evidence Dossiers for every flagged deviation[cite: 29, 30].

---

## ⚙️ System Architecture & Workflow Pipeline
[cite_start]The framework processes incoming customer logs through three distinct and reproducible stages[cite: 39, 96]:

```text
       [Raw Customer Support CRM Tickets Stream]
                         │
                         ▼
       [STAGE 1: PSEUDO-LABEL SIGNAL GENERATION]
    ├── 1. LLM Semantic Engine (Phi-3 Mini Instruct)
    ├── 2. Clustering Urgency Engine (all-MiniLM-L6-v2)
    ├── 3. Resolution-Time Operational Proxy Engine
    └── 4. Deterministic Expression Pattern Rule Engine
                         │
                         ▼
             [WEIGHTED SEVERITY FUSION]
                         │
                         ▼
         [BINARY MISMATCH PSEUDO-LABELS (0/1)]
                         │
                         ▼
        [STAGE 2: TRANSFORMER CLASSIFIER TRAINING]
    └── Fine-Tuned DeBERTa-v3-Small (LoRA Adapter Tuning)
                         │
                         ▼
       [STAGE 3: STRUCTURAL EVIDENCE LEADGER GENERATION]
    └── Zero-Hallucination JSON Evidence Dossier Extraction
                         │
                         ▼
         [STREAMLIT HOSTED AUDITING DASHBOARD UI]
