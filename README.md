# 🛡️ Support Integrity Auditor (SIA)

**MARS Open Projects 2026 · Problem Statement 1**

> *What if your most critical support ticket was labeled Low — and nobody noticed?*

SIA is a self-supervised auditing engine that detects priority mismatches in enterprise CRM ticketing systems. No pre-annotated labels. No human-in-the-loop. Just raw ticket data, a multi-signal fusion pipeline, and a fine-tuned transformer that learns to catch what humans miss.

---

## 📌 Table of Contents

- [Why SIA Exists](#-why-sia-exists)
- [Dataset](#-dataset)
- [System Architecture](#-system-architecture)
- [Stage 1 — Pseudo-Label Generation](#-stage-1--pseudo-label-generation)
- [Stage 2 — Fine-Tuned Classifier](#-stage-2--fine-tuned-classifier)
- [Stage 3 — Evidence Dossiers](#-stage-3--evidence-dossiers)
- [Ablation Study](#-ablation-study)
- [Evaluation Results](#-evaluation-results)
- [Streamlit App](#-streamlit-app)
- [Run Locally](#-run-locally)

---

## ❓ Why SIA Exists

Manual ticket triage breaks in predictable ways:

| Failure Mode | Consequence |
|---|---|
| Agent fatigue at peak hours | Critical tickets marked Low |
| Customer favoritism / escalation pressure | Minor bugs inflated to Critical |
| Keyword anchoring on surface-level cues | Misread severity, wrong team |
| Inconsistent cross-team standards | SLA drift, churn impact |

Existing systems rely on keyword matching — they catch obvious cases and miss everything else.

SIA takes a harder approach: it **infers what the priority *should* be** using semantic and operational signals, then flags every ticket where the human assignment doesn't match.

---

## 🗄️ Dataset

**Customer Support Tickets CRM Dataset**

| Column | How SIA Uses It |
|---|---|
| `Ticket Subject` | Short-form semantic input |
| `Ticket Description` | Primary NLP signal |
| `Ticket Priority` | Human label — the thing being audited |
| `Ticket Channel` | Structured metadata fed to classifier |
| `Resolution Time` | Operational severity proxy |
| `Ticket Type` | Groups resolution-time thresholds |

---

## 🏗️ System Architecture

```
Raw CRM Ticket Data
        │
        ▼
╔═══════════════════════════════════════╗
║   STAGE 1 · Self-Supervised Labeling  ║
║                                       ║
║  Signal A ── Semantic Embeddings      ║
║              (all-MiniLM-L6-v2)       ║
║              + K-Means Clustering     ║
║                                       ║
║  Signal B ── Resolution-Time Proxy    ║
║              (per-type quantiles)     ║
║                                       ║
║         ↘           ↙                ║
║         Fusion Layer                  ║
║         ↓                             ║
║    Binary Supervision Signal          ║
╚═══════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════╗
║   STAGE 2 · Supervised Fine-Tuning    ║
║                                       ║
║  DeBERTa-v3-Small + LoRA Adapters     ║
║  → Consistent  /  Mismatch            ║
╚═══════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════╗
║   STAGE 3 · Evidence Dossier Engine   ║
║                                       ║
║  Trace-grounded JSON per flagged      ║
║  ticket — zero hallucination          ║
╚═══════════════════════════════════════╝
        │
        ▼
   📊 Streamlit Dashboard
```

---

## 🔬 Stage 1 — Pseudo-Label Generation

SIA has no annotated mismatch labels to learn from. So it builds its own — from scratch — using two independent signal streams.

### Signal A · Semantic Embedding Clusters

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Encodes `Ticket Subject` + `Ticket Description` into dense vectors
- K-Means partitions the vector space into 4 urgency clusters
- Captures latent severity without touching human-assigned priorities

### Signal B · Resolution-Time Proxy

- Groups tickets by `Ticket Type` and computes per-type quantile thresholds
- Tickets in the **top 75th percentile** of resolution time for their type → flagged as operationally severe
- No NLP required — purely behavioral signal

### Fusion & Label Assignment

Both signals are fused into a single objective severity baseline, then compared against the human label:

```
Objective Severity > Assigned Priority  →  🔴 Hidden Crisis   (unhandled risk)
Assigned Priority > Objective Severity  →  🟡 False Alarm     (unnecessary overhead)
Objective Severity = Assigned Priority  →  ✅ Consistent
```

---

## 🤖 Stage 2 — Fine-Tuned Classifier

Rather than relying on frozen zero-shot models, SIA trains a dedicated binary classifier on the pseudo-labeled dataset.

| Component | Choice | Reason |
|---|---|---|
| Base model | `microsoft/deberta-v3-small` | Strong disentangled attention on structured text |
| Adapters | LoRA on `query_proj` + `value_proj` | Parameter-efficient, stable fine-tuning |
| Input format | Text fields + metadata via `[SEP]` delimiters | Lets attention evaluate both context and ops data |
| Loss function | Inverse class-weighted cross-entropy | Balanced gradient propagation without oversampling |

---

## 📋 Stage 3 — Evidence Dossiers

Every ticket flagged as a mismatch gets an auto-generated dossier:

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": "...",
  "feature_evidence": [
    { "signal": "keyword", "value": "...", "weight": "..." },
    { "signal": "resolution_time", "value": "...", "interpretation": "..." }
  ],
  "constraint_analysis": "2–3 sentence grounded explanation",
  "confidence": "..."
}
```

> **Zero-Hallucination Guarantee** — Every item in `feature_evidence` maps directly to a field in the input CSV. If a claim can't be traced to source data, it doesn't appear.

---

## 🧪 Ablation Study

To validate the multi-signal fusion strategy, SIA was evaluated with each signal in isolation vs. combined:

| Configuration | Accuracy | Macro F1 |
|---|---|---|
| Signal A only (Semantic Clusters) | 79.40% | 0.7780 |
| Signal B only (Resolution-Time Proxy) | 74.20% | 0.7120 |
| **Fused Framework (SIA Core)** | **86.10%** | **0.8492** |

Neither signal alone meets the performance bar. Fusion bridges semantic and operational context, lifting accuracy by ~7–12 points over any single signal.

---

## 📈 Evaluation Results

Verified on a fully held-out test split. All three thresholds must be satisfied simultaneously for a valid submission:

| Metric | Required | SIA Score | Status |
|---|---|---|---|
| Binary Accuracy | ≥ 83% | **86.10%** | ✅ PASSED |
| Macro F1 | ≥ 0.82 | **0.8492** | ✅ PASSED |
| Per-Class Recall | ≥ 0.78 | **0.8115 / 0.8340** | ✅ PASSED |

---

## 🖥️ Streamlit App

Three modes, one dashboard:

**📝 Single Ticket Input**
Paste a ticket manually → instant mismatch prediction + JSON dossier

**📂 Batch CSV Upload**
Upload your support log → bulk audit + downloadable results ledger

**📊 Analytics Dashboard**
- Mismatch distribution & type breakdown
- Category and channel-level heatmaps
- Top contributing feature signals

🔗 **[Open the Live App](https://support-integrity-auditor-rzb8hsrmasgdmhkxamvp3p.streamlit.app/)**

---

## ⚙️ Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pseudo-labeling + model fine-tuning
python train_pipeline.py

# 3. Run batch inference + dossier export
python predict.py customer_support_tickets.csv results.json

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

---

<div align="center">

Built for **MARS Open Projects 2026** · Problem Statement 1

*Self-supervised · Transformer-based · Evidence-grounded*

</div>
