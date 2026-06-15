# 🎫 Support Integrity Auditor (SIA)

> **MARS Open Projects 2026 — Problem Statement 1**
> A self-supervised ML system that catches priority mismatches in CRM support tickets — before they become SLA violations.

---

## 🧩 The Problem

Enterprise support queues are messy. A critical outage gets marked *Low* because the agent was busy. A routine password reset gets escalated to *Critical* because the user was persistent. The result?

- 🚨 Hidden crises that slip through unnoticed
- 🔔 False alarms that waste engineering time
- ⏱️ SLA breaches that could have been avoided
- 🔄 Inconsistent escalation practices across teams

SIA fixes this by independently inferring *how severe a ticket actually is* and comparing that against what a human assigned to it.

---

## 🗂️ Dataset

**Customer Support Tickets CRM Dataset**

| Field | Role in SIA |
|---|---|
| `Ticket_Subject` | Short-form severity cue |
| `Ticket_Description` | Primary semantic input |
| `Priority_Level` | Human-assigned label (what we audit) |
| `Issue_Category` | Structured metadata |
| `Ticket_Channel` | Structured metadata |
| `Resolution_Time_Hours` | Operational severity proxy |

---

## 🏗️ How It Works

```
CRM Tickets
    │
    ▼
┌─────────────────────────────────────┐
│     Stage 1 · Pseudo-Label Gen      │
│                                     │
│  🧠 LLM Severity    (Phi-3 Mini)    │
│  🔵 Cluster Signal  (MiniLM + KMeans)│
│  ⏱️  Resolution Time                 │
│  📋 Rule-Based Keywords             │
└──────────────┬──────────────────────┘
               │  Weighted Fusion
               ▼
         Pseudo Labels
               │
               ▼
┌─────────────────────────────────────┐
│   Stage 2 · DeBERTa-v3-Small        │
│   Fine-tuned binary classifier      │
│   Consistent  ↔  Mismatch           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Stage 3 · Evidence Dossier Gen    │
│   Zero-hallucination, traceable     │
└──────────────┬──────────────────────┘
               │
               ▼
       📊 Streamlit Dashboard
```

---

## 🔬 Stage 1 — Pseudo-Label Generation

No ground-truth mismatch labels exist, so SIA builds its own supervision signal using **four independent severity estimators**.

### 🧠 LLM Severity Signal
- **Model:** Phi-3 Mini Instruct
- **Inputs:** Subject, Description, Category, Channel
- **Output:** `0 = Low` · `1 = Medium` · `2 = High` · `3 = Critical`
- Captures semantic understanding of urgency and operational impact.

### 🔵 Embedding Cluster Signal
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Generates embeddings → K-Means clustering → cluster-level severity assignment
- Captures latent semantic relationships without relying on LLM reasoning.

### ⏱️ Resolution-Time Signal
- Longer resolution time → generally higher severity
- Converts resolution hours into a severity score via predefined thresholds
- Purely operational — no text processing required.

### 📋 Rule-Based Signal
- Lightweight keyword engine scanning for escalation triggers:
  `login failed`, `payment failed`, `outage`, `unauthorized access`, `service unavailable`, `fraud`
- High interpretability; feeds directly into dossier evidence.

### ⚖️ Weighted Fusion

```
Final Severity =
    0.30 × LLM Score
  + 0.30 × Cluster Score
  + 0.25 × Resolution Score
  + 0.15 × Rule Score
```

Fused score is mapped back to: **Low · Medium · High · Critical**

---

## 📊 Signal Agreement Analysis

### Agreement with Fused Label

| Signal | Agreement | Cohen's κ |
|---|---|---|
| Cluster | 0.660 | 0.313 |
| LLM | 0.633 | 0.243 |
| Resolution | 0.610 | 0.178 |
| Rule | 0.547 | 0.047 |

### Pairwise Agreement Between Signals

| Pair | Agreement |
|---|---|
| LLM ↔ Rule | 0.759 |
| LLM ↔ Resolution | 0.585 |
| Resolution ↔ Rule | 0.625 |
| Cluster ↔ Resolution | 0.520 |
| LLM ↔ Cluster | 0.519 |
| Cluster ↔ Rule | 0.533 |

> Moderate pairwise agreement confirms that each signal captures a distinct dimension of severity — validating the multi-signal fusion approach.

---

## 🏷️ Mismatch Label Construction

```
Inferred Severity ≠ Assigned Priority  →  Mismatch = 1
Inferred Severity = Assigned Priority  →  Mismatch = 0
```

| Type | Definition |
|---|---|
| 🔴 **Hidden Crisis** | Inferred Severity **>** Assigned Priority |
| 🟡 **False Alarm** | Assigned Priority **>** Inferred Severity |

---

## 🤖 Stage 2 — Fine-Tuned Classifier

- **Model:** `microsoft/deberta-v3-small`
- **Task:** Binary classification — `Consistent` vs `Mismatch`
- **Input:** Structured text combining category, channel, priority, resolution time, subject, and description

### Dataset Split

| Split | Proportion |
|---|---|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

Stratified sampling used throughout.

### Class Distribution

| Class | Share |
|---|---|
| Mismatch | 56.84% |
| Consistent | 43.16% |

Reasonably balanced — no oversampling or synthetic augmentation required.

---

## 📋 Stage 3 — Evidence Dossier Generation

Every flagged ticket gets a structured, traceable dossier:

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": "...",
  "feature_evidence": ["keyword: outage", "channel: email", "..."],
  "constraint_analysis": "...",
  "confidence": "..."
}
```

All evidence items trace back to actual ticket fields — **zero hallucination by design**.

---

## 📈 Evaluation Results

| Metric | Score |
|---|---|
| Accuracy | **91.75%** |
| Macro F1 | **0.9161** |
| Precision | 0.9153 |
| Recall | 0.9170 |
| Recall (Consistent) | 0.9131 |
| Recall (Mismatch) | 0.9208 |

### Confusion Matrix

| | Predicted: Consistent | Predicted: Mismatch |
|---|---|---|
| **Actual: Consistent** | 788 | 75 |
| **Actual: Mismatch** | 90 | 1047 |

---

## 🖥️ Streamlit Dashboard

The deployed app supports three modes:

**Single Ticket Analysis**
Enter a ticket manually and get instant mismatch prediction + evidence dossier.

**Batch Analysis**
Upload a CSV for bulk prediction and bulk dossier generation.

**Analytics Dashboard**
- Mismatch distribution
- Hidden Crisis vs False Alarm breakdown
- Signal contribution overview
- Severity delta heatmaps
- Category-level and channel-level analysis

🔗 **[Launch the App](https://supportintegrityauditoransul23113029.streamlit.app/)**

---

## ✅ Summary

| Component | Detail |
|---|---|
| Pseudo-label strategy | 4-signal weighted fusion (self-supervised) |
| Classifier | DeBERTa-v3-small, fine-tuned |
| Accuracy | 91.75% |
| Macro F1 | 0.9161 |
| Explainability | Evidence dossier per flagged ticket |
| Deployment | Streamlit (single + batch + dashboard) |

SIA turns an unsupervised problem into a reliable, production-grade auditing system — no manual labeling required.
