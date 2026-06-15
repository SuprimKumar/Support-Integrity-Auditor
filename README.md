# Support Integrity Auditor (SIA) 🛡️
**Artificial Intelligence & Machine Learning Framework for Automated CRM Ticket Priority Triage Audit**

---

## 📌 MARS Open Projects 2026 — Problem Statement 1
[cite_start]In enterprise-scale CRM ecosystems, manual ticket triage is systematically vulnerable to agent fatigue bias, customer favoritism, and keyword anchoring[cite: 18]. [cite_start]When high-impact operational incidents are mislabeled as "Low" or routine complaints are artificially inflated to "Critical," Service Level Agreements (SLAs) are severely jeopardized, driving up customer churn[cite: 19]. 

[cite_start]**Support Integrity Auditor (SIA)** addresses a fundamentally more complex variant of the triage problem: **there are no pre-annotated mismatch labels**[cite: 21]. [cite_start]The architecture autonomously bootstraps its own binary supervision signals from raw ticket data alone [cite: 22][cite_start], fine-tunes a localized language transformer classifier [cite: 48, 50][cite_start], and constructs fully grounded, zero-hallucination Evidence Dossiers for every flagged deviation[cite: 29, 30].

---

## ⚙️ System Architecture & Workflow Pipeline
[cite_start]The framework processes incoming customer logs through three distinct and reproducible stages[cite: 39]:

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
       [STAGE 3: STRUCTURAL EVIDENCE LEDGER GENERATION]
    └── Zero-Hallucination JSON Evidence Dossier Extraction
                         │
                         ▼
         [STREAMLIT HOSTED AUDITING DASHBOARD UI]

Stage 1: Self-Supervised Pseudo-Label GenerationTo bootstrap a supervision target without human annotation bias , the pipeline extracts and fuses independent severity estimation signals:  LLM Semantic Severity Engine: Leverages an open-source Phi-3-mini-instruct model to perform zero-shot evaluation on raw text, scoring semantic urgency from 0 (Low) to 3 (Critical).  Embedding-Based Clustering Engine: Utilizes sentence-transformers/all-MiniLM-L6-v2 to map full ticket texts into deep semantic vector fields for semantic urgency grouping.  Resolution-Time Regression Proxy: Translates resolution time durations into operational severity metrics, treating longer system cycles as a high-friction proxy signal for severe system issues.  Linguistic Rule Engine: Runs regex pattern checking to detect immediate high-impact escalation keywords (e.g., outage, payment failed, crash, leak) to calibrate systemic interpretability.  Mathematical Fusion & Mismatch Labeling StrategyThe final underlying inferred severity level is calculated through an explicit weighted fusion policy:  $$\text{Severity}_{\text{Inferred}} = \text{Round}\big(0.30 \cdot S_{\text{LLM}} + 0.30 \cdot S_{\text{Cluster}} + 0.25 \cdot S_{\text{Resolution}} + 0.15 \cdot S_{\text{Rule}}\big)$$The composite result is systematically cross-checked against the agent-assigned priority field to map discrepancies:  Consistent (0): Inferred Severity aligns with Assigned Priority.Priority Mismatch (1): Objective data deviates drastically from human tags.  Hidden Crisis: Inferred Severity > Assigned Priority (Unhandled Operational Risk).  False Alarm: Assigned Priority > Inferred Severity (Internal Escalation Overhead).  📈 Signal Contribution & Pairwise Agreement AnalysisA multi-signal paradigm is mathematically justified by checking pairwise signal consensus statistics across the raw corpus data:  Pairwise Inter-Signal Agreement RatesLLM vs Cluster: 51.9%LLM vs Resolution Time: 58.5%LLM vs Rule Engine: 75.9%Cluster vs Resolution Time: 52.0%Cluster vs Rule Engine: 53.3%Resolution Time vs Rule Engine: 62.5%Ablation Analysis Insight: The moderate agreement rate profiles verify that each selected signal tracks a separate characteristic of the ticketing cycle. Fusing these unique views stabilizes the generated supervision signal and limits tracking noise.  Stage 2: Classifier Fine-TuningSIA trains a supervised text classification head on the computed pseudo-labels using microsoft/deberta-v3-small rather than relying on frozen zero-shot APIs.  Feature Vector Integration: Fuses raw text elements (Subject, Description) with structured metadata variables (Channel, Resolution Time) within the transformer's sequence layer.  Parameter-Efficient Tuning: Employs LoRA (Low-Rank Adaptation) adapters to lock the transformer base layers while fine-tuning target modules.  Class Balance Validation: The dataset features a stable balanced target layout (56.84% Mismatch / 43.16% Consistent), verified via per-class precision and recall auditing.  Stage 3: Evidence Dossier GenerationFor every discrepancy detected, SIA generates a fully trace-grounded ledger document according to the required schema:  JSON{
  "ticket_id": "TC-101",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": 3,
  "feature_evidence": [
    { "signal": "keyword", "value": "crash", "weight": "High" },
    { "signal": "resolution_time", "value": "27.61 hours", "interpretation": "Exceeds standard SLA limits" }
  ],
  "constraint_analysis": "The system analyzed the intake channel (Email) and text fields. Objective context reveals structural priority errors.",
  "confidence": 0.9482
}
🛡️ Zero-Hallucination Guardrail: Every item listed in the feature_evidence block is linked back to input entries. Any fabricated parameter immediately invalidates the entire audit file.  📊 Evaluation Results & Mandatory Threshold VerificationThe system was evaluated against the strict held-out validation split specified in the verification guidelines. It easily clears all mandatory project performance limits:  Metric Evaluation ParameterOfficial Minimum ThresholdSIA Framework ScorePass StatusBinary Classification Accuracy$\ge 83\%$91.75%PASSED ✅Macro F1 Score$\ge 0.82$0.9161PASSED ✅Per-Class Recall (Consistent)$\ge 0.78$0.9131PASSED ✅Per-Class Recall (Mismatch)$\ge 0.78$0.9208PASSED ✅Validation Confusion MatrixPlaintext                  Predicted Consistent    Predicted Mismatch
Actual Consistent         788                      75
Actual Mismatch            90                    1047
💻 Streamlit Web Application InterfaceThe interactive frontend has been deployed to the public Streamlit Sharing Cloud platform:  👉 Live Hosted URL: Support Integrity Auditor Live PortalCore Features Supported:Single-Ticket Audit Panel: Allows manual form entry to test individual logs and view the generated JSON dossier in real time.  Batch File Processing: Supports bulk CSV data uploads to instantly audit and output full matching error ledgers.  Analytical Metrics Dashboard: Displays tracking charts for discrepancy volume ratios, mismatch type frequencies, and top contributing signal profiles.  Severity Delta Heatmaps: Evaluates priority drift magnitudes across all active issue categories and intake communication channels.  
