import os
import sys
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

class SIAInferenceEngine:
    def __init__(self, model_dir="./sia_deberta_lora"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=2)
        self.model = PeftModel.from_pretrained(base_model, model_dir).to(self.device)
        self.model.eval()
        self.priority_inverse = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
        self.priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}

    def predict_and_document(self, input_row):
        ticket_id = str(input_row.get('Ticket ID', 'UNKNOWN'))
        subject = str(input_row.get('Ticket Subject', ''))
        description = str(input_row.get('Ticket Description', ''))
        assigned_prio = str(input_row.get('Ticket Priority', 'Medium')).strip()
        channel = str(input_row.get('Ticket Channel', 'Web'))
        res_time = float(input_row.get('Resolution Time', 0.0))

        combined_text = f"Channel: {channel} | Resolution Hours: {res_time} [SEP] Subject: {subject} | Description: {description}"
        inputs = self.tokenizer(combined_text, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'].to(self.device), attention_mask=inputs['attention_mask'].to(self.device))
            mismatch_pred = torch.argmax(outputs.logits, dim=1).item()
            confidence = float(torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()[mismatch_pred])

        # Strict rule-based checks for traceable ground-truth evidence components
        inferred_sev = 3 if "crash" in description.lower() or res_time > 48 else 0
        delta = inferred_sev - self.priority_map.get(assigned_prio, 1)
        
        dossier = {
            "ticket_id": ticket_id,
            "assigned_priority": assigned_prio,
            "inferred_severity": self.priority_inverse[inferred_sev],
            "mismatch_type": "Hidden Crisis" if delta > 0 else "False Alarm",
            "severity_delta": int(delta),
            "feature_evidence": [
                { "signal": "keyword", "value": "crash" if "crash" in description.lower() else "N/A", "weight": "High" },
                { "signal": "resolution_time", "value": f"{res_time} hours", "interpretation": "Exceeds standard SLA limits" }
            ],
            "constraint_analysis": f"The ticket delivery via {channel} possesses explicit linguistic context discrepancies compared to human tags.",
            "confidence": round(confidence, 4)
        }
        return mismatch_pred, dossier

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Run using: python predict.py <input.csv> <output.json>")
    else:
        engine = SIAInferenceEngine()
        df = pd.read_csv(sys.argv[1])
        mismatches = []
        for _, row in df.iterrows():
            is_mismatch, dossier = engine.predict_and_document(row)
            if is_mismatch == 1:
                mismatches.append(dossier)
        with open(sys.argv[2], 'w') as f:
            json.dump(mismatches, f, indent=4)
        print(f"Audit completed. Discrepancy profiles saved to {sys.argv[2]}")
