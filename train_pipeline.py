import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ==========================================
# STAGE 1: DATA PREPARATION & PSEUDO-LABELING
# ==========================================

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Ticket Subject'] = df['Ticket Subject'].fillna('')
    df['Ticket Description'] = df['Ticket Description'].fillna('')
    df['Full_Text'] = "Subject: " + df['Ticket Subject'] + " | Description: " + df['Ticket Description']
    df['Ticket Priority'] = df['Ticket Priority'].fillna('Medium').str.strip()
    df['Ticket Channel'] = df['Ticket Channel'].fillna('Email')
    df['Resolution Time'] = pd.to_numeric(df['Resolution Time'], errors='coerce').fillna(24.0)
    df['Ticket Type'] = df['Ticket Type'].fillna('General Enquiry')
    
    priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    df['priority_numeric'] = df['Ticket Priority'].map(priority_map).fillna(1)
    return df

def generate_pseudo_labels(df):
    print("Generating Signal A: Semantic Urgency Clustering...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(df['Full_Text'].tolist(), show_progress_bar=True, batch_size=32)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    cluster_urgency = []
    for i in range(4):
        subset = df[cluster_labels == i]
        text_blob = " ".join(subset['Full_Text'].str.lower())
        urgency_score = text_blob.count('urgent') + text_blob.count('broken') + text_blob.count('down') * 2
        cluster_urgency.append((i, urgency_score))
    
    cluster_urgency.sort(key=lambda x: x[1])
    rank_map = {cluster_idx: rank for rank, (cluster_idx, _) in enumerate(cluster_urgency)}
    df['inferred_severity_cluster'] = pd.Series(cluster_labels).map(rank_map)

    print("Generating Signal B: Resolution Time Thresholding...")
    df['resolution_threshold'] = df.groupby('Ticket Type')['Resolution Time'].transform(lambda x: x.quantile(0.75))
    df['inferred_severity_restime'] = (df['Resolution Time'] > df['resolution_threshold']).astype(int) * 2
    
    df['inferred_severity'] = ((df['inferred_severity_cluster'] + df['inferred_severity_restime']) / 2).round().astype(int)
    df['mismatch_label'] = (abs(df['inferred_severity'] - df['priority_numeric']) >= 2).astype(int)
    
    signal_agreement = (df['inferred_severity_cluster'] == df['inferred_severity_restime']).mean()
    print(f"Pseudo-Label Signal Pairwise Agreement Rate: {signal_agreement * 100:.2f}%")
    return df

# ==========================================
# STAGE 2: SUPERVISED DEBERTA FINE-TUNING
# ==========================================

class CRMDataset(Dataset):
    def __init__(self, texts, metadata_strs, labels, tokenizer, max_len=256):
        self.texts = texts
        self.metadata = metadata_strs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        combined_text = f"{self.metadata[idx]} [SEP] {self.texts[idx]}"
        inputs = self.tokenizer(combined_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_sia_classifier(df):
    df['metadata_str'] = "Channel: " + df['Ticket Channel'] + " | Resolution Hours: " + df['Resolution Time'].astype(str)
    X_text = df['Full_Text'].values
    X_meta = df['metadata_str'].values
    y = df['mismatch_label'].values

    X_text_train, X_text_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
        X_text, X_meta, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor([sum(class_counts) / c for c in class_counts], dtype=torch.float).to(device)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    train_dataset = CRMDataset(X_text_train, X_meta_train, y_train, tokenizer)
    val_dataset = CRMDataset(X_text_val, X_meta_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    base_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=2)
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query_proj", "value_proj"])
    model = get_peft_model(base_model, peft_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(2):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            targets.extend(batch['labels'].numpy())

    print("\n================ SYSTEM VERIFICATION RESULTS ================")
    print(f"Binary Classification Accuracy: {accuracy_score(targets, preds)*100:.2f}% (Req: >=83%)")
    print(f"Macro F1 Score: {f1_score(targets, preds, average='macro'):.4f} (Req: >=0.82)")
    print("=============================================================")
    
    model.save_pretrained("./sia_deberta_lora")
    tokenizer.save_pretrained("./sia_deberta_lora")

if __name__ == "__main__":
    if not os.path.exists('customer_support_tickets.csv'):
        # Creates quick mock structural data if you haven't downloaded the Kaggle CSV yet
        mock_df = pd.DataFrame({
            'Ticket ID': [f'T-{i}' for i in range(40)],
            'Ticket Subject': ['System failure'] * 20 + ['Typo fix'] * 20,
            'Ticket Description': ['Database dropped connection crashing frontend'] * 20 + ['Small visual text change needed'] * 20,
            'Ticket Priority': ['Low'] * 20 + ['Critical'] * 20,
            'Ticket Channel': ['Web-Portal'] * 40,
            'Resolution Time': [72.0] * 20 + [1.0] * 20,
            'Ticket Type': ['Technical Error'] * 20 + ['Billing'] * 20
        })
        mock_df.to_csv('customer_support_tickets.csv', index=False)
        
    processed_df = load_and_preprocess_data('customer_support_tickets.csv')
    labeled_df = generate_pseudo_labels(processed_df)
    train_sia_classifier(labeled_df)
