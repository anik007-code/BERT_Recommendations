import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def load_dataset(path):
    dataset = pd.read_csv(path)
    return dataset


class BanglaEmotionsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }


class EmotionClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train(model, train_loader, criterion, optimizer, device, accumulate_grad_steps=8):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        if (step + 1) % accumulate_grad_steps == 0 or step == len(train_loader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            val_preds.extend(preds.tolist())
            val_targets.extend(labels.tolist())
    val_acc = accuracy_score(val_targets, val_preds)
    return val_acc


dataset_path = 'your_dataset.csv'
dataset = load_dataset(dataset_path)

train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=7)
model = BertModel.from_pretrained('bert-base-multilingual-cased', config=config)

for param in model.parameters():
    param.requires_grad = False

train_dataset = BanglaEmotionsDataset(train_df, tokenizer, max_length=128)
val_dataset = BanglaEmotionsDataset(val_df, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduce batch size
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # Reduce batch size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionClassifier(model, num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_acc = evaluate(model, val_loader, criterion, device)
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
