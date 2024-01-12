import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
import os
from data_loader.data_loader import load_data_from_csv


def load_data():
    return load_data_from_csv()


def tokenize_text(text, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokens = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    return tokens


def create_dataloader(inputs, labels, batch_size=32):
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_bert_model(train_dataloader, model, optimizer, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")


def save_model(model, output_dir='bert_model'):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
