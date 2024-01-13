import numpy as np
import torch
from transformers import AdamW, BertForSequenceClassification
from data_loader.data_loader import load_data_from_csv
from evaluations.evaluate import evaluate_model
from predictions.predict import predict
from recommendation.recommendation import train_bert_model, save_model, create_dataloader
from sklearn.model_selection import train_test_split


def main():
    tokenized_data, labels = load_data_from_csv()

    # Extract tensors to numpy arrays
    input_ids = tokenized_data['input_ids'].numpy()
    attention_mask = tokenized_data['attention_mask'].numpy()

    # Concatenate input_ids and attention_mask
    combined_input = np.concatenate((input_ids, attention_mask), axis=1)

    # Split dataset into train and validation sets
    train_data, val_data, labels_train, labels_val = train_test_split(
        combined_input,
        labels.numpy(),
        test_size=0.1,
        random_state=16
    )

    # Split the combined input back into input_ids and attention_mask
    input_ids_train, attention_mask_train = np.hsplit(train_data, 2)
    input_ids_val, attention_mask_val = np.hsplit(val_data, 2)

    # Create DataLoaders for train and validation sets
    train_dataloader = create_dataloader({'input_ids': torch.tensor(input_ids_train),
                                          'attention_mask': torch.tensor(attention_mask_train)},
                                         torch.tensor(labels_train))
    val_dataloader = create_dataloader({'input_ids': torch.tensor(input_ids_val),
                                        'attention_mask': torch.tensor(attention_mask_val)},
                                       torch.tensor(labels_val))

    # Initialize BERT model for sequence classification
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    optimizer = AdamW(bert_model.parameters(), lr=5e-5)

    train_bert_model(train_dataloader, bert_model, optimizer)

    save_model(bert_model)

    evaluate_model(bert_model, val_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Script completed successfully!")


if __name__ == "__main__":
    main()
