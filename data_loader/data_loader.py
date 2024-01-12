import pandas as pd
import re

import torch
from transformers import BertTokenizer

from configs.config_folder import DATA_PATH


def clean_text(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text


def load_data_from_csv():
    data_path = DATA_PATH
    imdb_data = pd.read_csv(data_path)

    imdb_data['review'] = imdb_data['review'].apply(clean_text)

    # Tokenize input text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_data = tokenizer(imdb_data['review'].tolist(), padding=True, truncation=True, return_tensors='pt')

    # Convert sentiments to tensor
    labels = torch.tensor(imdb_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0))

    # Check if the number of samples is consistent
    if len(tokenized_data['input_ids']) != len(labels):
        raise ValueError("Inconsistent number of samples between tokenized data and labels.")
    return tokenized_data, labels
