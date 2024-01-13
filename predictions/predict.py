import torch
from transformers import BertTokenizer
import torch.nn.functional as F

def predict(model, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_data = tokenizer(texts, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokenized_data['input_ids'].to(device)
    attention_mask = tokenized_data['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_labels = probs.argmax(dim=1).cpu().numpy()
    return predicted_labels, probs.cpu().numpy()


