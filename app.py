from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

model_path = "path/to/your/saved/model"
bert_model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_data = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

    input_ids = tokenized_data['input_ids'].to(device)
    attention_mask = tokenized_data['attention_mask'].to(device)

    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_label = probs.argmax(dim=1).item()

    return predicted_label, probs.tolist()[0]


@app.route('/predict', methods=['POST'])
def prediction():
    data = request.get_json(force=True)
    text = data['text']

    predicted_label, probabilities = predict(text)

    response = {
        'text': text,
        'predicted_label': predicted_label,
        'probabilities': probabilities
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
