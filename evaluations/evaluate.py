from datetime import datetime
import torch
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F


def evaluate_model(model, val_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    classification_rep = classification_report(all_labels, all_preds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{timestamp}.txt"

    with open(report_filename, "w") as report_file:
        report_file.write(f"Evaluation Report ({timestamp}):\n")
        report_file.write(f"Validation Accuracy: {accuracy:.4f}\n\n")
        report_file.write("Classification Report:\n")
        report_file.write(classification_rep)

    print(f"Evaluation report saved to: {report_filename}")
    return accuracy
