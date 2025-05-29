from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = "your_model_directory"  # 修改為你實際模型存放位置
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_with_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        score = probs[0][2].item() - probs[0][0].item()  # 正面 - 負面
        label = torch.argmax(probs, dim=1).item()
    label_map = {0: "負面", 1: "中立", 2: "正面"}
    return {
        "情緒分類": label_map[label],
        "情緒分數": round(score, 4)
    }

if __name__ == "__main__":
    text = input("請輸入一句中文：")
    print(predict_with_score(text))
