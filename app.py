import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

st.title("中文情緒分析小工具")
text = st.text_area("請輸入一句中文", "我覺得這款遊戲超好玩！")

model_path = "your_model_directory"  # 修改為你的模型資料夾路徑
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        score = probs[0][2].item() - probs[0][0].item()
        label = torch.argmax(probs, dim=1).item()
    label_map = {0: "負面", 1: "中立", 2: "正面"}
    return label_map[label], round(score, 4)

if st.button("分析情緒"):
    label, score = predict(text)
    st.success(f"分類結果：{label}（信心分數：{score}）")
