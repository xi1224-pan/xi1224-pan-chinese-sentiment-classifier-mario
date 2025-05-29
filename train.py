import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 參數設定
PRETRAINED_MODEL = "bert-base-chinese"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# 自訂 Dataset
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 載入資料
df = pd.read_csv("data/odyssey_labels.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# 分割資料
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# tokenizer 與 dataset
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
train_dataset = CommentDataset(train_texts, train_labels, tokenizer)
val_dataset = CommentDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 模型設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 訓練模型
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} - Train loss: {total_loss / len(train_loader):.4f}")

    # 驗證準確率
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds += torch.argmax(outputs.logits, dim=1).cpu().tolist()
            targets += batch['labels'].cpu().tolist()
    acc = accuracy_score(targets, preds)
    print(f"Epoch {epoch+1} - Validation Accuracy: {acc:.4f}")

# 儲存模型
model.save_pretrained("models/final_model")
tokenizer.save_pretrained("models/final_model")
