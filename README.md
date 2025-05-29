# 中文情緒分類模型（BERT）

本專案使用 BERT 微調模型，針對 PTT、巴哈姆特的中文遊戲評論留言，進行情緒分類任務，分為：

- 負面（0）
- 中立（1）
- 正面（2）

## 📌 專案特色
- 使用 Huggingface Transformers 微調 `bert-base-chinese`
- 訓練資料為自行整理之 PTT 與巴哈姆特留言
- 分類精度達 88% validation accuracy（在 3 類別上）
- 提供 `predict.py` CLI 工具與 Streamlit Demo App


## 📂 專案結構

```
chinese-sentiment-classifier/
├── README.md                     ← 專案說明文件
├── requirements.txt              ← 套件需求（transformers, torch, streamlit）
├── train.py                      ← 模型訓練主程式（BERT 微調）
├── predict.py                    ← 輸入中文句子 → 回傳情緒分類
├── data/
│   ├── irony_labels.csv          ← 反諷分類資料
│   ├── odyssey_labels.csv        ← 奧德賽評論標註資料
│   └── new_comments.csv          ← 最新評論資料
├── notebooks/
│   └── bert_sentiment_classifier.ipynb  ← 原始訓練筆記本
├── app/
│   └── app.py                    ← Streamlit Demo（目前不使用）
├── results/
│   └── training_summary.png      ← 訓練過程圖（loss/acc/val loss）
└── models/（可選）
    └── model.bin / config.json   ← 訓練完成模型參數（如需提供）
```

## 🖼️ 訓練結果範例
![](results/training_summary.png)
