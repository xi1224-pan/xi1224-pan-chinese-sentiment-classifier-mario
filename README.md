# 中文情緒分類模型（BERT）

本專案使用 BERT 微調模型，針對 PTT、巴哈姆特的中文遊戲評論留言，進行情緒分類任務，分為：

- 負面（0）
- 中立（1）
- 正面（2）

 專案特色
- 使用 Huggingface Transformers 微調 `bert-base-chinese`
- 訓練資料為自行整理之 PTT 與巴哈姆特留言
- 分類精度達 88% validation accuracy（在 3 類別上）
- 提供 `predict.py` CLI 工具與 Streamlit Demo App

## 📂 專案結構

```
chinese-sentiment-classifier/
├── README.md
├── train.py              ← 模型訓練主程式（可自行補上）
├── predict.py            ← 輸入句子後返回情緒分類
├── requirements.txt      ← 套件需求
├── notebooks/            ← 原始訓練 Notebook
├── app/                  ← Streamlit Demo 小工具
└── results/              ← 可放訓練圖表
```

使用方式

1. 安裝套件
```bash
pip install -r requirements.txt
```

2. 執行 CLI 模式
```bash
python predict.py
# 輸入中文句子，回傳情緒與分數
```

3. 執行網頁版 Streamlit（互動）
```bash
cd app
streamlit run app.py
```

4. 記得修改 `model_path = "your_model_directory"` 來指定你的模型目錄！

模型架構
- 預訓練模型：bert-base-chinese
- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Epochs: 5, Batch size: 16
