import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LEN = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadBertClassifier(nn.Module):
    def __init__(self, model_name, num_classes_a, num_classes_b, num_classes_c, dropout=0.2):
        super(MultiHeadBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.head_a = nn.Linear(hidden_size, num_classes_a)
        self.head_b = nn.Linear(hidden_size, num_classes_b)
        self.head_c = nn.Linear(hidden_size, num_classes_c)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0]
        x = self.proj(pooled_output)
        x = self.tanh(x)
        x = self.dropout(x)
        return self.head_a(x), self.head_b(x), self.head_c(x)

model = None
tokenizer = None
enc_a = None
enc_b = None
enc_c = None
id2a = None
id2b = None
id2c = None

def load_model():
    global model, tokenizer, enc_a, enc_b, enc_c, id2a, id2b, id2c
    
    print(f"Device: {device}")
    print("Model yükleniyor...")
    
    with open("/app/models/label_maps.json", "r", encoding="utf-8") as f:
        label_maps = json.load(f)
    
    enc_a = LabelEncoder()
    enc_b = LabelEncoder()
    enc_c = LabelEncoder()
    enc_a.classes_ = np.array(label_maps["flat_categories"]["ana_kategori"])
    enc_b.classes_ = np.array(label_maps["flat_categories"]["alt_kategori_1"])
    enc_c.classes_ = np.array(label_maps["flat_categories"]["alt_kategori_2"])
    
    num_a, num_b, num_c = len(enc_a.classes_), len(enc_b.classes_), len(enc_c.classes_)
    
    id2a = {i: c for i, c in enumerate(enc_a.classes_)}
    id2b = {i: c for i, c in enumerate(enc_b.classes_)}
    id2c = {i: c for i, c in enumerate(enc_c.classes_)}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = MultiHeadBertClassifier(MODEL_NAME, num_a, num_b, num_c)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('/app/models/model_finetune.pth'))
    else:
        model.load_state_dict(torch.load('/app/models/model_finetune.pth', map_location=torch.device('cpu')))
    
    model.to(device)
    model.eval()
    
    print(f"Model yüklendi ve {device}'a taşındı!")
    print(f"Sınıf sayıları → ana:{num_a}, alt1:{num_b}, alt2:{num_c}")

def predict_title(title, topk=3):
    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            title, truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        logits_a, logits_b, logits_c = model(input_ids, attention_mask)
        
        probs_a = torch.softmax(logits_a, dim=1).cpu().numpy()[0]
        probs_b = torch.softmax(logits_b, dim=1).cpu().numpy()[0]
        probs_c = torch.softmax(logits_c, dim=1).cpu().numpy()[0]
        
        top_a = np.argsort(probs_a)[::-1][:topk]
        top_b = np.argsort(probs_b)[::-1][:topk]
        top_c = np.argsort(probs_c)[::-1][:topk]
        
        return {
            "ana_kategori": id2a[top_a[0]],
            "ana_confidence": float(probs_a[top_a[0]]),
            "alt_kategori_1": id2b[top_b[0]],
            "alt1_confidence": float(probs_b[top_b[0]]),
            "alt_kategori_2": id2c[top_c[0]],
            "alt2_confidence": float(probs_c[top_c[0]]),
            "top_predictions": {
                "ana_top": [(id2a[j], float(probs_a[j])) for j in top_a],
                "alt1_top": [(id2b[j], float(probs_b[j])) for j in top_b],
                "alt2_top": [(id2c[j], float(probs_c[j])) for j in top_c]
            }
        }

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "device": str(device)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'title' not in data:
            return jsonify({"error": "Title gerekli"}), 400
        
        title = data['title'].strip()
        if not title:
            return jsonify({"error": "Boş title"}), 400
        
        topk = data.get('topk', 3)
        
        result = predict_title(title, topk)
        result['input_title'] = title
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        
        if not data or 'titles' not in data:
            return jsonify({"error": "Titles listesi gerekli"}), 400
        
        titles = data['titles']
        topk = data.get('topk', 3)
        
        if not isinstance(titles, list):
            return jsonify({"error": "Titles bir liste olmalı"}), 400
        
        results = []
        for title in titles:
            if isinstance(title, str) and title.strip():
                result = predict_title(title.strip(), topk)
                result['input_title'] = title.strip()
                results.append(result)
        
        return jsonify({"predictions": results, "count": len(results)})
        
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=False)