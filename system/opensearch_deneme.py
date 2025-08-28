import json
import requests
import pickle
import tensorflow as tf


def send_bulk_chunk(chunk, index_name):
    bulk_payload = ""
    for item in chunk:
        index_cmd = {"index": {"_index": index_name}}
        bulk_payload += json.dumps(index_cmd) + "\n"
        bulk_payload += json.dumps(item, ensure_ascii=False) + "\n"

    response = requests.post(
        'http://localhost:9200/_bulk',
        headers={'Content-Type': 'application/x-ndjson'},
        data=bulk_payload.encode('utf-8')
    )
    
    if response.status_code != 200:
        print("err:", response.text)
    else:
        print("done: ", len(chunk), "docs")

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

with open('C:/Users/emre-/Desktop/mava/trendyol_unique.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

CHUNK_SIZE = 1000
INDEX_NAME = "trendyol-urunler"

for chunk in chunked(data, CHUNK_SIZE):
    send_bulk_chunk(chunk, INDEX_NAME)
    
"""
threshold = 0.45

model = tf.keras.models.load_model('models/model1.h5')

with open('models/tokenizer1.pkl', 'rb') as f1:
    tokenizer = pickle.load(f1)

with open('models/label_encoder1.pkl', 'rb') as f2:
    label_encoder = pickle.load(f2)

def predidictCategories(title, top_k=1):
    seq = tokenizer.texts_to_sequences([title])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=30, padding='post')
    predicts = model.predict(pad)
    top = predicts[0].argsort()[-top_k:][::-1]
    categories = label_encoder.inverse_transform(top)
    return categories

title = "bluetooth kulaklık"
predicted_category = predidictCategories(title, top_k=1)
print(predicted_category)
categories = str(predicted_category[0]).split('_') # main cat, sub 1, sub 2
print(categories)

print("---------------------model kullanmadan dtabase search---------------------")

query = {
    "query": {
        "match": {
            "title": {
                "query": title,
                "fuzziness": "AUTO"
            }
        }
    }
}

response = requests.get(
    "http://localhost:9200/trendyol-urunler/_search",
    headers={"Content-Type": "application/json"},
    data=json.dumps(query)
)

results = response.json()

for hit in results['hits']['hits']:
    print(f"- {hit['_source']['title']} (Marka: {hit['_source'].get('marka', '-')})")

print("---------------------model kullanarak dtabase search---------------------")

query = {
    "query": {
        "function_score": {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": title,
                                "fields": ["title", "marka"],
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"ana_kategori.keyword": categories[0]}},
                        {"term": {"alt_kategori_1.keyword": categories[1]}},
                        {"term": {"alt_kategori_2.keyword": categories[2]}}
                    ]
                }
            },
            "boost_mode": "multiply",  # veya "sum", istersen dene
            "score_mode": "multiply",
            "field_value_factor": {
                "field": "weighted_rating",
                "factor": 1,
                "missing": 0.1
            }
        }
    }
}


response = requests.get(
    "http://localhost:9200/trendyol-urunler/_search",
    headers={"Content-Type": "application/json"},
    data=json.dumps(query)
)

results = response.json()

for hit in results['hits']['hits']:
    source = hit['_source']
    print(f"- {source['title']} (Marka: {source.get('marka', '-')}, Rating: {source.get('weighted_rating', '-')})")
    """

