from flask import Flask, request, jsonify, render_template_string
import requests
import json
import time
import random
import numpy as np
import logging
import traceback
import google.generativeai as genai
import re
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

THRESHOLD = 0.25
OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "trendyol-urunler"
ML_SERVICE_URL = "http://localhost:8080"

api_key = os.getenv('GEMINI')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ml_service():
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

MODEL_LOADED = check_ml_service()
if MODEL_LOADED:
    logger.info("Docker ML servisi başarıyla bağlandı")
else:
    logger.error("Docker ML servisi bağlantısı başarısız")

def extract_product_info(user_query):
    prompt = f"""
        Kullanıcının sorgusu: "{user_query}"

        Bu sorgudan ürün arama bilgilerini çıkar ve sadece JSON formatında yanıtla:

        {{
            "product_keywords": "temel ürün anahtar kelimeleri (örneğin "samsung telefon 256gb kırmızı")",
            "brands": ["marka1", "marka2"] veya null, örneğin ["sleepy", "samsung"]
            "price_min": sayı(float) veya null,  örneğin 400 tl üzeri ayakkabı : 400.0,
            "price_max": sayı(float) veya null, örneğin 800 tl altında ayakkabı : 800.0,
            "rating_min": 1-5 arası sayı veya null, örneğin 4 yıldız üzeri diyorsa 4 döndür,
            "search_intent": "Kullanıcının ne aradığına dair kısa açıklama",
            "llm_response": "Kullanıcıya gösterilecek doğal dil yanıtı (maksimum 2-3 cümle)"
        }}

        Örnekler:
        - "500-1000 lira arası samsung telefon" -> price_min: 500, price_max: 1000, brands: ["samsung"]
        - "4 yıldız üzeri bluetooth kulaklık" -> rating_min: 4
        - "kırmızı spor ayakkabı nike" -> "product_keywords": "kırmızı spor ayakkabı" , brands: ["nike"]

        Sadece JSON yanıtı ver, başka metin ekleme.
        """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return {
                "product_keywords": user_query,
                "brands": None,
                "price_min": None,
                "price_max": None,
                "rating_min": None,
                "search_intent": f"'{user_query}' araması yapılıyor",
                "llm_response": f"'{user_query}' için uygun ürünleri buluyorum."
            }
    except Exception as e:
        logger.error(f"Gemini API hatası: {e}")
        return {
            "product_keywords": user_query,
            "brands": None,
            "price_min": None,
            "price_max": None,
            "rating_min": None,
            "search_intent": f"'{user_query}' araması yapılıyor",
            "llm_response": f"'{user_query}' için uygun ürünleri arıyorum."
        }

def generate_search_response(products, product_info, query_text):
    try:
        if not products:
            no_results_prompt = f"""
                                    Kullanıcı "{query_text}" araması yaptı ama sonuç bulunamadı.
                                    Farklı anahtar kelimeler denemesini öneren 1-2 cümlelik yardımcı bir yanıt ver.
                                    """
            response = model.generate_content(no_results_prompt)
            return response.text.strip()
        
        results_prompt = f"""
                            Kullanıcı "{query_text}" araması yaptı ve {len(products)} ürün bulundu.
                            Arama sonuçları hakkında 2-3 cümlelik bilgilendirici ve yardımcı bir yanıt ver.
                            Bulunan ürün sayısını ve çeşitliliğini vurgula.
                            """
        response = model.generate_content(results_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Yanıt oluşturma hatası: {e}")
        if products:
            return f"{len(products)} ürün bulundu. Size en uygun seçenekleri listeledim."
        else:
            return "Aradığınız kriterlere uygun ürün bulunamadı. Farklı anahtar kelimeler deneyebilirsiniz."

def predict_categories_hierarchical(title, top_k=1):
    if not MODEL_LOADED:
        return ["Elektronik", "Telefon", "Akıllı Telefon"], 0.0
    
    try:
        if not title or not isinstance(title, str):
            logger.warning(f"Geçersiz title: {title}")
            return ["Elektronik", "Telefon", "Akıllı Telefon"], 0.0
            
        title = title.strip()
        if not title:
            return ["Elektronik", "Telefon", "Akıllı Telefon"], 0.0
        
        logger.info(f"Kategori tahmini yapılıyor: '{title}'")
        
        payload = {
            "title": title,
            "topk": top_k
        }
        
        response = requests.post(
            f"{ML_SERVICE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            ana_pred = result.get("ana_kategori", "Elektronik")
            alt1_pred = result.get("alt_kategori_1", "Telefon") 
            alt2_pred = result.get("alt_kategori_2", "Akıllı Telefon")
            
            logger.info(f"Ana kategori tahmini: {ana_pred}")
            logger.info(f"Alt1 kategori tahmini: {alt1_pred}")
            logger.info(f"Alt2 kategori tahmini: {alt2_pred}")
            
            ana_confidence = result.get("ana_confidence", 0.5)
            alt1_confidence = result.get("alt1_confidence", 0.5)
            alt2_confidence = result.get("alt2_confidence", 0.5)
            overall_confidence = (ana_confidence + alt1_confidence + alt2_confidence) / 3
            
            categories = [ana_pred, alt1_pred, alt2_pred]
            
            logger.info(f"Tahmin tamamlandı: {categories}, confidence: {overall_confidence:.3f}")
            return categories, overall_confidence
            
        else:
            logger.error(f"Docker ML servisi hata kodu: {response.status_code}")
            logger.error(f"Hata detayı: {response.text}")
            return ["Elektronik", "Telefon", "Akıllı Telefon"], 0.0
            
    except requests.exceptions.Timeout:
        logger.error("Docker ML servisi timeout")
        return ["Elektronik", "Telefon", "Akıllı Telefon"], 0.0
    except requests.exceptions.ConnectionError:
        logger.error("Docker ML servisi bağlantı hatası")
        return ["Elektronik", "Telefon", "Akıllı Telefon"], 0.0
    except Exception as e:
        logger.error(f"Kategori tahmin hatası: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ["Elektronik", "Telefon", "Akıllı Telefon"], 0.0

def build_enhanced_query(query_text, product_info, categories=None):
    query_parts = []
    
    must_clauses = [
        {
            "multi_match": {
                "query": product_info.get("product_keywords", query_text),
                "fields": ["title^2", "marka^1.5"],
                "fuzziness": "AUTO"
            }
        }
    ]
    
    filter_clauses = []
    
    if categories and len(categories) >= 3:
        filter_clauses.extend([
            {"term": {"ana_kategori.keyword": categories[0]}},
            {"term": {"alt_kategori_1.keyword": categories[1]}},
            {"term": {"alt_kategori_2.keyword": categories[2]}}
        ])

    if product_info.get("brands"):
        brand_filters = []
        for brand in product_info["brands"]:
            brand_filters.append({"term": {"marka.keyword": brand}})
        if len(brand_filters) == 1:
            filter_clauses.append(brand_filters[0])
        else:
            filter_clauses.append({"bool": {"should": brand_filters, "minimum_should_match": 1}})

    if product_info.get("price_min") or product_info.get("price_max"):
        price_filter = {
            "script": {
                "script": {
                    "source": """
                        if (doc['price.keyword'].size() == 0) return false;
                        String priceStr = doc['price.keyword'].value;
                        if (priceStr == null || priceStr.isEmpty()) return false;
                        
                        // TL ve diğer karakterleri temizle
                        String cleanPrice = priceStr.replaceAll('[^0-9,\\.]', '');
                        cleanPrice = cleanPrice.replace(',', '.');
                        
                        if (cleanPrice.isEmpty()) return false;
                        
                        try {
                            double price = Double.parseDouble(cleanPrice);
                            boolean result = true;
                            
                            if (params.containsKey('min') && params.min != null) {
                                result = result && (price >= params.min);
                            }
                            if (params.containsKey('max') && params.max != null) {
                                result = result && (price <= params.max);
                            }
                            
                            return result;
                        } catch (NumberFormatException e) {
                            return false;
                        }
                    """,
                    "params": {}
                }
            }
        }

        if product_info.get("price_min"):
            price_filter["script"]["script"]["params"]["min"] = product_info["price_min"]
        if product_info.get("price_max"):
            price_filter["script"]["script"]["params"]["max"] = product_info["price_max"]
            
        filter_clauses.append(price_filter)

    if product_info.get("rating_min"):
        rating_filter = {
            "script": {
                "script": {
                    "source": """
                        if (doc['rating.keyword'].size() == 0) return false;
                        String ratingStr = doc['rating.keyword'].value;
                        if (ratingStr == null || ratingStr.isEmpty()) return false;
                        
                        try {
                            double rating = Double.parseDouble(ratingStr);
                            return rating >= params.min_rating;
                        } catch (NumberFormatException e) {
                            return false;
                        }
                    """,
                    "params": {
                        "min_rating": product_info["rating_min"]
                    }
                }
            }
        }
        filter_clauses.append(rating_filter)

    should_clauses = []
    
    if product_info.get("color"):
        should_clauses.append({
            "match": {
                "title": {
                    "query": product_info["color"],
                    "boost": 1.5
                }
            }
        })
    
    if product_info.get("memory"):
        should_clauses.append({
            "match": {
                "title": {
                    "query": product_info["memory"],
                    "boost": 1.5
                }
            }
        })

    bool_query = {
        "must": must_clauses
    }
    
    if filter_clauses:
        bool_query["filter"] = filter_clauses
    
    if should_clauses:
        bool_query["should"] = should_clauses
        bool_query["minimum_should_match"] = 0

    query = {
        "query": {
            "function_score": {
                "query": {
                    "bool": bool_query
                },
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "weighted_rating",
                            "factor": 0.1,
                            "modifier": "log1p",
                            "missing": 0.1
                        }
                    }
                ],
                "boost_mode": "sum",
                "score_mode": "sum"
            }
        },
        "sort": [
            "_score",
            {
                "weighted_rating": {
                    "order": "desc",
                    "missing": "_last"
                }
            }
        ],
        "size": 50
    }
    
    return query

def search_without_model(query_text):
    search_query = {
        "query": {
            "match": {
                "title": {
                    "query": query_text,
                    "fuzziness": "AUTO"
                }
            }
        }
    }
    return execute_search(search_query)

def search_with_model(query_text, categories, product_info=None):
    if len(categories) < 3:
        logger.warning(f"Yetersiz kategori sayısı: {len(categories)}")
        return search_without_model(query_text)
    
    valid_categories = [cat for cat in categories if cat and cat.strip()]
    if len(valid_categories) < 3:
        logger.warning("Boş kategoriler bulundu, basit aramaya geçiliyor")
        return search_without_model(query_text)
    
    if product_info:
        search_query = build_enhanced_query(query_text, product_info, categories)
    else:
        search_query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "multi_match": {
                                        "query": query_text,
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
                    "boost_mode": "multiply", 
                    "score_mode": "multiply",
                    "field_value_factor": {
                        "field": "weighted_rating",
                        "factor": 1,
                        "missing": 0.1
                    }
                }
            }
        }
    
    return execute_search(search_query)

def execute_search(search_query):
    try:
        url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_search"
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=search_query,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            products = []
            
            for hit in results.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                
                product = {
                    'ana_kategori': source.get('ana_kategori', ''),
                    'alt_kategori_1': source.get('alt_kategori_1', ''),
                    'alt_kategori_2': source.get('alt_kategori_2', ''),
                    'marka': source.get('marka', ''),
                    'title': source.get('title', ''),
                    'link': source.get('link', ''),
                    'price': source.get('price', ''),
                    'rating': source.get('rating', ''),
                    'comment_count': source.get('comment_count', ''),
                    'weighted_rating': source.get('weighted_rating', 0.0),
                    'score': hit.get('_score', 0)
                }
                products.append(product)
            
            return products, len(products)
        else:
            logger.error(f"OpenSearch hata kodu: {response.status_code}")
            return [], 0
            
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenSearch bağlantı hatası: {e}")
        return [], 0
    except Exception as e:
        logger.error(f"Arama hatası: {e}")
        return [], 0

@app.route('/')
def index():
    try:
        with open('project/templates/index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        logger.error("index.html dosyası bulunamadı")
        return """
        <html>
        <body>
        <h1>AI Product Search</h1>
        <p>index.html dosyası bulunamadı</p>
        </body>
        </html>
        """, 404

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Query parametresi gerekli'
            }), 400
        
        query_text = data['query'].strip()
        
        if not query_text:
            return jsonify({
                'success': False,
                'error': 'Boş sorgu'
            }), 400
        
        time.sleep(random.uniform(0.8, 1.5))
        
        product_info = extract_product_info(query_text)
        
        products = []
        search_method = "fallback"
        confidence = 0.0
        predicted_categories = []

        current_ml_status = check_ml_service()
        
        if current_ml_status:
            try:
                predicted_categories, confidence = predict_categories_hierarchical(
                    product_info.get("product_keywords", query_text), top_k=1
                )
                
                logger.info(f"Tahmin edilen kategoriler: {predicted_categories}, Güven: {confidence:.3f}")

                if confidence >= THRESHOLD and len(predicted_categories) >= 3:
                    products, total_count = search_with_model(query_text, predicted_categories, product_info)
                    search_method = "ai_enhanced"
                    logger.info(f"AI destekli arama kullanıldı: {total_count} sonuç")
                else:
                    products, total_count = search_without_model(query_text)
                    search_method = "simple"
                    logger.info(f"Basit arama kullanıldı (güven: {confidence:.3f}): {total_count} sonuç")

            except Exception as e:
                logger.error(f"AI arama hatası: {e}")
                products, total_count = search_without_model(query_text)
                search_method = "fallback_after_error"
        else:
            products, total_count = search_without_model(query_text)
            search_method = "no_model"
            logger.info("Model yüklü değil, basit arama kullanıldı")
        
        llm_response = generate_search_response(products, product_info, query_text)
        
        return jsonify({
            'success': True,
            'products': products,
            'query': query_text,
            'total_results': len(products),
            'search_method': search_method,
            'confidence': float(confidence),
            'predicted_categories': predicted_categories,
            'threshold': THRESHOLD,
            'model_available': current_ml_status,
            'product_info': product_info,
            'llm_response': llm_response
        })
        
    except Exception as e:
        logger.error(f"Arama endpoint hatası: {e}")
        return jsonify({
            'success': False,
            'error': f'Arama sırasında hata: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    try:
        opensearch_status = "disconnected"
        try:
            response = requests.get(f"{OPENSEARCH_URL}/_cluster/health", timeout=5)
            if response.status_code == 200:
                opensearch_status = "connected"
        except Exception as e:
            logger.warning(f"OpenSearch bağlantı kontrolü başarısız: {e}")
        
        ml_service_status = check_ml_service()
        
        gemini_status = "connected"
        try:
            test_response = model.generate_content("Test")
        except Exception as e:
            gemini_status = "disconnected"
            logger.warning(f"Gemini API bağlantı kontrolü başarısız: {e}")
        
        return jsonify({
            'status': 'healthy',
            'message': 'AI Product Search Service',
            'model_loaded': ml_service_status,
            'opensearch_status': opensearch_status,
            'gemini_status': gemini_status,
            'ml_service_url': ML_SERVICE_URL,
            'threshold': THRESHOLD
        })
    except Exception as e:
        logger.error(f"Health check hatası: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/test-model')
def test_model():
    ml_status = check_ml_service()
    if not ml_status:
        return jsonify({
            'success': False,
            'message': 'Model yüklü değil'
        })
    
    test_title = "Apple iPhone 14 Pro Max 256 GB 6.7 inç 5G Akıllı Telefon - Mor"
    
    try:
        predicted_categories, confidence = predict_categories_hierarchical(test_title, top_k=1)

        return jsonify({
            'success': True,
            'test_title': test_title,
            'predicted_categories': predicted_categories,
            'confidence': float(confidence),
            'threshold': THRESHOLD,
            'will_use_ai': confidence >= THRESHOLD
        })
    except Exception as e:
        logger.error(f"Model test hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print(f"Uygulama: http://localhost:5000")
    print(f"Arama API: http://localhost:5000/search")
    print(f"Sağlık: http://localhost:5000/health")
    print(f"Model Test: http://localhost:5000/test-model")
    print(f"AI Model: {'Docker BERT Aktif' if MODEL_LOADED else 'Model Erişilemez'}")
    print(f"Threshold: {THRESHOLD}")
    print(f"OpenSearch: {OPENSEARCH_URL}")
    print(f"Docker ML: {ML_SERVICE_URL}")
    print(f"Gemini API: {'Aktif' if api_key != 'your-gemini-api-key-here' else 'API Key Gerekli'}")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)