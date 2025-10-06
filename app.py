from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from qdrant_client import QdrantClient
import os

app = Flask(__name__)
CORS(app)  # Zezwól na zapytania z frontendu

# Konfiguracja z zmiennych środowiskowych
QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
COLLECTION_NAME = "prezentacje_biochemia"

# Inicjalizacja klientów
genai.configure(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "message": "RAG API działa!",
        "endpoints": ["/search"]
    })

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Brak pytania"}), 400
        
        # 1. Utwórz embedding z Gemini
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = result['embedding']
        
        # 2. Wyszukaj w Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=5
        )
        
        # 3. Formatuj wyniki
        formatted_results = []
        for hit in search_results:
            formatted_results.append({
                "score": float(hit.score),
                "presentation": hit.payload.get("presentation", ""),
                "slide_id": hit.payload.get("slide_id", ""),
                "content": hit.payload.get("content", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "total_chunks": hit.payload.get("total_chunks", 1)
            })
        
        return jsonify({
            "success": True,
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)