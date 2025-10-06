from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from qdrant_client import QdrantClient
import os
import re

app = Flask(__name__)
CORS(app)

# Konfiguracja z zmiennych środowiskowych
QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
COLLECTION_NAME = "prezentacje_biochemia"

# Inicjalizacja klientów
genai.configure(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def clean_html(text):
    """Usuwa znaczniki HTML z tekstu"""
    # Usuń tagi HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Usuń nadmiarowe białe znaki
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_answer(query, search_results):
    """Generuje naturalną odpowiedź używając Gemini"""
    
    # Przygotuj kontekst z wyników wyszukiwania
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results[:3], 1):  # Top 3 wyniki
        payload = result.payload
        clean_content = clean_html(payload.get("content", ""))
        
        context_parts.append(f"Fragment {i}: {clean_content}")
        sources.append({
            "presentation": payload.get("presentation", ""),
            "slide_id": payload.get("slide_id", ""),
            "score": float(result.score)
        })
    
    context = "\n\n".join(context_parts)
    
    # Prompt dla Gemini
    prompt = f"""Jesteś asystentem AI, który odpowiada na pytania na podstawie materiałów szkoleniowych z biochemii.

Pytanie użytkownika: {query}

Dostępne fragmenty z materiałów:
{context}

INSTRUKCJE:
1. Odpowiedz na pytanie w naturalny, zrozumiały sposób
2. Bazuj TYLKO na informacjach z powyższych fragmentów
3. Jeśli fragmenty nie zawierają odpowiedzi, powiedz to wprost
4. Odpowiedź ma być zwięzła (2-4 zdania) ale merytoryczna
5. NIE dodawaj na końcu informacji o źródłach - zostaną dodane automatycznie

Odpowiedź:"""

    # Próbuj najpierw z gemini-2.5-pro, potem fallback na 2.5-flash
    models_to_try = ['gemini-2.5-pro', 'gemini-2.5-flash']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            answer = response.text
            
            return {
                "answer": answer,
                "sources": sources,
                "model_used": model_name
            }
        except Exception as e:
            # Jeśli to nie był ostatni model, próbuj dalej
            if model_name != models_to_try[-1]:
                continue
            # Jeśli to był ostatni model, zwróć błąd
            return {
                "answer": f"Przepraszam, wystąpił błąd podczas generowania odpowiedzi: {str(e)}",
                "sources": sources,
                "model_used": None
            }

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
        
        if not search_results:
            return jsonify({
                "success": True,
                "answer": "Nie znalazłem informacji na ten temat w dostępnych materiałach.",
                "sources": []
            })
        
        # 3. Wygeneruj naturalną odpowiedź
        result = generate_answer(query, search_results)
        
        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"]
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
