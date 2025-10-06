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

def is_question_slide(content):
    """Sprawdza czy slajd zawiera głównie pytania aktywizujące"""
    clean_content = clean_html(content).lower()
    
    # Heurystyki dla slajdów z pytaniami
    question_markers = ['?', 'pytani', 'zastanów się', 'rozgrzewka', 'aktywizuj']
    question_count = sum(1 for marker in question_markers if marker in clean_content)
    
    # Jeśli jest wiele pytań i mało treści merytorycznej
    if '?' in clean_content:
        question_marks = clean_content.count('?')
        words = len(clean_content.split())
        # Jeśli więcej niż 2 pytania i mało słów - prawdopodobnie slajd aktywizujący
        if question_marks >= 2 and words < 200:
            return True
    
    # Sprawdź charakterystyczne frazy
    if any(phrase in clean_content for phrase in ['czas na rozgrzewkę', 'pytania aktywizujące', 'zastanów się nad']):
        return True
    
    return False

def generate_answer(query, search_results):
    """Generuje naturalną odpowiedź używając Gemini"""
    
    # Filtruj slajdy z pytaniami i przygotuj kontekst
    context_parts = []
    sources = []
    
    # Użyj top 7-10 wyników, pomijając slajdy z pytaniami
    for i, result in enumerate(search_results[:15], 1):  # Sprawdź więcej wyników
        payload = result.payload
        content = payload.get("content", "")
        
        # Pomiń slajdy z pytaniami aktywizującymi
        if is_question_slide(content):
            continue
        
        clean_content = clean_html(content)
        
        # Dodaj tylko jeśli ma sens (nie puste)
        if len(clean_content.strip()) > 50:
            context_parts.append(f"Fragment {len(context_parts)+1} (z prezentacji '{payload.get('presentation', '')}', Slajd ID: {payload.get('slide_id', '')}):\n{clean_content}")
            sources.append({
                "presentation": payload.get("presentation", ""),
                "slide_id": payload.get("slide_id", ""),
                "score": float(result.score)
            })
        
        # Zbierz do 8 merytorycznych fragmentów
        if len(context_parts) >= 8:
            break
    
    if not context_parts:
        return {
            "answer": "Nie znalazłem merytorycznych informacji na ten temat w dostępnych materiałach.",
            "sources": [],
            "model_used": None
        }
    
    context = "\n\n".join(context_parts)
    
    # Prompt dla Gemini - poziom medyczny, bardzo szczegółowy
    prompt = f"""Jesteś ekspertem biochemii klinicznej, który przygotowuje materiały edukacyjne dla lekarzy i studentów medycyny.

Pytanie: {query}

Dostępne fragmenty z materiałów dydaktycznych:
{context}

WYMAGANIA DLA ODPOWIEDZI (POZIOM MEDYCZNY):

1. ODBIORCA: Lekarze i studenci medycyny - używaj pełnej terminologii medycznej i biochemicznej
2. SZCZEGÓŁOWOŚĆ: Odpowiedź musi zawierać:
   - Pełne nazwy chemiczne i enzymatyczne (np. γ-glutamylotranspeptydaza, a nie tylko GGTP)
   - Konkretne mechanizmy biochemiczne i ścieżki metaboliczne
   - Wartości referencyjne, jeśli są w materiałach
   - Lokalizacje subkomórkowe procesów
   - Produkty i substraty reakcji
   - Znaczenie kliniczne i diagnostyczne

3. STRUKTURA: 
   - Rozpocznij od definicji lub kontekstu biochemicznego
   - Przedstaw szczegółowy mechanizm/proces
   - Opisz zastosowania kliniczne lub znaczenie patofizjologiczne
   - Długość: 6-12 zdań (w zależności od złożoności tematu)

4. INTEGRACJA WIEDZY:
   - Połącz informacje ze WSZYSTKICH fragmentów w spójną całość
   - Zachowaj chronologię procesów biochemicznych
   - Wyjaśnij związki przyczynowo-skutkowe

5. PRECYZJA:
   - Używaj dokładnych nazw zgodnie z nomenklaturą biochemiczną
   - Cytuj konkretne enzymy, koenzymy, metabolity
   - Zachowaj wszystkie szczegóły techniczne z materiałów
   - Jeśli materiały zawierają skróty - użyj pełnych nazw przy pierwszym wystąpieniu

6. STYL: Merytoryczny, akademicki, bez upraszczania. Pisz akapitami z logicznym przepływem informacji.

7. OGRANICZENIA: Bazuj WYŁĄCZNIE na dostarczonych fragmentach. NIE dodawaj informacji o źródłach.

Odpowiedź (poziom medyczny, szczegółowa, merytoryczna):"""

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
