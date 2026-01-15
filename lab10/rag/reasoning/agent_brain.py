import json
import re
import requests
from datetime import datetime

MODEL_NAME = "gemma2:2b" 
OLLAMA_URL = "http://localhost:11434/api/chat"

def extract_search_params(user_query):

    system_prompt = """
    Jesteś analitykiem zapytań do wyszukiwarki.
    Twoim zadaniem jest przeanalizowanie pytania i wyodrębnienie:
    1. 'search_query': Główny temat pytania - słowa kluczowe.
    2. 'years': Lista lat w integerach wspomnianych w pytaniu.
    3. 'entities': Lista nazw własnych - miejsca, organizacje, osoby - ważnych dla kontekstu.
    
    Zwróć TYLKO czysty kod JSON w formacie:
    {
      "search_query": "tekst",
      "years": [2020],
      "entities": ["Warszawa"]
    }
    Nie dodawaj żadnych komentarzy ani znaczników markdown.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response_data = response.json()
        
        if 'error' in response_data:
            print(f"Błąd API Ollama: {response_data['error']}")
            return _fallback(user_query)

        content = response_data.get('message', {}).get('content', '')
        
        clean_json = re.sub(r'```json|```', '', content).strip()
        
        parsed = json.loads(clean_json)
        
        return {
            "text": parsed.get("search_query", user_query),
            "filters": {
                "years": parsed.get("years", []),
                "named_entities": parsed.get("entities", [])
            }
        }
        
    except Exception as e:
        print(f"Błąd parsowania agenta: {e}")
        print(f"   Otrzymana treść: {content if 'content' in locals() else 'Brak'}")
        return _fallback(user_query)

def _fallback(query):
    return {
        "text": query,
        "filters": {"years": [], "named_entities": []}
    }