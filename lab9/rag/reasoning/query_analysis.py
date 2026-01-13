import re
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"  

def analyze_query_intent(query: str):

    has_numbers = bool(re.search(r'\d+', query))
    
    words = query.split()
    has_acronyms = False
    
    if len(words) > 0:
        for i, w in enumerate(words):
            clean_w = w.strip(".,?!:;")
            if clean_w.isupper() and len(clean_w) >= 2:
                if i == 0 and len(clean_w) < 3:
                    continue
                has_acronyms = True
                break

    if has_numbers or has_acronyms:
        return {
            "mode": "factual",
            "es_weight": 0.8,
            "qdrant_weight": 0.2
        }
    else:
        return {
            "mode": "semantic",
            "es_weight": 0.4,
            "qdrant_weight": 0.6
        }

def query_ollama(prompt, model=MODEL_NAME):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.2,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get('response', 'Brak odpowiedzi z modelu.')
    except Exception as e:
        return f"Błąd połączenia z Ollama: {e}"
    

def decompose_query(user_input: str):

    prompt = f"""
    Jesteś asystentem AI. Przeanalizuj poniższe pytanie użytkownika.
    Twoim zadaniem jest rozbić je na listę prostszych pytań pomocniczych, 
    które pomogą znaleźć pełną odpowiedź.
    
    Zwróć wynik WYŁĄCZNIE jako obiekt JSON w formacie:
    {{
      "main_question": "{user_input}",
      "sub_questions": ["pytanie 1", "pytanie 2"]
    }}
    
    PYTANIE UŻYTKOWNIKA: {user_input}
    """
    
    response_text = query_ollama(prompt)
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print("Błąd parsowania JSON z LLM. Zwracam oryginał.")
        return {
            "main_question": user_input,
            "sub_questions": [user_input]
        }

def generate_clarification_question(user_input: str):

    prompt = f"""
    Przeanalizuj pytanie: "{user_input}".
    Czy jest ono niejednoznaczne? Jeśli tak, zaproponuj 2 możliwe interpretacje.
    
    Zwróć wynik WYŁĄCZNIE jako JSON:
    {{
       "is_ambiguous": true/false,
       "interpretations": ["interpretacja 1", "interpretacja 2"]
    }}
    """
    
    response_text = query_ollama(prompt)
    try:
        return json.loads(response_text)
    except:
        return {"is_ambiguous": False, "interpretations": []}
    
def clean_json_text(text):

    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        return text[start_idx : end_idx + 1]
    
    return text