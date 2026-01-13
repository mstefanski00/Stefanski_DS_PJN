import json
import requests
import re
from rag.memory.memory_manager import memory

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

def clean_json_text(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text

def query_ollama_json(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "temperature": 0.0
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        raw_text = resp.json().get('response', '')

        clean_text = clean_json_text(raw_text)
        
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            #print(f"LLM nie zwrócił poprawnego JSON. Tekst: '{raw_text[:50]}...'")
            #print("RAW LLM OUTPUT:\n", raw_text)
            
            text_lower = raw_text.lower()
            if "true" in text_lower and "false" not in text_lower:
                return {"is_valid": True, "reason": "Wykryto słowo 'true'."}
            elif "false" in text_lower:
                return {"is_valid": False, "reason": "Wykryto słowo 'false'."}
            else:
                return {"is_valid": False, "reason": "LLM output unreadable"}

    except Exception as e:
        print(f"Błąd walidacji LLM: {e}")
        return {"is_valid": False, "reason": "LLM Error"}

def validate_rag_answer(user_query, generated_answer, context_docs):

    context_text = "\n".join([f"- {doc.get('text', '')[:300]}..." for doc in context_docs])
    
    prompt = f"""
    Jesteś bezstronnym sędzią sprawdzającym fakty.
    
    PYTANIE UŻYTKOWNIKA: "{user_query}"
    
    ODPOWIEDŹ SYSTEMU: "{generated_answer}"
    
    DOSTĘPNE ŹRÓDŁA:
    {context_text}
    
    ZADANIE:
    Oceń, czy odpowiedź systemu jest w pełni poparta przez dostępne źródła.
    Jeśli odpowiedź zawiera informacje, których NIE MA w źródłach, wtedy zwróć fałsz.
    Jeśli odpowiedź jest poprawna, wówczas zwróć prawdę.
    
    Zwróć JSON:
    {{
        "is_valid": true,
        "reason": "Krótkie uzasadnienie decyzji"
    }}
    """
    # prompt = f"""
    # ZWRÓĆ WYŁĄCZNIE JEDEN OBIEKT JSON.
    # NIE DODAWAJ ŻADNEGO INNEGO TEKSTU.

    # FORMAT:
    # {{"is_valid": false, "reason": "Format error"}}

    # PYTANIE: "{user_query}"
    # ODPOWIEDŹ: "{generated_answer}"
    # ŹRÓDŁA:
    # {context_text}

    # Czy odpowiedź jest w pełni poparta źródłami?
    # """
    
    result = query_ollama_json(prompt)
    return result.get("is_valid", False), result.get("reason", "Brak uzasadnienia")

def safe_mode_retry(query, strategy="memory"):

    if strategy == "memory":
        memory.add_pending_query(query, reason="Failed validation")
        return "Nie jestem pewien odpowiedzi na podstawie dostępnych dokumentów. Pytanie zostało zapisane do wyjaśnienia przez eksperta."
    
    return "Brak danych - safe mode"