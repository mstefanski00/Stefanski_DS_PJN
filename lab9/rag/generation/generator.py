import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma2:2b" 

def generate_answer(query, context_docs):
    if not context_docs:
        return "Niestety, nie znalazłem żadnych dokumentów na ten temat."

    context_text = "\n\n".join([f"Dokument {i+1}: \n{doc['text']}" for i, doc in enumerate(context_docs)])
    
    prompt = f"""
    Jesteś pomocnym asystentem AI.
    Odpowiedz na pytanie użytkownika WYŁĄCZNIE na podstawie dostarczonych poniżej dokumentów.
    
    DOKUMENTY:
    {context_text}
    
    PYTANIE UŻYTKOWNIKA: {query}
    
    INSTRUKCJA:
    - Jeśli odpowiedź nie znajduje się w dokumentach, napisz "Nie wiem".
    - Nie wymyślaj faktów.
    - Odpowiadaj zwięźle i konkretnie po polsku.
    """
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        return resp.json().get('message', {}).get('content', '').strip()
    except Exception as e:
        return f"Błąd generowania: {e}"