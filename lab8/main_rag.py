import requests
import json
from rag_modules.hybrid_search import dynamic_hybrid_search
from rag_modules.text_processing import chunk_document

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"  



def query_ollama(prompt, model=MODEL_NAME):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get('response', 'Brak odpowiedzi z modelu.')
    except Exception as e:
        return f"Błąd połączenia z Ollama: {e}"

def rag_query(user_input, top_k=3, prompt_mode="B", verbose=True):

    if verbose:
        print(f"Szukam informacji dla: '{user_input}'.")

    _, _, hybrid_results, strategy, confidence = dynamic_hybrid_search(user_input, limit=top_k)
    
    if not hybrid_results:
        return {
            "query": user_input,
            "answer": "Nie znalazłem żadnych dokumentów w bazie spełniających kryteria.",
            "sources": [],
            "strategy": strategy
        }

    context_text = ""
    used_sources = []
    
    for i, doc in enumerate(hybrid_results, 1):
        chunks = chunk_document(doc['text'], word_limit=150)
        
        best_fragment = chunks[0] if chunks else ""
        
        context_text += f"\n[Źródło {i}] (Score: {doc['score']:.2f})\n{best_fragment}\n"
        
        used_sources.append({
            "id": doc['id'],
            "fragment": best_fragment,
            "score": doc['score']
        })

    # system_prompt = f"""
    # Jesteś pomocnym asystentem AI. Odpowiedz na pytanie użytkownika WYŁĄCZNIE na podstawie poniższych fragmentów tekstu.
    
    # ZASADY:
    # 1. Jeśli odpowiedź nie znajduje się w dostarczonym tekście, napisz: "BRAK INFORMACJI W DOKUMENTACH".
    # 2. Nie wymyślaj faktów. Opieraj się tylko na źródłach.
    # 3. Po każdej informacji podaj numer źródła w nawiasie, np. [Źródło 1].
    
    # Kontekst:
    # {context_text}
    
    # Pytanie użytkownika:
    # {user_input}
    
    # Odpowiedź:
    # """
    if prompt_mode == "A":
        system_prompt = f"""
        Odpowiedz na pytanie, używając poniższych fragmentów.
        
        Fragmenty:
        {context_text}
        
        Pytanie:
        {user_input}
        """
    else:

        system_prompt = f"""
        Jesteś pomocnym asystentem. Twoim zadaniem jest odpowiedzieć na pytanie, wykorzystując poniższe fragmenty tekstu.
        
        KONTEKST:
        {context_text}
        
        INSTRUKCJE:
        1. Przeanalizuj kontekst. Jeśli znajdziesz w nim odpowiedź, napisz ją własnymi słowami po polsku.
        2. Na końcu zdania dodaj numer źródła, np. [1].
        3. Jeśli kontekst nie pasuje do pytania, napisz: "Nie znalazłem informacji w bazie".
        
        PYTANIE: {user_input}
        ODPOWIEDŹ:
        """

    if verbose:
        print(f"Generuję odpowiedź za pomocą modelu {MODEL_NAME}.")
    
    ai_answer = query_ollama(system_prompt)
    
    return {
        "query": user_input,
        "answer": ai_answer,
        "sources": used_sources,
        "strategy": strategy,
        "confidence": confidence
    }