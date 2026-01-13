from rag.retrieval.search_engine import hybrid_search
from rag.reasoning.query_analysis import analyze_query_intent, generate_clarification_question
from rag.reasoning.filtering import filter_retrieved_docs
from rag.generation.generator import generate_answer
from rag.verification.validator import validate_rag_answer, safe_mode_retry

def run_rag_pipeline(user_query):
    print(f"\nZaczynamy! Pytanie użytkownika: '{user_query}'")
    
    # clarification = generate_clarification_question(user_query)
    # if clarification.get("is_ambiguous", False):
    #     print(f"Niejednoznaczność.")
    #     interpretations = "\n".join([f"- {i}" for i in clarification.get("interpretations", [])])
    #     return f"Twoje pytanie jest niejednoznaczne. Czy chodziło Ci o:\n{interpretations}"
    clarification = generate_clarification_question(user_query)
    if clarification.get("is_ambiguous", False):
        print(f"Wykryto niejednoznaczność, ale kontynuuję na potrzeby testu.")
        print(f"   -> System zapytałby: {clarification.get('interpretations', [''])[0]}...")

    intent = analyze_query_intent(user_query)
    print(f"Rozpoczynam analizę intencji. Tryb: {intent['mode'].upper()}. Wagi -> ES: {intent['es_weight']}, Qdrant: {intent['qdrant_weight']}")

    _, _, raw_docs = hybrid_search(
        user_query, 
        limit=10, 
        weight_es=intent['es_weight'], 
        weight_qdrant=intent['qdrant_weight']
    )
    
    context_docs, stats = filter_retrieved_docs(raw_docs, min_words=15, max_docs=4)
    print(f"Rozpoczynam filtrowanie. Wybrano: {len(context_docs)} dokumentów. Odrzucono krótkie: {stats['rejected_short']}")
    
    if not context_docs:
        return safe_mode_retry(user_query, strategy="memory")

    print("Generowanie odpowiedzi.")
    draft_answer = generate_answer(user_query, context_docs)
    print(f"   -> Wstępny draft: {draft_answer[:100]}...")
    
    print("Weryfikacja faktów.")
    is_valid, reason = validate_rag_answer(user_query, draft_answer, context_docs)
    
    if is_valid:
        print(f"Zweryfikowano pomyślnie.")
        return draft_answer
    else:
        print(f"Halucynacja modelu! Odrzucono: {reason}")
        return safe_mode_retry(user_query, strategy="memory")
    
# if __name__ == "__main__":
#     print("="*60)
#     print("Test 1: Pytanie poprawne.")
#     response_1 = run_rag_pipeline("Jak poprawić pracę zespołową?")
#     print(f"\nOdpowiedź finalna:\n{response_1}\n")
    
#     print("="*60)
#     print("Test 2: Celowa halucynacja.")
#     response_2 = run_rag_pipeline("Jaki jest ulubiony kolor prezesa PAN?")
#     print(f"\nOdpowiedź finalna:\n{response_2}\n")