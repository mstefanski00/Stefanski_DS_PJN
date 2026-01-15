from rag.reasoning.agent_brain import extract_search_params
from rag.retrieval.search_engine import hybrid_search

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_agent(user_query):
    print(f"\n{CYAN}UŻYTKOWNIK: {user_query}{RESET}")
    
    print(f"{YELLOW}AGENT: Analiza{RESET}. Przetwarzam intencję i metadane.")
    analyzed = extract_search_params(user_query)
    
    filters = analyzed['filters']
    search_query = analyzed['text']
    
    print(f"   -> Interpretowane pytanie: '{search_query}'")
    print(f"   -> Wykryte filtry: Lata={filters.get('years')}, Encje={filters.get('named_entities')}")
    
    print(f"{YELLOW}AGENT: Retrieval{RESET}. Wyszukuję w bazie hybrydowej.")
    docs = hybrid_search(search_query, filters=filters, limit=3)
    
    print(f"   -> Znaleziono {len(docs)} dokumentów.")
    
    if docs:
        print(f"{GREEN} WYNIKI: {RESET}:")
        for i, doc in enumerate(docs):
            meta_info = f"[Lata: {doc.get('years')} | Encje: {doc.get('named_entities')}]"
            snippet = doc.get('text', '')[:100].replace('\n', ' ')
            print(f"   {i+1}. {meta_info} {snippet}...")
            
        print(f"\n ODPOWIEDŹ SYSTEMU: Tutaj LLM wygenerowałby odpowiedź na podstawie powyższych {len(docs)} fragmentów.")
    else:
        print(f"\n ODPOWIEDŹ SYSTEMU: Nie znalazłem informacji spełniających kryteria.")

if __name__ == "__main__":

    run_agent("Jakie są metody zarządzania ludźmi?")
    
    run_agent("Co słychać w Polsce?")
    
    run_agent("Wydarzenia z 2020 roku")
    
    run_agent("Technologia w Polsce w 2018 roku")