import json
import requests
import re
from rag.retrieval.search_engine import hybrid_search
from rag.reasoning.memory import KnowledgeMemory

class ResearchAgent:

    def __init__(self, llm_model="gemma2:2b", llm_url="http://localhost:11434/api/chat"):
        self.memory = KnowledgeMemory()
        self.llm_model = llm_model
        self.llm_url = llm_url

    def _call_llm(self, prompt):
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            response = requests.post(self.llm_url, json=payload, timeout=60)
            return response.json()['message']['content']
        except Exception as e:
            print(f"Błąd LLM: {e}")
            return ""

    def _analyze_intent(self, query):
        prompt = f"""
        Wyodrębnij z tekstu kluczowe metadane do wyszukiwania:
        - locations: miasta, kraje
        - organizations: firmy, instytucje
        
        Zwróć TYLKO JSON: {{ "locations": [], "organizations": [] }}
        Tekst: {query}
        """
        response = self._call_llm(prompt)
        try:
            clean_json = re.sub(r'```json|```', '', response).strip()
            extraction = json.loads(clean_json)
        except:
            extraction = {"locations": [], "organizations": []}

        years = [int(y) for y in re.findall(r'\b(20\d{2})\b', query)]
        
        return {
            "entities": extraction.get("locations", []) + extraction.get("organizations", []),
            "years": years
        }

    def _verify_relevance(self, text, query):
        prompt = f"""
        Czy poniższy tekst zawiera odpowiedź na pytanie użytkownika (zwróć uwagę na daty)?
        Pytanie: "{query}"
        Tekst: "{text}"
        
        Zwróć TYLKO JSON: {{ "is_relevant": true/false }}
        """
        response = self._call_llm(prompt)
        try:
            clean_json = re.sub(r'```json|```', '', response).strip()
            return json.loads(clean_json).get('is_relevant', True)
        except:
            return True

    def solve(self, query):
        print(f"\nRozpoczynam proces dla: '{query}'")
        
        meta = self._analyze_intent(query)
        filters = {}
        if meta['years']: filters['years'] = meta['years']
        if meta['entities']: filters['named_entities'] = meta['entities']
        
        print(f"   Filtry: {filters}")
        
        docs = hybrid_search(query, filters=filters, limit=5)
        print(f"   Znaleziono {len(docs)} kandydatów.")
        
        if not docs:
            print("   Brak danych. Zapisuję w pamięci.")
            self.memory.add_pending(query, filters)
            return "Nie posiadam obecnie tych informacji. Zapisałem Twoje pytanie i powiadomię Cię, gdy pojawią się nowe dane."

        verified_docs = []
        for doc in docs[:3]: 
            if self._verify_relevance(doc['text'], query):
                verified_docs.append(doc)
            else:
                print(f"   Odrzucono dokument o ID {doc.get('id')} - Niepasujący kontekst.")
        
        if not verified_docs:
            print("   Dokumenty odrzucone przez LLM.")
            self.memory.add_pending(query, filters)
            return "Znalazłem dokumenty, ale nie odpowiadają one precyzyjnie na Twoje pytanie."

        print(f"   Tworzę odpowiedź z {len(verified_docs)} źródeł.")
        context = "\n".join([f"- {d['text']}" for d in verified_docs])
        
        final_prompt = f"""
        Odpowiedz na pytanie na podstawie notatek.
        Pytanie: {query}
        Notatki:
        {context}
        Odpowiedź (krótka i konkretna):
        """
        return self._call_llm(final_prompt)