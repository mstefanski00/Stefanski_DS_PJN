import json

class KnowledgeMemory:
    
    def __init__(self):
        self.pending_queries = []

    def add_pending(self, query, filters):

        record = {
            "query": query,
            "entities_hint": filters.get('named_entities', []),
            "years_hint": filters.get('years', []),
            "attempts": 0,
            "status": "waiting_for_data"
        }
        self.pending_queries.append(record)
        print(f"Zapisano intencję: '{query}' Czekam na dane: {filters}")

    def check_new_document(self, new_doc):

        doc_snippet = new_doc.get('text', '')[:40].replace('\n', ' ')
        print(f"Nowy dokument: '{doc_snippet}.' [Rok: {new_doc.get('years')}]")
        
        triggered_queries = []
        
        for q in self.pending_queries:
            if q['status'] != "waiting_for_data":
                continue
            
            year_match = False
            doc_years = new_doc.get('years', [])
            
            if not q['years_hint'] or any(y in q['years_hint'] for y in doc_years):
                year_match = True
                
            entity_match = False
            doc_ents = new_doc.get('named_entities', [])
            
            if not q['entities_hint'] or any(e in doc_ents for e in q['entities_hint']):
                entity_match = True
            
            if year_match and entity_match:
                q['status'] = "retry_ready"
                triggered_queries.append(q)
                print(f"   Dokument pasuje do oczekującego pytania: '{q['query']}'.")

        return triggered_queries