from rag.retrieval.search_engine import hybrid_search

def print_results(header, results):
    print(f"\n{header} Znaleziono: {len(results)}")
    for i, doc in enumerate(results[:3]):
        text_snippet = doc.get('text', '')[:80].replace('\n', ' ')
        entities = doc.get('named_entities', [])
        years = doc.get('years', [])
        print(f"   {i+1}. [{years} | {entities}] {text_snippet}...")

results_raw = hybrid_search("Co słychać w Polsce?", limit=5)
print_results("Test 1: Zwykłe wyszukiwanie", results_raw)

filters_entity = {"named_entities": ["Polsce", "Warszawa"]}
results_entity = hybrid_search("Co słychać?", filters=filters_entity, limit=5)
print_results("Test 2: Filtr Encji (Polsce/Warszawa)", results_entity)

filters_year = {"years": [2000, 2001, 2010, 2020, 2021, 2022, 2023, 2024]}
results_year = hybrid_search("technologia", filters=filters_year, limit=5)
print_results("Test 3: Filtr Lat 2000+", results_year)