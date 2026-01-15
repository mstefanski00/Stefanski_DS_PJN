import json

FILE = "data/culturax_enriched.jsonl"

print(f"Szukam ciekawych rekordów w {FILE}...\n")

found = 0
with open(FILE, 'r', encoding='utf-16') as f:
    for line in f:
        doc = json.loads(line)
        
        if doc['named_entities'] or doc['years']:
            print(f"Ddokument {doc['id']}")
            print(f"Tekst: {doc['text'][:100]}...")
            print(f"Encje: {doc['named_entities']}")
            print(f"Lata:  {doc['years']}")
            print("-" * 30)
            
            found += 1
            if found >= 5: 
                break

if found == 0:
    print("Nie znaleziono żadnych encji w całym pliku. Coś poszło nie tak.")