import json
import os
from rag.metadata.ner import NERExtractor
from rag.metadata.dates import extract_dates_and_years

INPUT_FILE = "data/culturax_pl_clean.jsonl"
OUTPUT_FILE = "data/culturax_enriched.jsonl"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Brak pliku: {INPUT_FILE}")
        return

    ner_model = NERExtractor()

    print(f"Przetwarzanie: {INPUT_FILE} => {OUTPUT_FILE}.")
    
    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-16') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-16') as fout:
        
        for line in fin:
            if not line.strip(): continue
            
            doc = json.loads(line)
            text = doc.get('text', '')

            entities = ner_model.extract(text)
            date_info = extract_dates_and_years(text)

            doc['named_entities'] = entities
            doc['years'] = date_info['years']
            doc['extracted_dates'] = date_info['dates']
            
            fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
            count += 1
            
            if count % 2 == 0:
                print(f"   Przetworzono {count} dokumentów.", end='\r')

    print(f"\nZakończono! Przetworzono {count} dokumentów.")
    print("Przykładowy wzbogacony rekord:")
    
    print(json.dumps(doc, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()