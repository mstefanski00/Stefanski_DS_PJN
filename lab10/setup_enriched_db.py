import json
import os
from elasticsearch import Elasticsearch, helpers
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ES_HOST = "http://localhost:9200"
QDRANT_HOST = "http://localhost:6333"
DATA_FILE = "data/culturax_enriched.jsonl"

INDEX_NAME_ES = "lab10_hybrid"
COLLECTION_NAME_QDRANT = "lab10_hybrid"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def setup_databases():
    print("Łączenie z bazami danych.")
    es = Elasticsearch(ES_HOST, basic_auth=("elastic", "changeme"))
    qdrant = QdrantClient(url=QDRANT_HOST)
    
    if es.indices.exists(index=INDEX_NAME_ES):
        es.indices.delete(index=INDEX_NAME_ES)
    
    if qdrant.collection_exists(COLLECTION_NAME_QDRANT):
        qdrant.delete_collection(COLLECTION_NAME_QDRANT)

    es_mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "named_entities": {"type": "keyword"}, 
                "years": {"type": "integer"},
                "id": {"type": "keyword"}
            }
        }
    }
    es.indices.create(index=INDEX_NAME_ES, body=es_mapping)
    print("Utworzono indeks ES z metadanymi.")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME_QDRANT,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("Utworzono kolekcję Qdrant.")

    return es, qdrant

def load_data(es, qdrant):
    if not os.path.exists(DATA_FILE):
        print(f"Brak pliku: {DATA_FILE}. Uruchom najpierw enrich_data.py.")
        return

    print(f"Ładowanie modelu do embeddingów: {EMBEDDING_MODEL}.")
    model = SentenceTransformer(EMBEDDING_MODEL)

    batch_es = []
    batch_qdrant = []
    batch_size = 100
    
    print("Rozpoczynam indeksowanie.")
    
    with open(DATA_FILE, 'r', encoding='utf-16') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        
        for i, line in tqdm(enumerate(f), total=total_lines, desc="Indeksowanie"):
            doc = json.loads(line)
            text = doc.get("text", "")
            doc_id = doc.get("id", str(i))
            
            clean_entities = [e for e in doc.get("named_entities", []) if len(e) < 50]
            clean_years = doc.get("years", [])

            es_doc = {
                "_index": INDEX_NAME_ES,
                "_id": doc_id,
                "_source": {
                    "text": text,
                    "named_entities": clean_entities,
                    "years": clean_years
                }
            }
            batch_es.append(es_doc)

            embedding = model.encode(text).tolist()
            payload = {
                "text": text,
                "named_entities": clean_entities,
                "years": clean_years
            }
            batch_qdrant.append(PointStruct(id=i, vector=embedding, payload=payload))

            if len(batch_es) >= batch_size:
                helpers.bulk(es, batch_es)
                qdrant.upsert(collection_name=COLLECTION_NAME_QDRANT, points=batch_qdrant)
                batch_es = []
                batch_qdrant = []

        # Końcówka
        if batch_es:
            helpers.bulk(es, batch_es)
            qdrant.upsert(collection_name=COLLECTION_NAME_QDRANT, points=batch_qdrant)

    print("\nSukces. Dane załadowane do Elasticsearch i Qdrant.")

if __name__ == "__main__":
    es_client, qdrant_client = setup_databases()
    load_data(es_client, qdrant_client)