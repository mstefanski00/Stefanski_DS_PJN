import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = "data/culturax_pl_clean.jsonl" 
QDRANT_URL = "http://localhost:6333"
ES_URL = "http://localhost:9200"
COLLECTION_NAME = "culturax_semantic"
ES_INDEX = "culturax_fts"
BATCH_SIZE = 100

print("Łączenie z bazami.")
qdrant = QdrantClient(QDRANT_URL, timeout=60)
es = Elasticsearch(ES_URL, request_timeout=30)
model = SentenceTransformer("all-MiniLM-L6-v2")

def setup_databases():
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Qdrant: Utworzono kolekcję '{COLLECTION_NAME}'")

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    
    es.indices.create(index=ES_INDEX, body={
        "mappings": {
            "properties": {
                "text": {"type": "text"}
            }
        }
    })
    print(f"ES: Utworzono indeks '{ES_INDEX}'")

    print("Rozpoczynam indeksowanie danych.")
    
    points_batch = []
    es_actions = []
    
    with open(DATA_PATH, "r", encoding="utf-16") as f:
        lines = f.readlines()
        
        for i, line in enumerate(tqdm(lines, desc="Przetwarzanie")):
            doc = json.loads(line)
            text = doc.get("text", "")
            doc_id = doc.get("id", str(i))
            
            if not text:
                continue

            embedding = model.encode(text).tolist()
            points_batch.append({
                "id": i,
                "vector": embedding,
                "payload": {"text": text, "original_id": doc_id}
            })


            action = {
                "_index": ES_INDEX,
                "_id": doc_id,
                "_source": {"text": text}
            }
            es_actions.append(action)

            if len(points_batch) >= BATCH_SIZE:
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_batch
                )
                helpers.bulk(es, es_actions)
                
                points_batch = []
                es_actions = []

    if points_batch:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points_batch
        )
    
    print(f"Załadowano {len(points_batch)} dokumentów.")

if __name__ == "__main__":
    setup_databases()