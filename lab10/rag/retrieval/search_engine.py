from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range
from sentence_transformers import SentenceTransformer

ES_HOST = "http://localhost:9200"
QDRANT_HOST = "http://localhost:6333"
INDEX_NAME_ES = "lab10_hybrid"
COLLECTION_NAME_QDRANT = "lab10_hybrid"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

es = Elasticsearch(ES_HOST, basic_auth=("elastic", "changeme"))
qdrant = QdrantClient(url=QDRANT_HOST)
model = SentenceTransformer(EMBEDDING_MODEL)

def build_es_query(query_text, filters=None):

    es_query = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["text"],
                        "type": "best_fields"
                    }
                }
            ],
            "filter": []
        }
    }

    if filters:
        if "years" in filters and filters["years"]:
            min_year = min(filters["years"])
            max_year = max(filters["years"])
            es_query["bool"]["filter"].append({
                "range": {
                    "years": {
                        "gte": min_year,
                        "lte": max_year
                    }
                }
            })
        
        if "named_entities" in filters and filters["named_entities"]:
            es_query["bool"]["filter"].append({
                "terms": {
                    "named_entities": filters["named_entities"]
                }
            })

    return es_query

def build_qdrant_filter(filters=None):

    if not filters:
        return None
    
    conditions = []
    
    if "years" in filters and filters["years"]:
        min_year = min(filters["years"])
        max_year = max(filters["years"])
        conditions.append(FieldCondition(
            key="years",
            range=Range(gte=min_year, lte=max_year)
        ))

    if "named_entities" in filters and filters["named_entities"]:
        conditions.append(FieldCondition(
            key="named_entities",
            match=MatchAny(any=filters["named_entities"])
        ))
        
    if not conditions:
        return None
        
    return Filter(must=conditions)

def hybrid_search(query_text, filters=None, limit=10, weight_es=0.5, weight_qdrant=0.5):

    es_body = {
        "query": build_es_query(query_text, filters),
        "size": limit
    }
    
    try:
        es_resp = es.search(index=INDEX_NAME_ES, body=es_body)
        es_hits = es_resp['hits']['hits']
    except Exception as e:
        print(f"Błąd ES: {e}")
        es_hits = []

    qdrant_hits = []
    try:
        query_vector = model.encode(query_text).tolist()
        qdrant_filter = build_qdrant_filter(filters)
        
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME_QDRANT,
            query=query_vector,
            query_filter=qdrant_filter,  
            limit=limit,
            with_payload=True
        )
        qdrant_hits = response.points 
        
    except Exception as e:
        print(f"Błąd Qdrant: {e}")
        qdrant_hits = []

    scores = {}
    
    for rank, hit in enumerate(es_hits):
        doc_id = hit['_id']
        score = weight_es * (1 / (60 + rank))
        
        if doc_id not in scores:
            scores[doc_id] = {"score": 0, "doc": hit['_source']}
        scores[doc_id]["score"] += score

    for rank, hit in enumerate(qdrant_hits):
        doc_id = str(hit.id)
        score = weight_qdrant * (1 / (60 + rank))
        
        if doc_id not in scores:
            scores[doc_id] = {"score": 0, "doc": hit.payload}
        else:
            scores[doc_id]["score"] += score

    sorted_docs = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
    
    return [item['doc'] for item in sorted_docs[:limit]]