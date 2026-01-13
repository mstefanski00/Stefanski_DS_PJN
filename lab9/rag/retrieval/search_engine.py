from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from collections import defaultdict

QDRANT_URL = "http://localhost:6333"
ES_URL = "http://localhost:9200"
COLLECTION_NAME = "culturax_semantic"
ES_INDEX = "culturax_fts"
VECTOR_SIZE = 384

qdrant = QdrantClient(QDRANT_URL)
es = Elasticsearch(ES_URL, request_timeout=30)

model = SentenceTransformer("all-MiniLM-L6-v2")


def search_qdrant(query: str, limit: int = 5):
    query_vector = model.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points

    return [
        {
            "id": r.id,
            "score": r.score,
            "text": r.payload.get("text", "")
        }
        for r in results
    ]

def search_es(query: str, limit: int = 5):
    body = {
        "size": limit,
        "query": {
            "match": {
                "text": {
                    "query": query
                }
            }
        }
    }

    res = es.search(index=ES_INDEX, body=body)

    return [
        {
            "id": hit["_id"],
            "score": hit["_score"],
            "text": hit["_source"]["text"]
        }
        for hit in res["hits"]["hits"]
    ]


def rank_fusion(
    qdrant_results,
    es_results,
    weight_qdrant=1.0,
    weight_es=1.0,
    k=5
):
    scores = defaultdict(float)
    documents = {}

    for rank, doc in enumerate(qdrant_results, start=1):
        scores[doc["id"]] += weight_qdrant / rank
        documents[doc["id"]] = doc["text"]

    for rank, doc in enumerate(es_results, start=1):
        scores[doc["id"]] += weight_es / rank
        documents[doc["id"]] = doc["text"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "id": doc_id,
            "score": score,
            "text": documents[doc_id]
        }
        for doc_id, score in ranked[:k]
    ]


def hybrid_search(
    query: str,
    limit: int = 5,
    weight_qdrant=1.0,
    weight_es=1.0
):
    qdrant_res = search_qdrant(query, limit)
    es_res = search_es(query, limit)

    hybrid_res = rank_fusion(
        qdrant_res,
        es_res,
        weight_qdrant=weight_qdrant,
        weight_es=weight_es,
        k=limit
    )

    return qdrant_res, es_res, hybrid_res

def dynamic_hybrid_search(
    query: str, 
    limit: int = 5, 
    threshold: float = 0.70
):
    
    q_res = search_qdrant(query, limit)
    es_res = search_es(query, limit)

    semantic_confidence = q_res[0]['score'] if q_res else 0.0

    if semantic_confidence < threshold:
        w_q = 1.0
        w_es = 10.0 
        strategy = "Niska pewność semantyczna - ES boost."
    else:
        w_q = 5.0
        w_es = 1.0
        strategy = "Wysoka pewność semantyczna - Qdrant boost."

    hybrid_res = rank_fusion(
        q_res, 
        es_res, 
        weight_qdrant=w_q, 
        weight_es=w_es, 
        k=limit
    )

    return q_res, es_res, hybrid_res, strategy, semantic_confidence

def expand_query_with_llm(user_query: str):

    mock_responses = {
        "co tam?": [
            "najnowsze wydarzenia w Polsce 2023",
            "ciekawostki kulturalne i społeczne",
            "przegląd prasy i newsy dnia"
        ],
        "opowiedz mi coś o technologii": [
            "rozwój sztucznej inteligencji i AI",
            "nowoczesne technologie w medycynie",
            "wpływ cyfryzacji na gospodarkę"
        ],
        "ciekawe miejsca": [
            "atrakcje turystyczne w Polsce",
            "zabytki i historia regionów",
            "parki narodowe i przyroda"
        ]
    }
    
    return mock_responses.get(user_query, [user_query])


def global_rank_fusion(results_list_of_lists, k=60):
 
    scores = defaultdict(float)
    texts = {}
    doc_details = {} 
    for result_list in results_list_of_lists:
        for rank, item in enumerate(result_list, 1):
            doc_id = str(item['id']) 
            
            scores[doc_id] += 1 / (k + rank)
            
            texts[doc_id] = item['text']
            doc_details[doc_id] = item

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for doc_id, score in sorted_docs[:5]:
        final_results.append({
            "id": doc_id,
            "score": score,
            "text": texts[doc_id]
        })
        
    return final_results