def filter_retrieved_docs(docs, min_words=20, max_docs=5):

    filtered_docs = []
    rejected_short = 0
    
    for doc in docs:
        text = doc.get('text', '')
        word_count = len(text.split())
        
        if word_count >= min_words:
            filtered_docs.append(doc)
        else:
            rejected_short += 1
            
    final_docs = filtered_docs[:max_docs]
    
    stats = {
        "input_count": len(docs),
        "rejected_short": rejected_short,
        "final_count": len(final_docs),
        "rejected_ratio": f"{(rejected_short / len(docs) * 100):.1f}%" if docs else "0%"
    }
    
    return final_docs, stats