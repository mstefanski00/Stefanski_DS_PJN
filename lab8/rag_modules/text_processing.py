def chunk_document(text, word_limit=150):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), word_limit):
        chunk = words[i:i + word_limit]
        chunks.append(" ".join(chunk))
        
    return chunks