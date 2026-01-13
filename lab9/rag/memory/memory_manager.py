import json
import os
from datetime import datetime

MEMORY_FILE = "rag/memory/pending_queries.json"

class MemoryManager:
    def __init__(self, filepath=MEMORY_FILE):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-16") as f:
                json.dump([], f)

    def add_pending_query(self, query: str, reason: str = "no_results"):
        pending = self.get_pending_queries()
        
        for item in pending:
            if item["query"] == query:
                return

        new_entry = {
            "id": len(pending) + 1,
            "query": query,
            "status": "pending",
            "reason": reason,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        pending.append(new_entry)
        self._save_file(pending)
        print(f"Odłożono pytanie do wyjaśnienia: '{query}'")

    def get_pending_queries(self):
        with open(self.filepath, "r", encoding="utf-16") as f:
            return json.load(f)

    def _save_file(self, data):
        with open(self.filepath, "w", encoding="utf-16") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

memory = MemoryManager()