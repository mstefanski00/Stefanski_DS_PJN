import re
import json
import requests

class DateExtractorHybrid:
    def __init__(self, llm_model="gemma2:2b", llm_url="http://localhost:11434/api/chat"):
        self.llm_model = llm_model
        self.llm_url = llm_url

    def extract_regex(self, text):

        if not text:
            return {"years": [], "dates": []}

        years = set()
        dates = set()

        dates.update(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text))
        
        dates.update(re.findall(r'\b\d{1,2}\.\d{1,2}\.\d{4}\b', text))


        for match in re.finditer(r'\b(19|20)\d{2}\b', text):
            years.add(int(match.group(0)))

        context_years = re.findall(r'w (\d{4}) roku', text)
        years.update([int(y) for y in context_years])

        return {
            "years": sorted(list(years)),
            "dates": sorted(list(dates))
        }

    def extract_llm(self, text):

        prompt = f"""
        Jesteś ekspertem od ekstrakcji czasu.
        Wyodrębnij z poniższego tekstu:
        1. 'dates': konkretne daty (YYYY-MM-DD lub DD.MM.YYYY).
        2. 'years': lata jako liczby całkowite.
        3. 'ranges': zakresy czasowe (tekstowo).
        
        Zwróć TYLKO poprawny JSON w formacie:
        {{
          "dates": [],
          "years": [],
          "ranges": []
        }}
        
        Tekst: {text}
        """

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0}
        }

        try:
            response = requests.post(self.llm_url, json=payload, timeout=60)
            if response.status_code != 200:
                return {"error": f"API Error {response.status_code}"}
                
            content = response.json().get('message', {}).get('content', '')
            
            clean_json = re.sub(r'```json|```', '', content).strip()
            return json.loads(clean_json)
        except Exception as e:
            return {"error": str(e), "dates": [], "years": [], "ranges": []}