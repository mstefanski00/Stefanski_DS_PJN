from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

MODEL_NAME = "Babelscape/wikineural-multilingual-ner"

class NERExtractor:
    def __init__(self):
        print(f"Ładowanie modelu NER: {MODEL_NAME}.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
            self.nlp = pipeline("token-classification", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
            print("Model NER gotowy.")
        except Exception as e:
            print(f"Błąd ładowania modelu NER: {e}")
            self.nlp = None

    def extract(self, text):
        if not self.nlp or not text:
            return []
        
        processed_text = text[:1000] 

        try:
            results = self.nlp(processed_text)
        except Exception:
            return []

        entities = set()
        for entity in results:
            if entity['score'] > 0.60:
                word = entity['word'].strip()
                if len(word) > 2:
                    entities.add(word)
        
        return list(entities)