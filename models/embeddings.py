import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, name: str) -> None:
        self.model_name = name
        
        if 'mpnet' in self.model_name:
            self.model = SentenceTransformer('all-mpnet-base-v2')
        elif 'distilroberta' in self.model_name:
            self.model = SentenceTransformer('all-distilroberta-v1')
            
    
    def generate_embeddings(self, text):
        return self.model.encode(text)