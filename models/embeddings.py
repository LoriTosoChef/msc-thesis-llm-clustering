import logging

from sentence_transformers import SentenceTransformer
import gensim.downloader as api

logger = logging.getLogger(__name__)


class SentenceEmbeddings:
    def __init__(self, name: str) -> None:
        self.model_name = name
        
        if 'mpnet' in self.model_name:
            self.model = SentenceTransformer('all-mpnet-base-v2')
        elif 'distilroberta' in self.model_name:
            self.model = SentenceTransformer('all-distilroberta-v1')
            
    
    def generate_embeddings(self, text):
        return self.model.encode(text)
    

class WordEmbeddings:
    def __init__(self, name: str) -> None:
        self.model_name = name
        
        if 'w2v-google' in self.model_name:
            self.model = api.load('word2vec-google-news-300')
        elif 'glove-twitter' in self.model_name:
            self.model = api.load('glove-twitter-200')
        elif 'glove-twitter' in self.model_name:
            self.model = api.load('glove-wiki-gigaword-300')
    
    
    def generate_embeddings(self, text):
        pass