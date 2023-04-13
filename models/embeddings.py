import logging
import config

from sentence_transformers import SentenceTransformer
import gensim.downloader as api

logger = logging.getLogger(__name__)


class SentenceEmbeddings:
    def __init__(self, name: str) -> None:
        self.model_name = name
        
        logger.info(f'Initializing {self.model_name.upper()} for Sentence Embeddings')
        
        if 'mpnet' in self.model_name:
            logger.debug('Loading all-mpnet-base-v2 model')
            self.model = SentenceTransformer('all-mpnet-base-v2')
        elif 'distilroberta' in self.model_name:
            logger.debug('Loading all-distilroberta-v1')
            self.model = SentenceTransformer('all-distilroberta-v1')
            
    
    def generate_embeddings(self, text: list):
        logger.info(f'{self.model_name.upper()} - Generating sentence embeddings...')
        return self.model.encode(text)
    

class WordEmbeddings:
    def __init__(self, name: str) -> None:
        self.model_name = name
        
        logger.info(f'Initializing {self.model_name.upper()} for Word Embeddings')
        
        if 'w2v-google' in self.model_name:
            logger.debug('Loading word2vec-google-news-300')
            self.model = api.load('word2vec-google-news-300')
        elif 'glove-twitter' in self.model_name:
            logger.debug('Loading glove-twitter-200')
            self.model = api.load('glove-twitter-200')
        elif 'glove-twitter' in self.model_name:
            logger.debug('Loading glove-wiki-gigaword-300')
            self.model = api.load('glove-wiki-gigaword-300')
    
    
    def generate_embeddings(self, text):
        pass