import logging
import pandas as pd
import numpy as np

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
        elif 'distil-roberta' in self.model_name:
            logger.debug('Loading all-distilroberta-v1')
            self.model = SentenceTransformer('all-distilroberta-v1')
            
    
    def generate_embeddings(self, input_texts: pd.Series):
        logger.info(f'{self.model_name.upper()} - Generating sentence embeddings...')
        return self.model.encode(input_texts.to_list())
    

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
        elif 'glove-wiki' in self.model_name:
            logger.debug('Loading glove-wiki-gigaword-300')
            self.model = api.load('glove-wiki-gigaword-300')
    
    
    def generate_embeddings(self, input_texts: pd.Series) -> list:
        logger.info(f'{self.model_name.upper()} - Generating sentence embeddings...')
        # init embeddings list
        embeddings = []
        for tokens in input_texts:
            # init zero vector and vector list
            zero_vector = np.zeros(self.model.vector_size)
            vectors = []
            for token in tokens:
                if token in self.model:
                    try:
                        logger.debug(f'Checking {token} in vocabulary')
                        vectors.append(self.model[token])
                    except KeyError:
                        logger.warning(f'{token} not in vocabulary, skipping...')
                        continue
            if vectors:
                vectors = np.asarray(vectors)
                # compute average across 0 axis
                avg_vec = vectors.mean(axis=0)
                embeddings.append(avg_vec)
            else:
                embeddings.append(zero_vector)
        
        return embeddings