from typing import List
import logging

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)



def clustering_scores(embeddings: List[float], clusters: List[int]) -> dict:
    scores = {}
    scores['calinski'] = calinski_harabasz_score(embeddings, clusters)
    scores['davies'] = davies_bouldin_score(embeddings, clusters)
    scores['silhouette'] = silhouette = silhouette_score(embeddings, clusters)
    
    logger.info(f'Calinski-Harabasz Index: {scores["calinski"]}')
    logger.info(f'Davies-Bouldin Index: {scores["davies"]}')
    logger.info(f'Silhouette Score: {scores["silhouette"]}')
    
    return scores