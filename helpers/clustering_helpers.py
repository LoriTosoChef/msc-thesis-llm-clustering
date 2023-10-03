from typing import List

import logging
import numpy as np

from models.clustering import ClusteringModel
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)


def dbscan_loop(data,
                n_components_space: List[int],
                min_samples_space = List[int],
                eps_space=List[float]) -> dict:

    res = {}
    res['n_components'] = []
    res['actual_components'] = []
    res['min_samples'] = []
    res['score'] = []
    res['eps'] = []
    res['clusters'] = []

    for n_components in n_components_space:
        for min_samples in min_samples_space:
            for eps in eps_space:
                dbscan = ClusteringModel(model_name='dbscan',
                                         min_samples=min_samples,
                                         metric='euclidean',
                                         eps=eps)

                dbscan.fit_predict(embeddings=data, pca_flag=True, n_components=n_components)

                try:
                    scores = silhouette_score(data, dbscan.clusters)
                except Exception as e:
                    logger.debug(e)
                    scores = -np.inf
                
                if 1 in dbscan.clusters:
                    logger.info(f'Found them')
                    
                    res['n_components'].append(n_components)
                    res['actual_components'].append(dbscan.actual_components)
                    res['min_samples'].append(min_samples)
                    res['score'].append(scores)
                    res['eps'].append(eps)
                    
    return res


def kmeans_loop(data,
                n_components_space: List[int],
                n_clusters_space: List[int],
                max_iter_space: List[int]) -> dict:

    res = {}
    res['n_clusters'] = []
    res['max_iter'] = []
    res['n_components'] = []
    res['actual_components'] = []
    res['score'] = []

    for n_components in n_components_space:
        for n_clusters in n_clusters_space:
            for max_iter in max_iter_space:
                kmeans = ClusteringModel(model_name='kmeans',
                                         n_clusters=n_clusters,
                                         max_iter=max_iter,
                                         n_init='auto')

                kmeans.fit_predict(embeddings=data, pca_flag=True, n_components=n_components)

                try:
                    scores = silhouette_score(data, kmeans.clusters, metric='euclidean')
                except Exception as e:
                    logger.debug(e)
                    scores = -np.inf

                res['n_components'].append(n_components)
                res['actual_components'].append(kmeans.actual_components)
                res['max_iter'].append(max_iter)
                res['n_clusters'].append(n_clusters)
                res['score'].append(scores)
        
    return res


def get_best_scores(results: dict, model_name: str):
    best_scores = {}
    if 'kmeans' in model_name.lower():
        res_array = {key: np.array(value) for key, value in results.items()}
        
        best_score_index = np.argmax(res_array['score'])

        best_scores['score'] = res_array['score'][best_score_index]
        best_scores['n_components'] = res_array['n_components'][best_score_index]
        best_scores['actual_components'] = res_array['actual_components'][best_score_index]
        best_scores['max_iter'] = res_array['max_iter'][best_score_index]
        best_scores['n_clusters'] = res_array['n_clusters'][best_score_index]
        
        logger.info(f"Best Score: {best_scores['score']}")
        
        return best_scores
    
    elif 'dbscan' in model_name:
        res_array = {key: np.array(value) for key, value in results.items()}
        try:
            best_score_index = np.argmax(res_array['score'])
            
            best_scores['score'] = res_array['score'][best_score_index]
            best_scores['n_components'] = res_array['n_components'][best_score_index]
            best_scores['actual_components'] = res_array['actual_components'][best_score_index]
            best_scores['min_samples'] = res_array['min_samples'][best_score_index]
            best_scores['eps'] = res_array['eps'][best_score_index]
            
            logger.info(f"Best Score: {best_scores['score']}")
            
            return best_scores
        except Exception as e:
            logger.warning(e)
            return {}