from typing import List

import logging
import numpy as np

from models.clustering import ClusteringModel
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)



def clustering_scores(embeddings: List[float], clusters: List[int]) -> dict:
    scores = {}
    scores['calinski'] = calinski_harabasz_score(embeddings, clusters)
    scores['davies'] = davies_bouldin_score(embeddings, clusters)
    scores['silhouette'] = silhouette_score(embeddings, clusters)
    
    logger.info(f'Calinski-Harabasz Index: {scores["calinski"]}')
    logger.info(f'Davies-Bouldin Index: {scores["davies"]}')
    logger.info(f'Silhouette Score: {scores["silhouette"]}')
    
    return scores


def dbscan_loop(data,
                n_components_space: List[int],
                min_samples_space = List[int]) -> dict:

    res = {}
    res['n_components'] = []
    res['min_samples'] = []
    res['score'] = []

    for n_components in n_components_space:
        for min_samples in min_samples_space:
            dbscan = ClusteringModel(model_name='dbscan',
                                    min_samples=min_samples,
                                    metric='euclidean',
                                    eps=0.5)

            dbscan.fit_predict(embeddings=data, pca_flag=True, n_components=n_components)

            try:
                scores = silhouette_score(data, dbscan.clusters)
            except Exception as e:
                print(e)
                scores = -np.inf

            res['n_components'].append(n_components)
            res['min_samples'].append(min_samples)
            res['score'].append(scores)
            
    return res


def kmeans_loop(data,
                n_components_space: List[int],
                n_clusters_space: List[int],
                max_iter_space: List[int],
                tol_space: List[float]) -> dict:

    res = {}
    res['n_clusters'] = []
    res['max_iter'] = []
    res['tol'] = []
    res['n_components'] = []
    res['score'] = []

    for n_components in n_components_space:
        for n_clusters in n_clusters_space:
            for max_iter in max_iter_space:
                for tol in tol_space:
                    kmeans = ClusteringModel(model_name='kmeans',
                                             n_clusters=n_clusters,
                                             max_iter=max_iter,
                                             tol=tol,
                                             n_init='auto')

                    kmeans.fit_predict(embeddings=data, pca_flag=True, n_components=n_components)

                    try:
                        scores = silhouette_score(data, kmeans.clusters)
                    except Exception as e:
                        print(e)
                        scores = -np.inf

                    res['n_components'].append(n_components)
                    res['max_iter'].append(max_iter)
                    res['n_clusters'].append(n_clusters)
                    res['tol'].append(tol)
                    res['score'].append(scores)
    
    return res