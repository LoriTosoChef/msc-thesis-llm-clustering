from typing import List
import logging

import hdbscan
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class ClusteringModel:
    def __init__(self,
                 model_name: str,
                 random_state: int = 42,
                 **kwargs) -> None:
        
        self.model_name = model_name
        self.random_state = random_state
        
        logger.info(f'Initializing {self.model_name.upper()}')
        if 'kmeans' in self.model_name:
            logger.info(f'N_CLUSTERS: {kwargs["n_clusters"]} - MAX_ITER: {kwargs["max_iter"]} - TOL: {kwargs["tol"]}')
            self.model = KMeans(n_clusters=kwargs['n_clusters'],
                                max_iter=kwargs['max_iter'],
                                tol=kwargs['tol'],
                                n_init=kwargs['n_init'],
                                random_state=self.random_state)
        elif 'dbscan' in self.model_name:
            logger.info(f'EPS: {kwargs["eps"]} - MIN_SAMPLES: {kwargs["min_samples"]} - METRIC: {kwargs["metric"]}')
            self.model = DBSCAN(eps=kwargs['eps'],
                                min_samples=kwargs['min_samples'],
                                metric=['metric'],)
        elif 'hdbscan' in self.model_name:
            logger.info(f'EPS: {kwargs["eps"]} - MIN_SIZE: {kwargs["min_cluster_size"]} - MIN_SAMPLE: {kwargs["min_samples"]}')
            self.model = hdbscan.HDBSCAN(min_cluster_size=kwargs['min_cluster_size'],
                                         min_samples=kwargs['min_minsaples'],
                                         metric=kwargs['metric'],
                                         cluster_selection_epsilon=kwargs['eps'])
            
    
    def fit_predict(self,
                    embeddings: List[float],
                    pca_flag: bool = False,
                    **kwargs):
        
        logger.info(f'{self.model_name.upper()}: searching clusters...')
        
        if pca_flag:
            logger.info(f'Performing PCA with {kwargs["n_components"]} components...')
            pca = PCA(n_components=kwargs['n_components'])
            embeddings = pca.fit_transform(embeddings)
        
        self.model.fit(embeddings)
        self.clusters = self.model.labels_
        self.embeddings = embeddings
        
