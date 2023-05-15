from typing import List
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_clusters(embeddings: List[float], clusters: List[int]):
    # Create a scatter plot of the embeddings colored by cluster
    plt.scatter(
        x=[e[0] for e in embeddings], # x-coordinates of the embeddings
        y=[e[1] for e in embeddings], # y-coordinates of the embeddings
        c=clusters, # cluster assignments for each data point
        cmap='viridis' # colormap to use for coloring the data points
    )
    plt.show()