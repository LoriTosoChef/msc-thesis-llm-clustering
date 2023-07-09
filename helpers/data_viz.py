from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from wordcloud import WordCloud


logger = logging.getLogger(__name__)


def plot_clusters_2D(df: pd.DataFrame,
                     model: str,
                     cluster_algo: str,
                     save: bool,
                     fig_name: str):

    # dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(np.array(df[f'{model}_embeddings'].to_list()))
    # init data temp df and plot
    temp_df = pd.DataFrame({'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1], 'label': df[f'{model}_{cluster_algo}'], 'text': df[f'{model}'], 'tweet': df['full_text']})
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    # set colors
    colors = ['lightpink', 'lightblue', 'lightgreen']
    for ax, column in zip(axs, ['text', 'tweet']):
        scatter = ax.scatter(temp_df['x'], temp_df['y'], c=temp_df['label'].apply(lambda label: colors[label]), alpha=0.5)
        # subsample of labels to plot
        subset_labels = temp_df.groupby('label').sample(n=100)
        # K-means clustering on the subset of points to find points that are far from each other
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(subset_labels[['x', 'y']])
        centroids = kmeans.cluster_centers_
        # find the closest point to each centroid and use it for annotation
        highlighted_points = []
        for centroid in centroids:
            closest_point = ((subset_labels['x'] - centroid[0])**2 + (subset_labels['y'] - centroid[1])**2).idxmin()
            highlighted_points.append(closest_point)

        # second plot for highlighted data points
        highlighted_df = temp_df.loc[highlighted_points]
        highlighted_scatter = ax.scatter(highlighted_df['x'], highlighted_df['y'], c=highlighted_df['label'].apply(lambda label: colors[label]), edgecolor='black', s=25)
        # create labels
        for i, point in highlighted_df.iterrows():
            words = point[column].split()[:18]
            label_lines = [' '.join(words[i:i+4]) for i in range(0, len(words), 6)]
            label = '\n'.join(label_lines + ['...\nCluster: ' + str(point['label'])])

            ax.annotate(label, (point['x'], point['y']))

    plt.tight_layout()
    if save:
        plt.savefig(f'fig/{fig_name}.png', dpi=300)  # Save with a high DPI value
    plt.show()
    
    return


def plot_wordcloud(df: pd.DataFrame,
                   model: str,
                   clustering_algo: str,
                   source_text_col: str,
                   save: bool):
    
    # loop through all cluster classes
    for cluster in df[f'{model}_{clustering_algo}'].unique():
        cluster_df = df[df[f'{model}_{clustering_algo}'] == cluster]
        # gather texts and create wordcloud
        text = ' '.join(cluster_df[source_text_col])
        wordcloud = WordCloud(max_font_size=50, max_words=75, background_color='white').generate(text=text)
        # plot
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f'{model.upper()} - Cluster {cluster}')
        if save:
            plt.savefig(f'fig/wordcloud_{model}_{clustering_algo}.png', dpi=300)
        plt.show()
        
        return