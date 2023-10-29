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
                     llm: str,
                     cluster_algo: str,
                     text_col: str,
                     n: int = 5,
                     indices: list = None,
                     manual: bool = False,
                     save_fig: bool = False,
                     fig_name: str = None):
   
   # init plot
   fig, ax = plt.subplots(1,1,figsize=(15,7))
   # Apply t-SNE to reduce embeddings to 2D
   tsne = TSNE(n_components=2, random_state=42)
   embeddings_2d = tsne.fit_transform(np.array(df[f'{llm}_embeddings'].tolist()))
   # create a new DataFrame for 2D embeddings and cluster labels
   embeddings_df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
   embeddings_df['label'] = df[f'{llm}_{cluster_algo}'].values
   if text_col == 'tweet':
      embeddings_df['text'] = df['full_text'].values
   elif text_col == 'output':
      embeddings_df['text'] = df[llm].values
   #color mapping
   labels = embeddings_df['label'].unique()
   colors = ['lightpink', 'lightblue', 'lightgreen', 'lightsalmon', 'green', 'red', 'yellow', 'bisque', 'lightcyan', 'thistle', 'lavender']
   color_dict = dict(zip(labels, colors))
   # create labels
   for label in labels:
      temp = embeddings_df[embeddings_df['label'] == label]
      ax.scatter(temp['x'], temp['y'], c=color_dict[label], label=label)

   if manual:
      highlighted_df = embeddings_df.loc[indices]
      ax.scatter(highlighted_df['x'], highlighted_df['y'], c=highlighted_df['label'].apply(lambda label: color_dict[label]), edgecolor='black', s=25)
   else:
         # KMeans on the embeddings to select four labels that are far from each other
      kmeans = KMeans(n_clusters=n, random_state=np.random.randint(low=1, high=100))
      kmeans_label = kmeans.fit_predict(embeddings_2d)
      # four datapoints closest to the kmeans cluster centers
      indices = []
      for center in kmeans.cluster_centers_:
         distance = np.sum((embeddings_2d - center)**2, axis=1)
         indices.append(np.argmin(distance))
      highlighted_df = embeddings_df.loc[indices]
      ax.scatter(highlighted_df['x'], highlighted_df['y'], c=highlighted_df['label'].apply(lambda label: color_dict[label]), edgecolor='black', s=25)
      
   # create labels
   for i, point in highlighted_df.iterrows():
      words = point['text'].split()[:18]
      label_lines = [' '.join(words[i:i+4]) for i in range(0, len(words), 6)]
      label = '\n'.join(label_lines + ['...\nCluster: ' + str(point['label'])])

      ax.annotate(label, (point['x'], point['y']))
   
   plt.title(f'{llm.upper()} - {cluster_algo.upper()} clusters')   
   #plt.legend()
   if save_fig:
        plt.savefig(f'fig/{llm}_{cluster_algo}_{fig_name}.png', dpi=300)  # Save with a high DPI value
   plt.show()


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