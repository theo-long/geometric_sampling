"""Various methods and classes for graph community detection."""
from sklearn.cluster import spectral_clustering
import networkx as nx

def spectral_clusters(G: nx.Graph, k: int):
    A = nx.adjacency_matrix(G)
    clusters = spectral_clustering(A, n_clusters=k, n_components=k)
    return clusters
