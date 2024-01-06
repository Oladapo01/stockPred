import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def perform_clustering(preprocessed_data, num_clusters, ticker_names):
    # Perform PCA
    pca = PCA(n_components=min(preprocessed_data.shape[0], preprocessed_data.shape[1]))
    data_reduced = pca.fit_transform(preprocessed_data)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(data_reduced)

    # Create a DataFrame with Ticker and corresponding cluster labels
    tickers = ticker_names

    print("Length of ticker_names:", len(ticker_names))
    print("Length of cluster_labels:", len(cluster_labels))

    # Make sure the length of tickers matches the length of cluster_labels
    if len(ticker_names) != len(cluster_labels):
        raise ValueError("Mismatch in lengths of tickers and cluster labels")
    result_data = pd.DataFrame({'Ticker': tickers, 'Cluster': cluster_labels})

    return result_data
