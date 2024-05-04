from sklearn.cluster import KMeans
from utils import rfmv

def apply_kmeans(df_rfmv, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_rfmv = df_rfmv[["recency", "frequency", "monetary", "product_variety", "category_variety"]]
    clusters = kmeans.fit_predict(df_rfmv)
    
    return clusters
