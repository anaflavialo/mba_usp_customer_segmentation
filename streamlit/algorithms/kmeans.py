from sklearn.cluster import KMeans
from utils import rfm

def apply_kmeans(df_rfm, n_clusters, all_columns):
    kmeans = KMeans(
        n_clusters=n_clusters, 
        algorithm='lloyd',
        copy_x=True,
        init='k-means++',
        max_iter=300,
        n_init='warn',
        random_state=42,
        tol=0.0001,
        verbose=0
    )
    df_rfm = df_rfm[all_columns]
    clusters = kmeans.fit(df_rfm)
    
    return clusters
