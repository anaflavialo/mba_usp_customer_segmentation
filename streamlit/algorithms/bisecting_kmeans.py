from sklearn.cluster import BisectingKMeans

def apply_bisecting_kmeans(df_rfm, n_clusters, all_columns):
    bsc = BisectingKMeans(
        n_clusters=n_clusters, 
        algorithm='lloyd',
        bisecting_strategy='biggest_inertia',
        copy_x=True,
        init='k-means++',
        max_iter=300,
        n_init=1,
        random_state=42,
        tol=0.0001,
        verbose=0
    )
    df_rfm = df_rfm[all_columns]
    clusters = bsc.fit(df_rfm)
    
    return clusters
