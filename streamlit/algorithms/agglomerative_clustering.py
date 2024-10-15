from sklearn.cluster import AgglomerativeClustering

def apply_agglomerative_clustering(df_rfm, n_clusters, all_columns):
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='deprecated',
        compute_distances=False,
        compute_full_tree='auto',
        connectivity=None,
        distance_threshold=None,
        linkage='ward',
        memory=None,
        metric=None,
    )
    df_rfm = df_rfm[all_columns]
    clusters = agg.fit(df_rfm)
    
    return clusters
