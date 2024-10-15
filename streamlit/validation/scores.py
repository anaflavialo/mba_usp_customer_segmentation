import pandas as pd
from sklearn import metrics
from algorithms import agglomerative_clustering, bisecting_kmeans, kmeans

def get_scores_from_alg(df_rfm, alg):
    silhouette_score = []
    ch_score = []
    db_score = []

    df_scores = pd.DataFrame()

    min_number_of_clusters = 2
    max_number_of_clusters = 10

    all_columns = df_rfm.columns[1:]

    for i in range(min_number_of_clusters, max_number_of_clusters + 1):

        if(alg == "kmeans"): 
            labels = kmeans.apply_kmeans(df_rfm, i, all_columns).labels_
        elif(alg == "agg"): 
            labels = agglomerative_clustering.apply_agglomerative_clustering(df_rfm, i, all_columns).labels_
        else: 
            labels = bisecting_kmeans.apply_bisecting_kmeans(df_rfm, i, all_columns).labels_

        silhouette_score.append(metrics.silhouette_score(df_rfm[all_columns], labels, metric='euclidean'))
        ch_score.append(metrics.calinski_harabasz_score(df_rfm[all_columns], labels))
        db_score.append(metrics.davies_bouldin_score(df_rfm[all_columns], labels))

    df_scores['n'] = range(min_number_of_clusters, max_number_of_clusters + 1)
    df_scores['silhouette_score'] = silhouette_score
    df_scores['ch_score'] = ch_score
    df_scores['db_score'] = db_score

    return df_scores
            