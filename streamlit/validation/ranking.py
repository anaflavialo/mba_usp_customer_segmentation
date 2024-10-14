import pandas as pd

def get_ranking(df):
    ranking = pd.DataFrame()
    metrics = ["silhouette_score", "ch_score", "db_score"]
    
    ranking_1 = df["silhouette_score"].sort_values(ascending=False).reset_index()
    ranking_1["posicao1"] = ranking_1.index + 1
    del ranking_1["index"]
    
    ranking_2 = df["ch_score"].sort_values(ascending=False).reset_index()
    ranking_2["posicao2"] = ranking_2.index + 1
    del ranking_2["index"]
    
    ranking_3 = df["db_score"].sort_values(ascending=True).reset_index()
    ranking_3["posicao3"] = ranking_3.index + 1
    del ranking_3["index"]
        
    ranking_final = (
        df
        .join(
            ranking_1.set_index("silhouette_score"),
            "silhouette_score",
            "left"
        )
        .join(
            ranking_2.set_index("ch_score"),
            "ch_score",
            "left"
        )
        .join(
            ranking_3.set_index("db_score"),
            "db_score",
            "left"
        )
    )
    
    ranking_final["posicao_final"] = ranking_final["posicao1"] + ranking_final["posicao2"] + ranking_final["posicao3"]
    
    return ranking_final

def get_best_n(df_ranking): 
    return df_ranking[df_ranking["posicao_final"] == df_ranking["posicao_final"].min()]["n"].values[0]

def get_best_alg(df_ranking): 
    return df_ranking[df_ranking["posicao_final"] == df_ranking["posicao_final"].min()]["alg"].values[0]

def get_scores_from_n(df_ranking): 
    return df_ranking[df_ranking["posicao_final"] == df_ranking["posicao_final"].min()]

