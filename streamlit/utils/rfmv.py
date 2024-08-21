def get_recency(df):
    max_date = df["order_purchase"].max()
    df_recency = df.groupby("customer_unique_id").agg({"order_purchase": "max"}).reset_index()
    df_recency["max_date"] = max_date
    df_recency["recency"] = df_recency["max_date"] - df_recency["order_purchase"]
    df_recency["recency"] = df_recency["recency"].apply(lambda x: x.days)
    df_recency = df_recency[["customer_unique_id","recency"]]

    return df_recency


def get_frequency(df):
    df_frequency = df.groupby("customer_unique_id").agg({"order_id": "count"}).rename({"order_id": "frequency"}, axis=1).reset_index()
    return df_frequency

def get_monetary(df):
    df_monetary = df.groupby("customer_unique_id").agg({"monetary": "sum"}).reset_index()
    return df_monetary

def get_variety(df):
    df_variety = df.groupby("customer_unique_id").agg({"product_id": "nunique", "product_category_name": "nunique"}).rename({"product_id": "product_variety", "product_category_name": "category_variety"}, axis=1).reset_index()
    return df_variety

def get_rfmv(df):
    df_recency = get_recency(df)
    df_frequency = get_frequency(df)
    df_monetary = get_monetary(df)
    df_variety = get_variety(df)

    df_rfmv = (
        df_recency
        .join(
            df_frequency.set_index("customer_unique_id"),
            on="customer_unique_id",
            how="inner"
        )
        .join(
            df_monetary.set_index("customer_unique_id"),
            on="customer_unique_id",
            how="inner"
        )
        .join(
            df_variety.set_index("customer_unique_id"),
            on="customer_unique_id",
            how="inner"
        )
    )

    return df_rfmv

def get_rfmv_std(df):
    df_rfmv = get_rfmv(df)
    df_rfmv_std = df_rfmv.copy()

    cols_to_std = ["recency", "frequency", "monetary", "product_variety", "category_variety"]

    for col in cols_to_std:
        x_mean = df_rfmv[col].mean()
        x_std = df_rfmv[col].std()
        
        df_rfmv_std[col] = df_rfmv[col].apply(lambda x: (x - x_mean)/(x_std))
    
    return df_rfmv_std

def get_customer_segmentation(df, df_rfmv):
    df_rfmv = df_rfmv[["customer_unique_id","segmentation"]]

    df_final = (
        df 
        .join(
            df_rfmv.set_index("customer_unique_id"),
            on="customer_unique_id",
            how="inner"
        )
    )

    return df_final

