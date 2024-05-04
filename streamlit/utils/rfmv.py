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

def get_rfmv_normalized(df):
    df_rfmv = get_rfmv(df)

    cols_to_normalize = ["recency", "frequency", "monetary", "product_variety", "category_variety"]

    for col in cols_to_normalize:
        x_max = df_rfmv[col].max()
        x_min = df_rfmv[col].min()
        
        df_rfmv[col] = df_rfmv[col].apply(lambda x: (x - x_min)/(x_max - x_min))
    
    return df_rfmv

