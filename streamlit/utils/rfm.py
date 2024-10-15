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

def get_rfm(df):
    df_recency = get_recency(df)
    df_frequency = get_frequency(df)
    df_monetary = get_monetary(df)

    df_rfm = (
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
    )

    return df_rfm

def get_rfm_std(df_rfm):
    df_rfm_std = df_rfm.copy()

    df_rfm_std["recency"] = df_rfm_std["recency"].apply(lambda x: 1 if x == 0 else 1/x)

    cols_to_std = df_rfm_std.columns[1:]

    for col in cols_to_std:
        x_mean = df_rfm_std[col].mean()
        x_std = df_rfm_std[col].std()
        
        df_rfm_std[col] = df_rfm_std[col].apply(lambda x: (x - x_mean)/(x_std))
    
    return df_rfm_std

def get_customer_segmentation(df, df_rfm):
    df_rfm = df_rfm[["customer_unique_id","cluster"]]

    df_final = (
        df 
        .join(
            df_rfm.set_index("customer_unique_id"),
            on="customer_unique_id",
            how="inner"
        )
    )

    return df_final

