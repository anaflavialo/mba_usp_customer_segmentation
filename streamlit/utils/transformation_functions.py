import pandas as pd

def get_renamed_dataset(df):
    col_names_old = df.columns
    col_names_new = [
        "order_id",
        "customer_unique_id",
        "order_purchase",
        "product_id",
        "product_category_name",
        "monetary",
    ]

    if(len(col_names_old) == len(col_names_new)):
        dict_renamed = dict(zip(col_names_old, col_names_new))
        df_renamed = df.rename(dict_renamed, axis=1)

        return df_renamed
    else: return -1

def convert_to_date(df):
    df["order_purchase"] = pd.to_datetime(df["order_purchase"]).dt.date
    return df

def get_formatted_value(format_string, value):
    return format_string.format(value).replace(",", "_").replace(".", ",").replace("_", ".")

def get_grouped_by_monetary(df):
    df_grouped = df.groupby("product_category_name").agg({"monetary": "sum"}).sort_values(by="monetary", ascending=False)
    df_grouped["total"] = df["monetary"].sum()
    df_grouped["percent"] = 100*df_grouped["monetary"] / df_grouped["total"]
    df_grouped["monetary"] = df_grouped["monetary"].apply(lambda x: "R$ {:,.2f}".format(x).replace(",", "_").replace(".", ",").replace("_", "."))

    return df_grouped

def get_grouped_by_amount(df):
    df_grouped = df["product_category_name"].value_counts(sort=True).reset_index()
    df_grouped["total"] = df["product_category_name"].count()
    df_grouped["percent"] = 100*df_grouped["count"] / df_grouped["total"]
    df_grouped["count"] = df_grouped["count"].apply(lambda x: "{:,}".format(x).replace(",", "_").replace(".", ",").replace("_", "."))

    return df_grouped