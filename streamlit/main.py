import streamlit as st
import pandas as pd
import numpy as np
from utils import rfmv, elbow_method, kmeans

st.title('Customer Segmentation')

uploaded_file = st.file_uploader(
    "Informações de compras dos clientes", 
    type=["csv", "xlsx"], 
    accept_multiple_files=False,
    help="Veja o arquivo de exemplo disponível."
)

if(uploaded_file):
    file_type = uploaded_file.type
    if(file_type == "text/csv"): df = pd.read_csv(uploaded_file)
    else: df = pd.read_excel(uploaded_file)

    df["order_purchase"] = pd.to_datetime(df["order_purchase"]).dt.date

    df_rfmv = rfmv.get_rfmv(df)
    df_rfmv_norm = rfmv.get_rfmv_normalized(df)

    inertias = elbow_method.get_inertias(df_rfmv_norm)
    n_clusters = elbow_method.get_optimal_number_of_clusters(inertias)

    clusters = kmeans.apply_kmeans(df_rfmv_norm, n_clusters)

    df_rfmv["segmentation"] = clusters 

    st.write(df_rfmv.head())

    st.write("Quantidade de clientes por segmentação")
    st.write(df_rfmv.groupby("segmentation").agg({"customer_unique_id": "count"}).rename({"customer_unique_id": "count"}, axis=1).head())