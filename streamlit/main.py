import streamlit as st
import pandas as pd
import numpy as np
from utils import rfmv, elbow_method, kmeans, plot_segmentation

st.set_page_config(page_title='Customer Segmentation', layout="wide")

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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Valor total da venda (R$)", df["monetary"].sum())

    with col2:
        st.metric("Quantidade de clientes", len(df["customer_unique_id"].unique()))

    with col3:
        st.metric("Quantidade de itens vendidos", len(df["product_id"].unique()))

    with col4:
        st.metric("Quantidade de categorias vendidas", len(df["product_category_name"].unique()))

    with st.status("Criando segmentação", expanded=True):
        st.write("Calculando as métricas por cliente...")
        df_rfmv = rfmv.get_rfmv(df)
        df_rfmv_norm = rfmv.get_rfmv_normalized(df)

        st.write("Calculando o número ótimo de segmentações...")
        inertias = elbow_method.get_inertias(df_rfmv_norm)
        n_clusters = elbow_method.get_optimal_number_of_clusters(inertias)

        st.write("Segmentando seus clientes...")
        clusters = kmeans.apply_kmeans(df_rfmv_norm, n_clusters)
        df_rfmv["segmentation"] = clusters 

    st.write("Quantidade de clientes por segmentação")
    st.write(df_rfmv.groupby("segmentation").agg({"customer_unique_id": "count"}).rename({"customer_unique_id": "count"}, axis=1).head())

    col1, col2, col3 = st.columns(3)

    with col1:
        cols_to_filter = set(["recency", "frequency", "monetary", "product_variety", "category_variety"])
        sel1 = st.selectbox(label="Escolha uma coluna", options=cols_to_filter)

    with col2:
        cols_to_filter = cols_to_filter - set([sel1])
        sel2 = st.selectbox(label="Escolha uma coluna", options=cols_to_filter)
    
    with col3:
        cols_to_filter = cols_to_filter - set([sel2])
        sel3 = st.selectbox(label="Escolha uma coluna", options=cols_to_filter)

    fig = plot_segmentation.plot_segmentation(df_rfmv, sel1, sel2, sel3)
    st.plotly_chart(fig, use_container_width=True)