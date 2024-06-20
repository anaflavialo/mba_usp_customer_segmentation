import streamlit as st
import pandas as pd
import numpy as np
from utils import rfmv, elbow_method, kmeans, plot_segmentation, transformation_functions
from components import sidebar

st.set_page_config(
    page_title='Segmentação de Clientes', 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title('Segmentação de Clientes')

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

    df = transformation_functions.get_renamed_dataset(df)
    if(not isinstance(df, pd.DataFrame)): st.error("O número de colunas do arquivo inserido não corresponde com a quantidade de colunas do arquivo modelo.", icon="🚨")
    else:
        df = transformation_functions.convert_to_date(df)

        with st.sidebar:
            max_date = df["order_purchase"].max()
            min_date = df["order_purchase"].min()

            date_filter = st.date_input("Selecione uma data para visualizar os dados", (min_date,max_date), min_value=min_date, max_value=max_date)
            if(len(date_filter) == 2):
                df = df[(df.order_purchase >= date_filter[0]) & (df.order_purchase <= date_filter[1])]

        col = st.columns(3, gap="medium")
        with col[0]:
            st.metric("Valor total da venda", transformation_functions.get_formatted_value("R$ {:,.2f}", df["monetary"].sum()))
            st.metric("Quantidade de itens vendidos", transformation_functions.get_formatted_value("{:,}", len(df["product_id"].unique())))
            st.metric("Quantidade de clientes", transformation_functions.get_formatted_value("{:,}",len(df["customer_unique_id"].unique())))
            st.metric("Quantidade de categorias vendidas", transformation_functions.get_formatted_value("{:,}",len(df["product_category_name"].unique())))

        with col[1]:
            st.markdown('#### Top Categorias por Valor')

            df_grouped = transformation_functions.get_grouped_by_monetary(df)

            st.dataframe(
                df_grouped,
                column_order=("product_category_name","monetary","percent"),
                width=None,
                hide_index=True,
                column_config={
                    "product_category_name": st.column_config.TextColumn("Categoria"),
                    "monetary": st.column_config.TextColumn("Valor"),
                    "percent": st.column_config.ProgressColumn(
                        "%",
                        format="%.2f",
                        min_value=0,
                        max_value=100
                    )
                }
            )

        with col[2]:
            st.markdown('#### Top Categorias por Quantidade')

            df_grouped = transformation_functions.get_grouped_by_amount(df)

            st.dataframe(
                df_grouped,
                column_order=("product_category_name","count","percent"),
                width=None,
                hide_index=True,
                column_config={
                    "product_category_name": st.column_config.TextColumn("Categoria"),
                    "count": st.column_config.TextColumn("Quantidade"),
                    "percent": st.column_config.ProgressColumn(
                        "%",
                        format="%.2f",
                        min_value=0,
                        max_value=100
                    )
                }
            )

        
        with st.status("Criando segmentação", expanded=True):
            st.write("Calculando as métricas por cliente...")
            df_rfmv = rfmv.get_rfmv(df)
            df_rfmv_norm = rfmv.get_rfmv_std(df)

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

        #fig = plot_segmentation.plot_segmentation(df_rfmv, sel1, sel2, sel3)
        #st.plotly_chart(fig, use_container_width=True)