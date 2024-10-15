import streamlit as st
import pandas as pd
import numpy as np

from algorithms import kmeans, agglomerative_clustering, bisecting_kmeans
from utils import rfm, elbow_method, plot_segmentation, transformation_functions
from validation import scores, ranking

st.set_page_config(
    page_title='Segmentação de Clientes', 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.session_state.clear()

st.title('Segmentação de Clientes')

df_model = pd.read_csv("./data/model.csv")

st.download_button(
    label="Baixar modelo",
    data=df_model.to_csv(index=False).encode('utf-8'),
    file_name="segmentacao_de_clientes.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader(
    "Informações de compras dos clientes", 
    type=["csv", "xlsx"], 
    accept_multiple_files=False,
    help="Veja o arquivo de exemplo disponível."
)

if(uploaded_file and "file" not in st.session_state):
    st.session_state.file = uploaded_file

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
        total_sales = df["monetary"].sum()
        sales_qty = len(df.index)
        product_qty = len(df["product_id"].unique())
        customer_qty = len(df["customer_unique_id"].unique())
        category_qty = len(df["product_category_name"].unique())
        avg_ticket = total_sales / sales_qty

        with col[0]:
            st.metric("Valor total da venda", transformation_functions.get_formatted_value("R$ {:,.2f}", total_sales))
            st.metric("Ticket médio", transformation_functions.get_formatted_value("R$ {:,.2f}", avg_ticket))
            
            st.metric("Quantidade de clientes", transformation_functions.get_formatted_value("{:,}",customer_qty))

            st.metric("Quantidade de itens vendidos", transformation_functions.get_formatted_value("{:,}", product_qty))
            st.metric("Quantidade de categorias vendidas", transformation_functions.get_formatted_value("{:,}",category_qty))

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

        if(customer_qty >= 10):
            if("best_number" not in st.session_state):
                st.session_state.best_number = True
                with st.status("Criando segmentação", expanded=True):
                    st.write("Calculando as métricas por cliente...")
                    df_rfm = rfm.get_rfm(df)
                    df_rfm_std = rfm.get_rfm_std(df_rfm)

                    all_columns = df_rfm_std.columns[1:]

                    
                    st.write("Calculando o número ótimo de segmentações...")
                    
                    if(customer_qty >= 25000):
                        inertias, fit_time = elbow_method.get_inertias(df_rfm_std, all_columns)
                        n_clusters, distances = elbow_method.get_optimal_number_of_clusters(inertias) 
                        alg = None                   
                    
                    else:
                        algorithms = ["kmeans", "agg", "bisect_kmeans"]
                        posicoes_final = pd.DataFrame()

                        for alg in algorithms:
                            scores_alg = scores.get_scores_from_alg(df_rfm_std, alg)
                            ranking_alg = ranking.get_ranking(scores_alg)
                            n_scores_alg = ranking.get_scores_from_n(ranking_alg)
                            n_scores_alg["alg"] = alg

                            posicoes_final = pd.concat([posicoes_final, n_scores_alg])[['n', 'silhouette_score', 'ch_score', 'db_score', 'alg']]

                        ranking_final = ranking.get_ranking(posicoes_final)
                        n_clusters = ranking.get_best_n(ranking_final)
                        alg = ranking.get_best_alg(ranking_final)

                    st.session_state.n_clusters = n_clusters
                    st.session_state.alg = alg


                    st.write("Segmentando seus clientes...")            

            st.title("Visualização das segmentações")            
            
            n_clusters = st.session_state.n_clusters 
            alg = st.session_state.alg 

            col1, _ = st.columns(2)
            with col1:
                current_n_clusters = st.number_input(label="Alterar número de segmentações", value=n_clusters, min_value=2, max_value=10, step=1)
            st.write(f"Seus clientes foram divididos em {str(current_n_clusters)} segmentações.")

            if(customer_qty >= 25000):
                labels = kmeans.apply_kmeans(df_rfm_std, current_n_clusters, all_columns).labels_
                df_rfm["cluster"] = labels 

            else:
                if(alg == "kmeans"): 
                    labels = kmeans.apply_kmeans(df_rfm, current_n_clusters, all_columns).labels_
                elif(alg == "agg"): 
                    labels = agglomerative_clustering.apply_agglomerative_clustering(df_rfm, current_n_clusters, all_columns).labels_
                else: 
                    labels = bisecting_kmeans.apply_bisecting_kmeans(df_rfm, current_n_clusters, all_columns).labels_

                df_rfm["cluster"] = labels 

            df_to_plot = df_rfm.rename({"recency": "R", "frequency": "F", "monetary": "M"}, axis=1)
            
            fig = plot_segmentation.plot_segmentation(df_to_plot, "R", "F", "M",border=False, marker_size=2)
            st.plotly_chart(fig, use_container_width=True)

            df_final = rfm.get_customer_segmentation(df, df_to_plot)
            df_final = df_final.rename(
                {
                    "monetary": "Valor monetário",
                    "cluster": "Segmentação",
                    "product_category_name" : "Categoria do Produto"
                }, 
                axis=1
            )

            st.divider()
            st.title("Categorias mais compradas por cada segmentação (valor monetário)")

            fig = plot_segmentation.plot_top_profitable_category_by_segmentation(df_final, x="Segmentação", y="Valor monetário", color="Categoria do Produto")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.title("Categorias mais compradas por cada segmentação (quantidade de vendas)")
            fig = plot_segmentation.plot_top_category_by_segmentation(df_final, x="Segmentação", y="Contagem", color="Categoria do Produto")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.title("Estatísticas descritivas de cada segmentação")
            df_rfm.columns = ["id_cliente","recência","frequência","valor monetário","segmentação"]
            
            fig = plot_segmentation.plot_boxplot(df_rfm)
            st.plotly_chart(fig, use_container_width=True)

            df_to_describe = transformation_functions.get_df_to_describe(df_rfm).T
            btn_stats = st.download_button(
                label="Salvar as estatísticas descritivas como CSV",
                data=df_to_describe.to_csv().encode('utf-8'),
                file_name="estatisticas_descritivas.csv",
                mime="text/csv",
            )

            btn_rfm = st.download_button(
                label="Salvar as informações de RFM como CSV",
                data=df_rfm.to_csv(index=False).encode('utf-8'),
                file_name="segmentacao_de_clientes.csv",
                mime="text/csv",
            )

        else:
            st.error("Insira dados de pelo menos 10 clientes para iniciar a segmentação.")