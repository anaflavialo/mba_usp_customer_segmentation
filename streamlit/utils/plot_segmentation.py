import plotly.express as px

def plot_segmentation(df_rfm, x, y, z):
    fig = px.scatter_3d(
        df_rfm, 
        x=x, 
        y=y, 
        z=z,
        color='segmentation'
    )

    return fig

def plot_top_category_by_segmentation(df, x, y, color):
    df_to_plot = df.groupby(["segmentation","product_category_name"]).size().reset_index(name="count").sort_values(by=["count"], ascending=[False])
    
    fig = px.bar(df_to_plot, x=x, y=y, color=color)
    fig.update_layout(
        xaxis_title=dict(text='Segmentação', font=dict(size=16, color='#FFFFFF')),
        yaxis_title=dict(text='Contagem', font=dict(size=16, color='#FFFFFF')),
        legend_title="Categorias"
    )
    return fig

def plot_top_profitable_category_by_segmentation(df, x, y, color):
    df_to_plot = df.groupby(["segmentation","product_category_name"]).agg({"monetary": "sum"}).reset_index().sort_values(by=["monetary"], ascending=[False])
    
    fig = px.bar(df_to_plot, x=x, y=y, color=color)
    fig.update_layout(
        xaxis_title=dict(text='Segmentação', font=dict(size=16, color='#FFFFFF')),
        yaxis_title=dict(text='Valor monetário', font=dict(size=16, color='#FFFFFF')),
        legend_title="Categorias"
    )
    return fig

def plot_boxplot(df, col):
    fig = px.box(df, x="segmentação", y=col)
    return fig

