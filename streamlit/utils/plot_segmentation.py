import plotly.express as px
import plotly.graph_objects as go

def plot_segmentation(df_rfmv, x, y, z, border=True, marker_size=5):
    fig = px.scatter_3d(
        df_rfmv, 
        x=x, 
        y=y, 
        z=z,
        color='cluster',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    
    if border:
        fig.update_traces( 
            marker = dict(line=dict(width=0.01, color='black')),
            marker_size=marker_size
        )

    return fig

def plot_top_category_by_segmentation(df, x, y, color):
    df_to_plot = df.groupby([x,color]).size().reset_index(name=y).sort_values(by=[y], ascending=[False])
    
    fig = px.bar(df_to_plot, x=x, y=y, color=color)
    fig.update_layout(
        xaxis_title=dict(text='Segmentação', font=dict(size=16, color='#FFFFFF')),
        yaxis_title=dict(text='Contagem', font=dict(size=16, color='#FFFFFF')),
        legend_title="Categorias"
    )
    return fig

def plot_top_profitable_category_by_segmentation(df, x, y, color):
    df_to_plot = df.groupby([x,color]).agg({y: "sum"}).reset_index().sort_values(by=[y], ascending=[False])
    
    fig = px.bar(df_to_plot, x=x, y=y, color=color)
    fig.update_layout(
        xaxis_title=dict(text='Segmentação', font=dict(size=16, color='#FFFFFF')),
        yaxis_title=dict(text='Valor monetário', font=dict(size=16, color='#FFFFFF')),
        legend_title="Categorias"
    )
    return fig

def plot_boxplot(df):
    cols_to_show = ["recência","frequência","valor monetário"]
            
    fig = go.Figure()

    for col in df[cols_to_show]:
        fig.add_trace(go.Box(x=df["segmentação"], y=df[col].values, name=df[col].name))
    
    fig.update_layout(
        xaxis_title='Segmentação',
        boxmode='group'
    )

    return fig

