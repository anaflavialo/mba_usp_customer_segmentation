import plotly.express as px

def plot_segmentation(df_rfmv, x, y, z):
    fig = px.scatter_3d(
        df_rfmv, 
        x=x, 
        y=y, 
        z=z,
        color='segmentation'
    )

    return fig