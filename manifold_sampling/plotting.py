import numpy as np
import plotly.graph_objects as go

def plot_torus(r, R, precision=50):
    U = np.linspace(0, 2*np.pi, precision)
    V = np.linspace(0, 2*np.pi, precision)
    U, V = np.meshgrid(U, V)
    X = (R+r*np.cos(V))*np.cos(U)
    Y = (R+r*np.cos(V))*np.sin(U)
    Z = r*np.sin(V)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="viridis")])
    fig.update_layout(autosize=True)
    return fig

def plot_sphere(r, precision=50):
    U = np.linspace(0, 2*np.pi, precision)
    V = np.linspace(0, np.pi, precision)
    U, V = np.meshgrid(U, V)
    x = r * np.cos(U)*np.sin(V)
    y = r * np.sin(U)*np.sin(V)
    z = r * np.cos(V)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="viridis")])
    fig.update_layout(autosize=True)
    return fig