import numpy as np
import plotly.graph_objects as go

def plot_torus(r, R, p=None, precision=50):
    U = np.linspace(0, 2*np.pi, precision)
    V = np.linspace(0, 2*np.pi, precision)
    U, V = np.meshgrid(U, V)
    X = (R+r*np.cos(V))*np.cos(U)
    Y = (R+r*np.cos(V))*np.sin(U)
    Z = r*np.sin(V)

    if p is None:
        color = None
    else:
        color = p(np.stack([X, Y, Z]))
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="viridis", surfacecolor=color)])
    fig.update_layout(autosize=True)
    return fig

def plot_sphere(r, p=None, precision=50):
    U = np.linspace(0, 2*np.pi, precision)
    V = np.linspace(0, np.pi, precision)
    U, V = np.meshgrid(U, V)
    x = r * np.cos(U)*np.sin(V)
    y = r * np.sin(U)*np.sin(V)
    z = r * np.cos(V)

    if p is None:
        color = None
    else:
        color = p(np.stack([x, y, z]))

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="viridis", surfacecolor=color)])
    fig.update_layout(autosize=True)
    return fig

def plot_ellipsoid(x_factor, y_factor, z_factor, p=None, precision=50):
    U = np.linspace(0, 2*np.pi, precision)
    V = np.linspace(0, np.pi, precision)
    U, V = np.meshgrid(U, V)
    x = np.cos(U)*np.sin(V) * x_factor
    y = np.sin(U)*np.sin(V) * y_factor
    z = np.cos(V) * z_factor

    if p is None:
        color = None
    else:
        color = p(np.stack([x, y, z]))

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="viridis", surfacecolor=color)])
    fig.update_layout(autosize=True)
    return fig