import os
import json
import numpy as np
import natsort
import torch


def overwrite_params(params_input, default_params) -> dict:
    ## Load master params
    if isinstance(params_input, os.PathLike):
        with open(params_input) as param_handle:
            master_params = json.load(param_handle)
    elif isinstance(params_input, dict):
        master_params = params_input

    ## Overwrite default params with master params
    for key in master_params:
        default_params[key] = master_params[key]

    return default_params


def center_mean(
    X: np.ndarray,
    axis: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    X_mean = np.nanmean(X, axis=axis, keepdims=True)

    if verbose:
        print(f"Given matrix shape: {X.shape}")
        print(f"Subtract matrix shape: {X_mean.shape}")

    return X - X_mean


def compute_col_angle(
    X: np.ndarray,
    Y: np.ndarray,
    mean_center: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    if mean_center:
        X = center_mean(X, verbose=verbose)
        Y = center_mean(Y, verbose=verbose)

    cossim = np.sum(
        (X * Y)
        / (
            np.linalg.norm(X, axis=1, keepdims=True)
            * np.linalg.norm(Y, axis=1, keepdims=True)
        ),
        axis=0,
    )

    ## Change to angle (metric space) and take the mean of angles
    cossim = np.clip(cossim, -1, 1)  # This line is necessary to prevent error
    angles = np.arccos(cossim)

    return angles


def weight_transform(
    X: np.ndarray,
    weights: np.ndarray = None,
    weight_method: str = "uniform",
) -> np.ndarray:
    if weight_method == "uniform":
        return X
    elif weight_method == "EV":
        feature_variance = np.var(X, axis=0) / np.sum(np.var(X, axis=0))
        return X * feature_variance
    elif weight_method == "custom":
        return X * weights


def plotly_imshow(
    image: np.ndarray,
    autosize: bool = False,
    width: int = 1000,
    height: int = 1000,
    cmap: str = "Viridis",
    zmin: float = None,
    zmax: float = None,
    y_flip: bool = True,
):
    import plotly.graph_objects as go

    # Create the figure
    fig = go.Figure(data=go.Heatmap(z=image, colorscale=cmap, zmin=zmin, zmax=zmax))

    # Flip y-axis
    fig.update_yaxes(autorange="reversed") if y_flip else None

    # Adjust aspect ratio to make the plot square and define the output size
    fig.update_layout(
        autosize=autosize,
        width=width,
        height=height,
        margin=dict(
            l=50,  # left margin
            r=50,  # right margin
            b=100,  # bottom margin
            t=100,  # top margin
            pad=10,
        ),
    )

    return fig
