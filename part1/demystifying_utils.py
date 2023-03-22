#! /usr/bin/python
#
# @author: Dr Musashi Jacobs-Harukawa, DDSS Princeton
# @title: demystifying_utils

"""
Script containing utils used in the accompany notebook.

In general I've put things in here because being able to code them is not relevant to the point of the coding exercise.
"""

# imports
import torch
from torch import nn
import plotly.graph_objs as go


# Simple plot stuff
def visualize_linear_model(X_, y_, wOLS, bOLS) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_[:,0],
                            y=y_[:,0],
                            mode='markers',
                            marker_color='blue',
                            hovertemplate="%{x:.3g}, %{y:.3g}",
                            name='Data'))
    fig.add_trace(go.Scatter(x=[X_.min(), X_.max()],
                            y=[X_.min() * wOLS + bOLS, X_.max() * wOLS + bOLS],
                            mode='lines',
                            name=f'OLS: {wOLS:.3g}x+{bOLS:.3g}',
                            # visible='legendonly',
                            hovertext=f"y = ({wOLS:.3g})x + {bOLS:.3g}",
                            line=dict(color='black', dash='dash')))
    fig.update_layout(title='Comparing True Relationship to Best Model on Data',
                    xaxis_title='X',
                    yaxis_title='Y')
    return fig


