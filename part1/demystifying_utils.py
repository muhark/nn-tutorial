#! /usr/bin/python
#
# @author: Dr Musashi Jacobs-Harukawa, DDSS Princeton
# @title: demystifying_utils

"""
Script containing utils used in the accompany notebook.

In general I've put things in here because being able to code them is not relevant to the point of the coding exercise.
"""

# imports
# Some utilities
from tqdm import tqdm
from typing import Union, Optional, Literal

# Tools for creating toy datasets
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Visualization Tools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px

# Core libraries for deep learning and numerical computing
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# Simple plot stuff
def visualize_linear_model(X_, y_, wOLS, bOLS, write_to_html=False) -> go.Figure:
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
                            visible='legendonly',
                            hovertext=f"y = ({wOLS:.3g})x + {bOLS:.3g}",
                            line=dict(color='black', dash='dash')))
    fig.update_layout(title='Comparing True Relationship to Best Model on Data',
                    xaxis_title='X',
                    yaxis_title='Y')
    if write_to_html:
        fig.write_html(write_to_html)
    return fig



def generate_2d_data(problem: Literal['XOR', 'Moons']
                    ) -> tuple[np.ndarray, np.ndarray]:
    if problem=='XOR':
        N = 1000
        x1 = np.concatenate([rng.uniform(0, 100, N//4),
                            rng.uniform(-100, 0, N//4),
                            rng.uniform(0, 100, N//4),
                            rng.uniform(-100, 0, N//4)])
        x2 = np.concatenate([rng.uniform(-100, 0, N//4),
                            rng.uniform(0, 100, N//4),
                            rng.uniform(0, 100, N//4),
                            rng.uniform(-100, 0, N//4)])
        X = np.vstack([x1, x2]).T
        y = np.concatenate([np.zeros(shape=N//2), np.ones(shape=N//2)])
    elif problem=='Moons':
        X, y = datasets.make_moons(
            n_samples=500,
            random_state=0)
    return X, y
X, y = generate_2d_data('Moons')

class ModelVisualization():
    def __init__(self,
                 model: nn.Module,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 h: float=1.0) -> None:
        self.model = model
        self.X = X_test
        self.y = y_test
        self.h = h
        self.build_grid(
            self.X[:,0].min(),
            self.X[:,0].max(),
            self.X[:,1].min(),
            self.X[:,1].max(),
            self.h)
        self.fit_grid()
        self.init_figure(mode='plotly')

    def build_grid(self,
                   xmin: float,
                   xmax: float,
                   ymin: float,
                   ymax: float,
                   h: float=1.0) -> np.ndarray:
        if all(e is None for e in [xmin, xmax, ymin, ymax]):
            xmin = self.X[:,0].min()
            xmax = self.X[:,0].max()
            ymin = self.X[:,1].min()
            ymax = self.X[:,1].max()
        self.grid_x1, self.grid_x2 = np.meshgrid(np.arange(xmin, xmax+h, h),
                                                 np.arange(ymin, ymax+h, h))
        self.points = np.stack([self.grid_x1.reshape(-1),
                                self.grid_x2.reshape(-1)],
                                axis=1)
        return self.points
    
    def fit_grid(self) -> np.ndarray:
        inputs = torch.tensor(self.points).float()
        preds = self.model(inputs).detach().numpy()
        self.preds = preds.reshape(self.grid_x1.shape)
        return self.preds
    
    def generate_pred_heatmap(self, step: int=0):
        # colors = px.colors.sequential.s
        # intervals = np.linspace(self.preds.min(), self.preds.max(), len(colors))
        # cs =  [[intervals[n], colors[n]]
        #         for s in [[i, i+1] for i in range(len(colors))]
        #         for n in s if n < len(colors)]
        # intervals = np.linspace(0, 1, len(colors))
        # cs = [[intervals[n], colors[n]] for n in range(len(colors))]
        hm = go.Heatmap(x=np.arange(self.grid_x1.min(), self.grid_x1.max()+self.h, self.h),
                        y=np.arange(self.grid_x2.min(), self.grid_x2.max()+self.h, self.h),
                        z=self.preds,
                        name='Preds: ' + str(step),
                        colorscale='Spectral',
                        # colorbar=dict(
                            # tick0=0,
                            # dtick=self.preds.ptp()//10,
                            # title='$\hat{y}$'),
                        hovertemplate="X1: %{x:.3g}, X2: %{y:.3g}: Pred: %{z:.3g}",
                        hoverongaps=False)
        return hm
    
    def init_figure(self, mode = Literal['matplotlib', 'plotly'],
                    **kwargs) -> Union[go.Figure, plt.Figure]:
        if mode == 'matplotlib':
            fig, ax = plt.subplots()
            ax.imshow(self.preds,
                      origin='lower',
                      extent=(self.grid_x1.min(), self.grid_x1.max(),
                              self.grid_x2.min(), self.grid_x2.max()))
            ax.scatter(self.X[:,0], self.X[:,1],
                    c=['#4444ff' if i<=0 else '#ff4444' for i in y],
                    alpha=0.5, s=4)
        elif mode == 'plotly':
            fig = go.Figure()
            fig.add_trace( # Scatter of test data
                go.Scattergl(
                    x=self.X[:, 0],
                    y=self.X[:, 1],
                    mode='markers',
                    name='Test Set',
                    marker=dict(
                        color=['blue' if i==1 else 'red' for i in self.y],
                        line=dict(width=1)),
                    text=self.y))
            fig.add_trace( # Predictions map
                self.generate_pred_heatmap())
            # Slider configuration
            self.total_steps = 1
            self.steps = []
            step = dict(method="update",
                        args=[{"visible": [False] * len(fig.data)},
                              {"title": "Step: " + str(self.total_steps)}])
            step["args"][0]["visible"][0] = True # Make Scatter visible
            step["args"][0]["visible"][-1] = True # Set step visible
            self.steps.append(step)
            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Updates: "},
                pad={"t": self.total_steps},
                steps=self.steps)]
            
            fig.update_layout(title='Model Predictions',
                              xaxis_title='X1',
                              yaxis_title='X2',
                              legend_title_text='Predicted Value',
                              sliders=sliders)
        self.mode = mode
        self.fig = fig
        return self.fig

    def update_figure(self, model: nn.Module) -> Union[go.Figure, plt.Figure]:
        self.fit_grid()
        self.total_steps += 1
        self.fig.add_trace(self.generate_pred_heatmap(self.total_steps))
        # Update all previous steps
        for i, step in enumerate(self.steps):
            step['args'][0]['visible'] = [False] * len(self.fig.data)
            step["args"][0]["visible"][0] = True
            step["args"][0]["visible"][i+1] = True
        new_step = dict(method="update",
                        args=[{"visible": [False] * len(self.fig.data)},
                              {"title": "Step: " + str(self.total_steps)}])
        new_step["args"][0]["visible"][0] = True # Make Scatter visible
        new_step["args"][0]["visible"][-1] = True # Set update visible
        self.steps.append(new_step)
        sliders = [dict(
            active=self.total_steps-1,
            currentvalue={"prefix": "Updates: "},
            pad={"t": self.total_steps},
            steps=self.steps)]
        self.fig.update_layout(sliders=sliders)
        
        return self.fig