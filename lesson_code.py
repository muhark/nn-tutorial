#! /usr/bin/python3
#
# @title:   Lesson Code Drafting
# @author:  Dr Musashi Jacobs-Harukawa, DDSS

# %% imports
from typing import Optional, List, Literal

from sklearn import datasets
from sklearn import svm, naive_bayes, tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
from torch import nn

sns.set_style('darkgrid')

%matplotlib inline

# %% dataset generation
# Generate data
X, y = datasets.make_moons(
    n_samples=500,
    # centers=2,
    random_state=0)

# Train-test split to simulate 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# %% visualise blobs
def visualise_blobs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray):
    with plt.rc_context({
    'figure.facecolor': '#2d2d2d',
    'axes.edgecolor': '#cccccc',
    'axes.facecolor': '#515151',
    'xtick.color': '#999999',
    'ytick.color': '#999999'}):
        fig, ax = plt.subplots(figsize=(15, 8))
        fig.suptitle("Two Blobs", color='#999999')
        ax.scatter(X_train[:,0], X_train[:,1], c=
            ['#4444ff' if i==0 else '#ff4444' for i in y_train],
            alpha=1, s=8)
        ax.scatter(X_test[:,0], X_test[:,1], c=
            ['#33e' if i==0 else '#e33' for i in y_test],
            ec='w', lw=0.05, alpha=1, s=4, zorder=0)
        return fig

visualise_blobs(X_train, y_train, X_test, y_test).show()

# %% classifier-> figure
def visualise_classification(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    h: float=0.02,
    show_scatter=True 
    ):
    # Set up mesh
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Generate predictions (need logic for handling torch vs skl)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) # Reshape predictions back to mesh

    # Create figure
    fig, ax = plt.subplots()
    # fig.suptitle("Two Blobs", color='#999999')
    ax.contourf(xx, yy, Z, cmap=plt.cm.Greens)
    if show_scatter:
        ax.scatter(X[:,0], X[:,1],
                c=['#4444ff' if i<=0 else '#ff4444' for i in y],
                alpha=0.5, s=4)
    ax.text(-0.5, 1.5, 'pred=0', ha='center')
    ax.text(2.5, -1.0, 'pred=1', ha='center')
    return fig

# %% model visualization
class ModelVisualization():
    def __init__(self,
                 model: nn.Module,
                 X: np.ndarray,
                 y: np.ndarray,
                 h: float=0.02) -> None:
        self.model = model
        self.X = X
        self.y = y
        self.h = h
        self.build_grid()
        self.fit_grid(model=self.model)
        self.build_figure(mode='plotly')

    def build_grid(self,
                   xmin: float,
                   xmax: float,
                   ymin: float,
                   ymax: float,
                   h: float) -> np.ndarray:
        if all(e is None for e in [xmin, xmax, ymin, ymax]):
            xmin = self.X[:,0].min()
            xmax = self.X[:,0].max()
            ymin = self.X[:,1].min()
            ymax = self.X[:,1].max()
        self.grid_x1, self.grid_x2 = np.meshgrid(np.arange(xmin-h, xmax+h, h),
                                                 np.arange(ymin-h, ymax+h, h))
        self.points = np.stack([grid_x1.reshape(-1), grid_x2.reshape(-1)], axis=1)
        return self.points
    
    def fit_grid(self, model: nn.Module) -> np.ndarray:
        preds = model.forward(self.points).detach().numpy()
        self.preds = preds.reshape(self.grid_x1.shape)
        return self.preds
    
    def build_figure(self,
                     mode = Literal['matplotlib', 'plotly'],
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
            fig.add_trace( # Predictions map
                go.Heatmap(x=np.arange(self.grid_x1.min(), self.grid_x1.max()),
                           y=np.arange(self.grid_x2.min(), self.grid_x2.max()),
                           z=self.preds,
                           hoverongaps=False))
            fig.add_trace( # Scatter of test data
                go.Scatter(x=self.X[:, 0],
                           y=self.X[:, 1],
                           mode='markers',
                           marker=dict(color=y_test),
                           text=['Class 1' if y==1 else 'Class 0' for y in y_test]))
        self.fig = fig




# %% Regressors make more sense in the beginnning -- classifier can be seen as subset
reg = LinearRegression()
reg.fit(X_train, y_train)

grid_x1, grid_x2 = np.meshgrid(np.arange(-1, 2.25, 0.25),
                               np.arange(-1, 2.25, 0.25))
points = np.stack([grid_x1.reshape(-1), grid_x2.reshape(-1)], axis=1)
predictions = reg.predict(points).reshape(grid_x1.shape)

# %% Matplotlib

fig, ax = plt.subplots()

ax.scatter(X_test[:, 0], X_test[:, 1], c=[ 'b' if i==0 else 'r' for i in y_test ])

# %% Plotly
import plotly.graph_objs as go

fig = go.Figure()
fig.add_trace(
    go.Heatmap(x=np.arange(-1, 2.25, 0.25),
               y=np.arange(-1, 2.25, 0.25),
               z=predictions,
               hoverongaps=False
               ))


fig.add_trace(
    go.Scatter(x=X_test[:, 0],
               y=X_test[:, 1],
               mode='markers',
               marker=dict(color=y_test),
               text=['Class 1' if y==1 else 'Class 0' for y in y_test]
))


# fig.add_scatter(x=X_test[:, 0], y=X_test[:, 1], customdata=y_test, mode='markers')

# fig = px.imshow(predictions, origin='lower')
fig.show()

# %% Fit SVM
linear_svm = svm.SVC(kernel = 'linear')
linear_svm.fit(X_train, y_train)
visualise_classification(clf=linear_svm, X=X_test, y=y_test)

# %% Radial SVM
radial_svm = svm.SVC(kernel = 'rbf')
radial_svm.fit(X_train, y_train)
visualise_classification(clf=radial_svm, X=X_test, y=y_test)

# %% Polynomial SVM
poly_svm = svm.SVC(kernel = 'poly', degree=9)
poly_svm.fit(X_train, y_train)
visualise_classification(clf=poly_svm, X=X_test, y=y_test)

# %% Bernoulli NB
nb_bernoulli = naive_bayes.BernoulliNB()
nb_bernoulli.fit(X_train, y_train)
visualise_classification(clf=nb_bernoulli, X=X_test, y=y_test, show_scatter=True)

# %% Gaussian NB
nb_gaussian = naive_bayes.GaussianNB()
nb_gaussian.fit(X_train, y_train)
visualise_classification(clf=nb_gaussian, X=X_test, y=y_test, show_scatter=True)

# %% Single tree
dtree_default = tree.DecisionTreeClassifier(random_state=0)
dtree_default.fit(X_train, y_train)
visualise_classification(clf=dtree_default, X=X_test, y=y_test, show_scatter=True)

# %% SINGLE NN STUFF

# %% Build a simple neural network from scratch
# Tensors hold values
weights = torch.tensor([[1, 2]])
bias = torch.tensor([[1]])
print(weights, bias)

# Simple y = WX' + b
inputs = torch.arange(10).reshape(-1, 2)
print(inputs)
preds = torch.matmul(weights, inputs.T) + bias
print(bias)


# %% Let's create a class to hold this for us
class SimpleLinearNeuralNetwork:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def predict(self, inputs):
        preds = torch.matmul(self.weights, inputs.T) + self.bias
        return preds

slnn = SimpleLinearNeuralNetwork(
    weights = torch.tensor([[1, 2]]),
    bias = torch.tensor([[1]]))

# Can check:
print(slnn.weights, slnn.bias)
print(slnn.predict(inputs))

# %% Let's get back to our data
# Better to randomly initialise, there's some literature about the specifics here
rng = np.random.default_rng(seed=0)
init_w = rng.normal(size=(1, 2))
init_b = rng.normal(size=(1, 1))

slnn = SimpleLinearNeuralNetwork(
    weights = torch.tensor(init_w),
    bias = torch.tensor(init_b))

# %% So we need to change this into a classifier
class SimpleLogisticNeuralNetwork:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
        preds = torch.matmul(self.weights, inputs.T) + self.bias
        logits = preds.logit()
        return logits

# %% Visualise Neural Network
def visualise_nn_regressor(
    nn,
    X: torch.tensor,
    y: torch.tensor,
    h: float=0.02,
    show_scatter=True 
    ):
    # Set up mesh
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    xx, yy = torch.tensor(xx), torch.tensor(yy)

    # Generate predictions (need logic for handling torch vs skl)
    Z = nn.forward(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape) # Reshape predictions back to mesh

    # Create figure
    fig, ax = plt.subplots()
    # fig.suptitle("Two Blobs", color='#999999')
    ax.contourf(xx, yy, Z, cmap=plt.cm.viridis_r)
    if show_scatter:
        ax.scatter(X[:,0], X[:,1],
                c=['#4444ff' if i<=0 else '#ff4444' for i in y],
                alpha=0.5, s=4)
    # ax.text(-0.5, 1.5, 'pred=0', ha='center')
    # ax.text(2.5, -1.0, 'pred=1', ha='center')
    return fig

# %%
slnn = SimpleLogisticNeuralNetwork(
    weights = torch.tensor(init_w),
    bias = torch.tensor(init_b))
visualise_nn_regressor(slnn, X_test, y_test)

# %%


# %% 
weights = torch.zeros(size=(2, 1), dtype=torch.float16, requires_grad=True)
biases = torch.zeros(size=(2, 1), dtype=torch.float16, requires_grad=True)

# %% Let's make a prediction off the first input
inputs = torch.tensor(X_train[:1, :], requires_grad=False)
preds = weights*inputs + biases # wx+b

# How wrong were we?
# Use squared loss
correct = torch.tensor(y_train[0]).reshape(1, -1)
loss = torch.sqrt(correct - preds)

# Now for the magic: backprop
print(weights, biases)
print(weights*inputs + biases)
print(preds, correct)

# %%

loss.backward()

print(weights, biases)
print(weights*inputs + biases)
print(preds, correct)

# %%
single_neuron = nn.Linear(
    in_features=2,
    out_features=1,
    bias=True)

# %%
list(single_neuron.named_parameters())

# %%
dir(single_neuron.get_parameter('weight'))


# %% GRAVEYARD

# %% Remembering how to do linalg in numpy/torch
# weights.dot(torch.tensor(X_train[:1, :]))
# init_w[:, 0] * X_train[0, :]
# np.array([1, 2]) * np.array([3, 4])
# torch.tensor([1, 2]) * torch.tensor([3, 4])
# np.array([[1, 2]]) * np.array([[3, 4], [5, 6]])
# np.array([[1, 2]]).dot(np.array([[3, 4], [5, 6]]))
# np.dot(np.array([1, 2]), np.array([[3, 4], [5, 6]]))
# torch.matmul(torch.tensor([1, 2]), torch.tensor([[3, 4], [5, 6]]))
# torch.tensor([1, 2]) * torch.tensor([[3, 4], [5, 6]])
# torch.tensor([1, 2]).matmul(torch.tensor([[3, 4], [5, 6], [7, 8]])) + torch.tensor([[1]])
# torch.tensor([[1, 2]]).shape, torch.tensor([[3, 4], [5, 6]]).shape, torch.tensor([[1]]).shape
# weights = torch.tensor([[1, 2]])
# bias = torch.tensor([[1]])
# inputs = torch.arange(10).reshape(-1, 2)
# print(inputs)
# weights.matmul(inputs.T) + bias
