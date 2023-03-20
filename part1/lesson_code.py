#! /usr/bin/python3
#
# @title:   Lesson Code Drafting
# @author:  Dr Musashi Jacobs-Harukawa, DDSS

# %% imports
# Some utilities
from tqdm import tqdm
from typing import Union, Literal

# Tools for creating toy datasets
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Visualization Tools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

# Core libraries for deep learning and numerical computing
import numpy as np
import torch
from torch import nn

# Visualization defaults
sns.set_style('darkgrid')
%matplotlib inline

## %% Generate linear regression data
# Generate data
X_, y_, coef_ = datasets.make_regression(n_samples=20,
                                  n_features=1,
                                  bias=10,
                                  noise=3,
                                  random_state=0,
                                  coef=True)
# %% OLS Model
wOLS, bOLS = (LinearRegression().fit(X_, y_).coef_[0],
              LinearRegression().fit(X_, y_).intercept_)


# %% Plotly it
# Points for regression line
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_[:,0],
                         y=y_,
                         mode='markers',
                         marker_color='blue',
                         hovertemplate="%{x:.3g}, %{y:.3g}",
                         name='Observations'))
fig.add_trace(go.Scatter(x=[X_.min(), X_.max()],
                         y=[X_.min() * wOLS + bOLS, X_.max() * wOLS + bOLS],
                         mode='lines',
                         name='OLS Model',
                         visible='legendonly',
                         hovertext=f"y = ({wOLS:.3g})x + {bOLS:.3g}",
                         line=dict(color='black', dash='dash')))
fig.add_trace(go.Scatter(x=[X_.min(), X_.max()],
                         y=[X_.min() * coef_ + 10, X_.max() * coef_ + 10],
                         mode='lines',
                         name='True Model',
                         visible='legendonly',
                         hovertext=f"y = ({coef_:.3g})x + 10",
                         line=dict(color='rgba(0, 0, 0, 0.5)')))
fig.data = fig.data[::-1]  # Move the True Model trace under the points
fig.update_layout(title='Sample Linear Regression Data',
                  xaxis_title='X',
                  yaxis_title='Y')
fig

# %% How do we learn this model from the data?
# Let's convert our data to PyTorch tensors
X_, y_ = torch.tensor(X_).float(), torch.tensor(y_).float()

# %% Fit model y = wx + b: learn w and b
# Creating tensors to store values of w and b
# Initializations as "empty"
w = torch.empty(1, 1, requires_grad=True)   # 1x1 uninitialised weight matrix
b = torch.empty(1, 1, requires_grad=True)   # 1x1 uninitialised bias matrix
# We randomize the values of these coefficients
torch.manual_seed(0)
with torch.no_grad():                       # Will explain this later
    nn.init.normal_(w, mean=0, std=0.5)     # Fill with random values
    nn.init.normal_(b, mean=0, std=0.5)
print(w, b)

# %% What does this predict?
fig.add_trace(go.Scatter(x=[X_.min(), X_.max()],
                         y=[(X_.min() * w + b).detach().squeeze(), (X_.max() * w + b).detach().squeeze()],
                         mode='lines',
                         name='Random Guess',
                         line=dict(color='green', dash='dash')))
fig

# %% Quick intro to gradient descent
# How do we figure out if a line is a good fit to some data?
# Quick calculus reminder:
# Given the model y = wx + b, we want to find the values of
# w and b that minimize some error function.

# Let's begin one point
x0 = X_[:2]
y0 = y_[:2]

# Make prediction
yhat_ = x0 * w + b

# Begin by defining the loss function: squared loss 
loss = ((y0 - yhat_).pow(2)).mean() # Could also do sum, but with fixed batch size these are equivalent

# Let's visualize the loss versus the parameters
# First for w:
wvals = []
for wval in np.linspace(-10, 30, 100):
    yhat_ = x0 * wval + b
    loss = ((y_ - yhat_).pow(2)).mean()
    wvals.append([wval, loss.item()])

# %% 
fig = go.Figure()
fig.add_trace(go.Scatter(x=[witem[0] for witem in wvals],
                         y=[witem[1] for witem in wvals],
                         name='w',
                         mode='markers+lines'))
fig.update_layout(title='Sum Squared Loss vs Weight Parameter',
                  xaxis_title='Parameter',
                  yaxis_title='Loss')
fig.show()
# Intuition--loss is minimized for a particular value of the parameters w

# %% Generate prediction
preds = torch.matmul(w, inputs.T) + b
print(preds)

# %% How far off were we?
loss = (labels - preds).pow(2) # L = (y' - y)^2
print(loss)

# Differentiate the loss w.r.t. parameters (w, b)
if w.grad is not None: 
    w.grad -= w.grad; b.grad -= b.grad
loss.backward()
print(w.grad, b.grad)  # dL/dw, dL/db
print(w - w.grad)      # w - dL/dw

labels - torch.matmul(w, inputs.T) + b
labels - torch.matmul(w - w.grad, inputs.T) + (b - b.grad)
labels

# %% COME BACK TO HERE


# %% MOONS PROBLEM
##### WORKING ON THE MOONS PROBLEM
##### SHOWS CHALLENGES IN 2 DIMENSIONS
# %% dataset generation
# Generate data
X, y = datasets.make_moons(
    n_samples=500,
    # centers=2,
    random_state=0)

# Train-test split to simulate 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# %% Visualize dataset, train/test split
fig = go.Figure()
fig.add_trace( # Visualize train data
    go.Scatter(x=X_train[:, 0],
               y=X_train[:, 1], 
                mode='markers',
                name='Train Set',
                hovertemplate="%{x:.3g}, %{y:.3g}",
                marker=dict(size=1, color=y_train, colorscale='bluered'),
                text=['Class 1' if y==1 else 'Class 0' for y in y_train]))
fig.add_trace( # Visualize test data
    go.Scatter(x=X_test[:, 0],
               y=X_test[:, 1],
               mode='markers',
               name='Test Set',
               hovertemplate="%{x:.3g}, %{y:.3g}",
               marker=dict(size=5, color=y_test, colorscale='bluered'),
               text=['Class 1' if y==1 else 'Class 0' for y in y_train]))
fig.update_layout(title='\"Moons\" Classification Problem in Two Dimensions',
                  xaxis_title='X1',
                  yaxis_title='X2',
                  showlegend=False)
fig

# %% Model Visualization
class ModelVisualization():
    def __init__(self,
                 model: nn.Module,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 h: float=0.02) -> None:
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
        self.fit_grid(model=self.model)
        self.build_figure(mode='plotly')

    def build_grid(self,
                   xmin: float,
                   xmax: float,
                   ymin: float,
                   ymax: float,
                   h: float=0.02) -> np.ndarray:
        if all(e is None for e in [xmin, xmax, ymin, ymax]):
            xmin = self.X[:,0].min()
            xmax = self.X[:,0].max()
            ymin = self.X[:,1].min()
            ymax = self.X[:,1].max()
        self.grid_x1, self.grid_x2 = np.meshgrid(np.arange(xmin-h, xmax+h, h),
                                                 np.arange(ymin-h, ymax+h, h))
        self.points = np.stack([self.grid_x1.reshape(-1),
                                self.grid_x2.reshape(-1)],
                                axis=1)
        return self.points
    
    def fit_grid(self, model: nn.Module) -> np.ndarray:
        inputs = torch.tensor(self.points).float()
        preds = model.forward(inputs).detach().numpy()
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
                go.Heatmap(x=np.arange(self.grid_x1.min(), self.grid_x1.max(), self.h),
                           y=np.arange(self.grid_x2.min(), self.grid_x2.max(), self.h),
                           z=self.preds,
                           name='Predictions',
                           colorscale='Spectral',
                           hovertemplate="X1: %{x:.3g}, X2: %{y:.3g}: Pred: %{z:<001.2f}",
                           hoverongaps=False))
            fig.add_trace( # Scatter of test data
                go.Scatter(x=self.X[:, 0],
                           y=self.X[:, 1],
                           mode='markers',
                           name='Test Set',
                           marker=dict(color=y_test,
                                       cmin=self.preds.min(),
                                       cmax=self.preds.max(),
                                       colorscale='Spectral',
                                       line=dict(width=1)),
                           text=y_test))
            fig.update_layout(title='\"Moons\" Predictions',
                            xaxis_title='X1',
                            yaxis_title='X2',
                            legend_title_text='Predicted Value')
                            # showlegend=False)
        self.mode = mode
        self.fig = fig
        return self.fig

    def update_figure(self, model: nn.Module) -> Union[go.Figure, plt.Figure]:
        self.fit_grid(model)
        self.build_figure(self.mode)
        return self.fig


# mv = ModelVisualization(slnn, X_test, y_test, h=0.02)
# mv.fig

# %% Build a simple neural network from scratch
# %% Randomize weights and biases
# Delete old ones to avoid confusion
# Same as before, initialize "empty" tensors
w = torch.empty(1, 2, requires_grad=True)   # 1x2 uninitialised weight matrix
b = torch.empty(1, 1, requires_grad=True)   # 1x1 uninitialised bias matrix

# Populate with draws from random normal distribution
torch.manual_seed(0)
with torch.no_grad():                       # Will explain this later
    nn.init.normal_(w, mean=0, std=0.5)     # Fill with random values
    nn.init.normal_(b, mean=0, std=0.5)
print(w, b)

# %% Usually we use modules instead ; Custom simple module
class SimpleLinearNeuralNetwork(nn.Module): # nn.Module parent class
    def __init__(self, weights, bias):      # takes weights and bias as args  
        super().__init__()                  # parent class __init__
        self.weights = nn.Parameter(weights)# need to define as parameter so
        self.bias = nn.Parameter(bias)      # optimizer knows to optimize it
    
    def forward(self, inputs):              # forward pass over input data
        preds = torch.matmul(               # same calculation as above
            self.weights, inputs.T          # W∙X'
            ) + self.bias                   # + b
        return preds                        # return predictions


# Construct network
slnn = SimpleLinearNeuralNetwork(w, b)

# %% Visualize the uninitialised model and its predictions
mv = ModelVisualization(slnn, X_test, y_test, h=0.02)
mv.fig.update_layout(title="Moons Predictions with Uninitialised Model")

# %% Training the model
# For training convenience convert to tensors
X_train, y_train = (torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32))

# %% Single forward and backward pass
# Batch of 5 observations
inputs = X_train[:5, :]
labels = y_train[:5]

# Initialize optimizer
optim = torch.optim.SGD(slnn.parameters(), lr=0.01)

# Reset the gradients - will explain later why this is necessary 
optim.zero_grad()

# Forward pass: Data -> Predictions
preds = slnn.forward(inputs)

# Loss calculation
loss = (preds - labels).pow(2).mean() # mean squared loss
loss.backward()

# Update parameters
print(slnn.weights, slnn.bias)
optim.step()
print(slnn.weights, slnn.bias)

mv.fit_grid(model=slnn)
mv.build_figure(mode='plotly')
mv.fig

# %% Datasets and Dataloaders - a topic for the next workshop
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, X: torch.tensor, y: torch.tensor):
        self.features = X
        self.labels = y
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx: int):
        return self.features[idx, :], self.labels[idx]

dataloader = DataLoader(dataset=SimpleDataset(X_train, y_train), batch_size=32, shuffle=True)

# %% Define training loop
loss_fn = nn.MSELoss()
optim   = torch.optim.SGD(slnn.parameters(), lr=0.01)

for features, labels in tqdm(dataloader):























# %% Matplotlib
fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], X_test[:, 1], c=[ 'b' if i==0 else 'r' for i in y_test ])

# %% Plotly

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
               text=['Class 1' if y==1 else
                     'Class 0' for y in y_test]
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
inputs = torch.arange(10).reshape(-1, 2)
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

# %% Showing how WX' + b in 2D works
# # Tensors hold values
# weights = torch.tensor([[1, 2]], dtype=torch.float32)
# bias = torch.tensor([[1]], dtype=torch.float32)
# print("Weights: ", weights)
# print("Bias", bias)

# # Simple y = WX' + b
# inputs = torch.arange(10).reshape(-1, 2).float()
# print("Inputs: ", inputs)
# preds = torch.matmul(weights, inputs.T) + bias
# print("Preds: ", preds)
# print("Check : 1*0 + 2*1 + 1 = 3\n"
#       "        1*2 + 2*3 + 1 = 9\n"
#       "        1*4 + 2*5 + 1 = 15\n"
#       "        1*6 + 2*7 + 1 = 21\n"
#       "        1*8 + 2*9 + 1 = 27")
