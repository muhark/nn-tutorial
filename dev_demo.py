#%% Little ANN Tutorial
import torch
from torch import nn
from torch import optim
from sklearn import datasets
from sklearn import svm, naive_bayes, tree
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('whitegrid')


#%%  Create a dataset of blobs
X, y = datasets.make_blobs(
    n_samples=1000,
    n_features=2,
    centers=2,
    random_state=0)



# %% Quick visualization
plt.scatter(X[:,0], X[:,1], c=
    ['r' if i==0 else 'b' for i in y]
)
plt.show()

# %% https://stackoverflow.com/a/22356267
# Build a mesh
h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# %% Fit SVM
clf = svm.SVC(kernel = 'linear')
clf.fit(X, y)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) # Reshape predictions back to mesh


# %% Put the result into a color plot
plt.contourf(xx, yy, Z, cmap=plt.cm.Greens)
plt.scatter(X[:,0], X[:,1], c=
    ['r' if i==0 else 'b' for i in y],


# %% Implement basic neural network
# There's different ways to approach this
# Let's do it outside of a function etc. first
weights = torch.tensor(shape=(2, 1), requires_grad=True)
biases = torch.tensor(shape=(2, 1), requires_grad=True)

# Let's make a prediction off the first input
inputs = torch.tensor(X[0, :], requires_grad=False)
preds = weights*inputs + biases # wx+b

# How wrong were we?
# Use squared loss
correct = torch.tensor(y[0]).reshape(1, -1)
loss = torch.sqrt(correct - preds)

# Now for the magic: backprop
print(weights, biases)
print(weights*inputs + biases)
print(preds, correct)

loss.backward()

print(weights, biases)
print(weights*inputs + biases)
print(preds, correct)














def linear_forward(params, inputs):
    """
    Given a collection of parameters



