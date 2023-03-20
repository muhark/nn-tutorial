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
    alpha=0.6, s=3, zorder=10
)

# %% Also try a decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) # Reshape predictions back to mesh

# %%
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=
    ['#4444ff' if i==0 else '#ff4444' for i in y],
    alpha=1, s=4
)

# %% And finally a Naive Bayes
clf = naive_bayes.GaussianNB()
clf.fit(X, y)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) # Reshape predictions back to mesh

# %%
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=
    ['#4444ff' if i==0 else '#ff4444' for i in y],
    alpha=1, s=4
)

## Looking at Neural Networks now
# %% Simple Neural Network Classifier
class SingleNeuron(nn.Module):
    def __init__(self):
        super(SingleNeuron, self).__init__()
        self.linear = nn.Linear(2, 1, bias=True)
    
    def forward(self, x):
        x = self.linear(x)
        logits = torch.sigmoid(x)
        #preds = (logits>0.5).float()
        return logits
    
# %%
device = torch.device('mps')
single_neuron = SingleNeuron().to(device)

# %%
X_tensor = torch.tensor(X, device=device, dtype=torch.float32)
y_tensor = torch.tensor(y, device=device, dtype=torch.float32)

# %%
with torch.no_grad():
    print(single_neuron(X_tensor[0, :]).gt(0.5).float())

# %%
mesh = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                    device=device,
                    dtype=torch.float32)

# %% Uninitialised model makes random predictions
Z = single_neuron(mesh).gt(0.5).float().detach().cpu().numpy()
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=
    ['#4444ff' if i==0 else '#ff4444' for i in y],
    alpha=1, s=4
)

# %% Prepare loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(single_neuron.parameters(), lr=3e-4)

# %% Train one observation
optimizer.zero_grad()
pred = single_neuron(X_tensor[0, :])
loss = loss_fn(pred, y_tensor[:1])
loss.backward()
optimizer.step()

# %%
Z = single_neuron(mesh).detach().cpu().numpy()#.gt(0.5).float().detach().cpu().numpy()
Z = Z.reshape(xx.shape)

CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
plt.clabel(CS, inline=1, fontsize=10)
plt.scatter(X[:,0], X[:,1], c=
    ['#4444ff' if i==0 else '#ff4444' for i in y],
    alpha=1, s=4
)

# %%
Z = single_neuron(mesh).gt(0.5).float().detach().cpu().numpy()
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=
    ['#4444ff' if i==0 else '#ff4444' for i in y],
    alpha=1, s=4
)

# %% Train one epoch
trainloader = torch.utils.data.DataLoader(X_tensor, shuffle=True)

# %%
for row in trainloader:
optimizer.zero_grad()
pred = single_neuron(X_tensor[0, :])
loss = loss_fn(pred, y_tensor[:1])
loss.backward()
optimizer.step()
