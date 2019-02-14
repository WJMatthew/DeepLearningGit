import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import my_classes as mc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import my_classes as mc

iris_data = load_iris()
features, pre_labels = iris_data.data, iris_data.target

root = './iris'

if not os.path.exists(root):
    os.mkdir(root)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = mc.one_hot_encode(pre_labels)

feature_train, feature_test, labels_train, labels_test = train_test_split(features, labels, random_state = 17)

# Load the standard scaler
sc = StandardScaler()

# Compute the mean and standard deviation based on the training data
sc.fit(feature_train)

# Scale the training data to be of mean 0 and of unit variance
feature_train = sc.transform(feature_train)

# Scale the test data to be of mean 0 and of unit variance
feature_test = sc.transform(feature_test)

# Data type for tensors
dtype = torch.float

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 112, 4, 7, 3

#
x = torch.as_tensor(torch.from_numpy(feature_train), device=device, dtype=dtype)
y = torch.as_tensor(torch.from_numpy(labels_train), device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

loss_list = []

learning_rate = 1e-6

num_steps = 50000

loss_fn = 'CE'
criterion = nn.CrossEntropyLoss()
activ_fn = 'ReLU'

for t in range(num_steps):
    # Forward pass: compute predicted y
    h = x.mm(w1)

    if activ_fn is 'ReLU':
        h_activation = h.clamp(min=0)
    elif activ_fn is 'Sig':
        h_activation = mc.sigmoid_activation(h)

    y_pred = h_activation.mm(w2)

    # Compute and print loss
    if loss_fn is 'MSE':
        loss = mc.mean_sum_square_errors(y_pred, y)
    elif loss_fn is 'CE':
        #print(y_pred.shape, y.shape)
        #print(y_pred, y)
        loss = mc.cross_entropy(y_pred, y.long())
        #loss = criterion(torch.argmax(y_pred), y.long())

    loss_list.append(loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    if loss_fn is 'MSE':
        grad_y_pred = mc.sum_square_errors_delta(y_pred, y)
    elif loss_fn is 'CE':
        grad_y_pred = mc.cross_entropy_delta(y_pred, y)

    grad_w2 = h_activation.t().mm(grad_y_pred)
    grad_h_activation = grad_y_pred.mm(w2.t())
    grad_h = grad_h_activation.clone()

    if activ_fn is 'ReLU':
        grad_h[h < 0] = 0
    elif activ_fn is 'Sig':
        grad_h = grad_h * (1 - grad_h)

    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    if (t + 1) % 1000 == 0:
        print(f'Step [{t+1}/{num_steps}], Loss: {loss:.4}')


# Testing
x_test = torch.from_numpy(feature_test).float()
y_test = torch.from_numpy(labels_test).float()

y_preds = mc.predict(x_test, w1, w2)[1]

# Compute accuracy
_, argmax = torch.max(y_test, 1)
accuracy = (y_preds == argmax.squeeze()).float().mean()

print(f'Acc: {accuracy:.2}')
