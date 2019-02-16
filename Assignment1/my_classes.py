import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

save_dir = 'data'
iris_link = 'https://raw.githubusercontent.com/WJMatthew/Deep-Learning/master/iris.csv'


class IrisDataset(Dataset):
    """Iris dataset."""

    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(iris_link)
        self.target = pd.get_dummies(self.data['species'])
        self.data.drop('species', axis=1, inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx]
        y = self.target.iloc[idx]

        sample = {'x': x, 'y': y}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x = sample['x']
        y = sample['y']

        return {'x': torch.from_numpy(x.values),
                'y': torch.from_numpy(y.values).long()}


# One Hot Encode Vectors
def one_hot_encode(pre_labels, n_classes):
    labels = []

    for num in range(len(pre_labels)):
        # idx <- class: 0, 1, 2
        idx = pre_labels[num]
        # Array of Zeros
        arr = np.zeros(n_classes)
        # One hot encode array by class index
        arr[idx] = 1
        # Add array to labels (list)
        labels.append(arr)

    labels = np.array(labels, dtype=int)

    return labels


##############################
#    ACTIVATION FUNCTIONS    #
##############################

def relu_activation(z):
    return z.clamp(min=0)


def relu_delta(grad_h, h):
    grad_h[h < 0] = 0
    return grad_h


def sigmoid_activation(z):
    return 1 / (1 + torch.exp(-z))


def sigmoid_delta(z):
    return z * (1 - z)


########################
#    COST FUNCTIONS    #
########################

def mean_sum_square_errors(y_hat, y):
    return (y_hat - y).pow(2).sum().item()/len(y_hat)


def sum_square_errors_delta(y_hat, y):
    return (y_hat - y)/2


def cross_entropy2(y_hat, y):
    N = len(y_hat)
    ce = -np.sum(y * np.log(y_hat)) / N
    return ce


def cross_entropy1(y_hat, y_):
    y = torch.argmax(y_, 1)
    m = len(y)
    p = softmax(y_hat)
    log_likelihood = -torch.log()
    loss = torch.mean(log_likelihood.sum())
    return loss.item()


def cross_entropy(outputs, labels):
#def myCrossEntropyLoss(outputs, labels):
        outputs = torch.argmax(outputs, 1)
        batch_size = outputs.size()[0]  # batch_size
        outputs = F.log_softmax(outputs, dim=1)  # compute the log of softmax values
        outputs = outputs[range(batch_size), labels]  # pick the values corresponding to the labels
        return -torch.sum(outputs) / len(outputs)

def ce(y_hat, y_):
    #y = torch.argmax(y_, 1)
    return torch.mean(-torch.sum(y_ * torch.log(y_hat), 1))

def cross_entropy_delta(y_hat, y_):
    y = torch.argmax(y_, 1)
    m = len(y)
    grad = softmax(y_hat)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


# SOFTMAX
def softmax(X):
    exps = torch.exp(X - torch.max(X))
    return exps / exps.sum()


# TESTING
def get_accuracy(y_hat, y):
    y_pred = torch.max(y_hat, 1)
    print('-', y_pred.shape)
    y_test = torch.max(y, 1)[1]
    accuracy = (y_pred == y_test).float().mean().item()
    return accuracy


def predict(x, w1, w2):
    h = x.mm(w1)
    h_act = relu_activation(h)
    y_pred = h_act.mm(w2)
    y_pred = softmax(y_pred)
    return torch.max(y_pred, 1)
