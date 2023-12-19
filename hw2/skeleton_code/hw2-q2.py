#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    # Output width = ((input width − kernel width + 2 × padding width) / stride) + 1
    
    def __init__(self, dropout_prob, no_maxpool=False):
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool
        if not no_maxpool:
            # Implementation for Q2.1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(in_features=16*6*6, out_features=320)
            #raise NotImplementedError
        else:
            # Implementation for Q2.2
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0)
            self.fc1 = nn.Linear(in_features=16*1*2, out_features=320)
            #raise NotImplementedError
        
        # Implementation for Q2.1 and Q2.2
        
        self.drop = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(in_features=320, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        #raise NotImplementedError
        
    def forward(self, x):
        # initial shape: [8, 1, 28, 28]
        #print('Initial shape: ', x.shape)

        # input should be of shape [b, c, w, h]

        x = x.reshape((x.shape[0], 1, 28, 28))
        #print('Second shape: ', x.shape)

        # conv and relu layers
        x = F.relu(self.conv1(x))
        #print('Third shape: ', x.shape)
        
        # Convolution with 3x3 filter with padding (1) and 8 channels =>
        #    x.shape = [8, 8, 28, 28] since 28 = 28 - 3 + 1 + 2 * 1
        #    x.shape = [8, 8, _, _] since 28 = ((28 - 3 + 2 * 1) / 2) + 1
        # max-pool layer if using it
        if not self.no_maxpool:
            x = self.max_pool(x)
            #print('Forth shape: ', x.shape)
            # Max pooling with stride of 2 =>
            #     x.shape = [8, 8, 14, 14]
            #raise NotImplementedError
        
        # conv and relu layers
        x = F.relu(self.conv2(x))
        #print('Fiveth shape: ', x.shape)
        # With max pool:
        #   Convolution with 3x3 filter with padding and 8 channels =>
        #         x.shape = [8, 16, 12, 12] since 12
        # Without max pool:
        #   Convolution with 3x3 filter with padding and 8 channels =>
        #         x.shape = [8, 16, 28, 28] since _ = ((28 - 3 + 2 * 0) / 2) + 1

        # max-pool layer if using it
        if not self.no_maxpool:
            x = self.max_pool(x)
            #print('Sixth shape: ', x.shape)
            # Max pooling with stride of 2 =>
            #     x.shape = [8, 8, 6, 6]
            #raise NotImplementedError
        
        # prep for fully connected layer + relu
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        #print('Sevend shape: ', x.shape)
        
        # drop out
        x = self.drop(x)
        #print('Eight shape: ', x.shape)

        # second fully connected layer + relu
        x = F.relu(self.fc2(x))
        #print('Nineth shape: ', x.shape)
        
        # last fully connected layer
        x = self.fc3(x)
        #print('Semi-final shape: ', x.shape)
        
        return F.log_softmax(x,dim=1)

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    #print(X.shape)
    #print(y.shape)
    #print(X)
    #print(y)

    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


# Number of trainable weights = Number of filters × ((kernel width × kernel height ×
# number of channels) + bias)
def get_number_trainable_params(model):
    ## TO IMPLEMENT - REPLACE return 0
    model_parameters_cnn = filter(lambda p: p.requires_grad, model.parameters())
    params_cnn = sum([np.prod(p.size()) for p in model_parameters_cnn])
    #print('Number of parameters in the CNN model: {}'.format(params_cnn))
    return params_cnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.7)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # initialize the model
    model = CNN(opt.dropout, no_maxpool=opt.no_maxpool)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer, opt.no_maxpool)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
