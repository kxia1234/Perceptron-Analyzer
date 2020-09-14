# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        The network should have the following architecture (in terms of hidden units):
        in_size -> 128 ->  out_size
        """

        # GIVEN CODE #
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # INITIALIZE LINEAR VARIABLES #
        self.linear_in = nn.Linear(in_features=in_size, out_features=32, bias=True)
        self.linear_out = nn.Linear(in_features=32, out_features=out_size, bias=True)

        # self.linear_in_weight = nn.Linear(in_features=in_size, out_features=32, bias=True).weight
        # self.linear_out_weight = nn.Linear(in_features=32, out_features=out_size, bias=True).weight

        # INITIALIZE OPTIM #
        self.optim = torch.optim.SGD(self.get_parameters(), lr=lrate, weight_decay= 0.25)

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return [self.linear_in.weight, self.linear_in.bias, self.linear_out.weight, self.linear_out.bias]

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        return self.linear_out(nn.functional.relu(self.linear_in(x)))

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optim.zero_grad()
        loss_calc = self.loss_fn(self.forward(x),y)
        loss_calc.backward()
        self.optim.step()
        return float(loss_calc.data)


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, 784) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M, 784) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """

    # INITIALIZE MODEL #
    net = NeuralNet(0.001, torch.nn.CrossEntropyLoss(), train_set.shape[1], 5)
    losses = []

    # PREDICT THE DEV SET #
    for iter in range(n_iter):
        for x in range(0, len(train_labels) - batch_size + 1, batch_size):
            losses.append(net.step(train_set[x:x+batch_size], train_labels[x:x+batch_size]))
    _, yhats_max = torch.max(net.forward(dev_set), 1)
    yhats = yhats_max.tolist()
    return losses, yhats, net
