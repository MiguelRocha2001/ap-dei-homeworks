#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        
        x_i_expanded = np.expand_dims(x_i, axis=0)

        y_hat = np.argmax(self.W.dot(x_i_expanded.T))

        #print(np.shape(w))
        #print(np.shape(x_i))
        #print(x_i)
        #print(y_hat)
        #print(y_i)
        #print(x_i)
        #print(self.W)

        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i

        #raise NotImplementedError


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        #print(self.W)

        # Computing One Hot Vector
        n_of_labels = np.shape(self.W)[0]
        one_hot_vector = np.zeros(n_of_labels, )
        one_hot_vector[y_i] = 1
        #print(one_hot_vector)

        # Computs probability vector
        x_i_expanded = np.expand_dims(x_i, axis = 0)
        Zx = np.sum(np.exp(x_i_expanded.dot(self.W.T)))

        #print(Zx)

        probs = np.zeros(n_of_labels, )
        for index, w in enumerate(self.W):
            #w_expanded = np.expand_dims(w, axis = 1)
            #print(x_i_expanded)
            #print(w_expanded)
            #raise NotImplementedError
            probs[index] = np.exp(x_i.dot(w)) / Zx

        # Computs gradient
        aux_expanded = np.expand_dims(probs - one_hot_vector, axis=1)
        x_i_expanded = np.expand_dims(x_i, axis=0)
        #print(aux_expanded)
        #print(x_i_expanded)

        gradient = aux_expanded.dot(x_i_expanded)
        #print(gradient)

        # Updates weights
        self.W -= learning_rate * gradient
        #print(self.W)

        #raise NotImplementedError


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):

        # Initialize an MLP with a single hidden layer. 
        self.B1 = np.zeros(hidden_size,)
        
        self.W2 = np.zeros((n_classes, hidden_size))
        self.B2 = np.zeros(n_classes,)

        mu = 0.1
        sigma = 0.1**2
        
        # initialize W1
        units = hidden_size * n_features
        random = np.random.normal(mu, sigma, units)
        desired_shape = (hidden_size, n_features)
        self.W1 = random.reshape(desired_shape)
        #print(self.W1)

        # initialize W2
        units = n_classes * hidden_size
        random = np.random.normal(mu, sigma, units)
        desired_shape = (n_classes, hidden_size)
        self.W2 = random.reshape(desired_shape)
        #print(self.W2)
        
        #raise NotImplementedError

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        H0 = X

        #print(np.shape(H0))
        #print(np.shape(self.W1))

        Z1 = self.W1.dot(H0.T) + np.expand_dims(self.B1, axis=1)
        #print('JHBDCHSBHC')
        #print(np.shape(Z1))
        H1 = np.maximum(0, Z1)
        #print(np.shape(H1))

        #print(np.shape(self.W2))
        Z2 = self.W2.dot(H1) + np.expand_dims(self.B2, axis=1)
        H2 = Z2
        #print(np.shape(H2))
        #print(H2)
        
        argmax = np.argmax(H2, axis=0)
        #print(np.shape(argmax))
        #print(argmax)
        #raise NotImplementedError
        return argmax
    
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


    # TODO: review softmax: is causing overflow
    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        learning_rate=0.00001
        i = 0
        loss = 0
        for x_i, y_i in zip(X, y):
            #print(np.shape(X))
            #print(f"treino {np.shape(x_i)}")
            #self.predict(x_i, save_values=True)

            # foward pass
            h0 = np.expand_dims(x_i, axis=1)
            #print(self.W1.dot(h0))
            z1 = self.W1.dot(h0) + np.expand_dims(self.B1, axis=1)
            #print(z1)
            h1 = np.maximum(0, z1)
            z2 = self.W2.dot(h1) + np.expand_dims(self.B2, axis=1)

            one_hot_y = np.zeros(np.unique(y).size,)
            one_hot_y[y_i] = 1
        
            # update weights
            #print(one_hot_y)
            #print(z2)

            # compute softmax for predicted y
            sum = np.sum(np.exp(z2))
            #print(sum)
            p = np.exp(z2) / sum
            #print(p)
            #print()

            loss += -np.sum(one_hot_y * np.log(p))
            #print(loss)

            # backpropagation

            # compute gradients
            p_to_vector = p[1, :]
            grad_z2 = np.expand_dims(p_to_vector - one_hot_y, axis=1)
            #print(p_to_vector)
            #print(one_hot_y)
            #print(grad_z2)
            grad_w2 = grad_z2.dot(h1.T)
            grad_b2 = grad_z2

            grad_h1 = self.W2.T.dot(grad_z2)
            relu_derivated = np.where(z1 < 0, 0, 1)
            grad_z1 = grad_h1 * relu_derivated
            
            grad_w1 = grad_z1.dot(h0.T)
            grad_b1 = grad_z1

            # update weights
            self.W2 -= learning_rate * grad_w2
            self.B2 -= learning_rate * grad_b2[1, :]

            self.W1 -= learning_rate * grad_w1
            self.B1 -= learning_rate * grad_b1[1, :]
            print(self.B1)

            #print(self.W1)

            #print(self.W2)
            #print(self.B2)
            #print(self.W1)
            #print(self.B1)
            
            #print(np.shape(self.W1))
            #print(np.shape(self.B1))

            i += 1
            if i == 3:
                break

        return loss / np.shape(X)[0]



def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"   
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    '''
    print('Number of classes: ', n_classes)
    print('Number of fetures: ', n_feats)
    print('X shape: ', np.shape(train_X))
    print('Y shape: ', np.shape(train_y))
    print('Training X: ', train_X)
    print('Training y: ', train_y)
    '''

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
