#!/usr/bin/python
'''A simple python implementation of a restricted Boltzmann machine (RBM).

This module provides a single class representing an RBM. Directly executing
this module will create a single RBM, train it on a small dataset, and allow
the machine to daydream for a short time.

'''

import math
from operator import add
import random

class RBM:
    '''A restricted Boltzmann machine.

    Attributes:
      num_visible (int): The number of visible units in the machine.
      num_hidden (int): The number of hidden units in the machine.
      learning_rate (float): Adjusts how rapidly the machine converges while training.
      weights (list of list of float): The weights of the connections between the visible and
        hidden units.

    '''

    def __init__(self, num_visible, num_hidden, weights = [], learning_rate = 0.1):
        '''Constructor for RBM

        Creates a new RBM with the given number of visible and hidden units. Optionally, the
        weights and learning rate of the machine may be specified.

        Args:
          num_visible (int): The number of visible units in the machine.
          num_hidden (int): The number of hidden units in the machine.
          weights (list of list of float, optional): The weights of the connections between the
            visible and hidden units. defaults to an empty list.
          learning_rate (float, optional): Adjusts how rapidly the machine converges while
            training. defaults to 0.1.
        
        '''

        random.seed()

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate

        weights_are_valid = len(weights) == num_visible and all([len(w) == num_hidden for w in weights])
        self.weights = weights if weights_are_valid else self._format_weights(weights, num_visible, num_hidden)

    def load_weights(self, path):
        '''Reads weight values from a file.

        Args:
          path (str): A path to a csv file containing weight values.
        
        '''
        with open(path, 'r') as f:
            weights = [[float(d) for d in line.strip().split(',')] for line in f]

        weights_are_valid = len(weights) == self.num_visible and all([len(w) == self.num_hidden for w in weights])
        self.weights = weights if weights_are_valid else self._format_weights(weights, self.num_visible, self.num_hidden)

    def train(self, data, epochs = 100, sample_rate = 1):
        '''Trains the machine on the given dataset.

        Args:
          data (list of list): The dataset on which to train the machine. This should be a list of
            inputs to the machine, each of which should be a list of ints or floats of length equal
            to the number of visible units in the machine.
          epochs (int, optional): The number of training epochs. defaults to 100.
          sample_rate (float, optional): The proportion of the provided training data to use in
            each epoch. If sample_rate is between 0 and 1, a subset of the data will be sampled Attributes
            the start of each epoch. defaults to 1.

        '''
        num_samples = len(data) if sample_rate >= 1 or sample_rate <= 0 else int(len(data) * sample_rate)
        for epoch in range(epochs):
            print 'Epoch ' + str(epoch)
            if sample_rate < 1 and sample_rate > 0:
                samples = random.sample(data, num_samples)
            else:
                samples = data

            hidden_activations, hidden_probabilities = self.run_visible_units(samples)
            positive_gradients = self._calculate_gradients(samples, hidden_probabilities)
            visible_activations, visible_probabilities = self.run_hidden_units(hidden_activations)
            hidden_activations, hidden_probabilities = self.run_visible_units(visible_activations)
            negative_gradients = self._calculate_gradients(visible_probabilities, hidden_probabilities)
            deltas = [[self.learning_rate * (-positive_gradients[j][i] + negative_gradients[j][i]) / float(len(samples)) for i in range(self.num_hidden)] for j in range(self.num_visible)]
            for i in range(self.num_visible):
                for j in range(self.num_hidden):
                    self.weights[i][j] += deltas[i][j]
            error = sum([sum([(samples[s][i] - visible_probabilities[s][i]) ** 2 for i in range(self.num_visible)]) for s in range(num_samples)])
            print 'Error: ' + str(error)

    def run_visible_units(self, data):
        '''Activates the machine for a given set of inputs to the visible units.

        Args:
          data (list of list): The dataset on which to activate the machine. This should be a list
            of inputs to the machine, each of which should be a list of ints or floats of length
            equal to the number of visible units in the machine.

        Returns:
          hidden_activations (list of list of int): A sampled state of the hidden units based on
            probabilities calculated from the given data.
          hidden_probabilities (list of list of float): The probabilities of activation for the
            hidden units based on the given data.

        '''

        weighted_data = [[sum([datum[i] * self.weights[i][j] for i in range(self.num_visible)]) for j in range(self.num_hidden)] for datum in data] 
        hidden_probabilities = [[(1.0 / (1.0 + math.exp(value))) for value in weighted_datum] for weighted_datum in weighted_data]
        hidden_activations = [[1 if prob > random.random() else 0 for prob in probs] for probs in hidden_probabilities]
        return (hidden_activations, hidden_probabilities)

    def run_hidden_units(self, data):
        '''Activates the machine for a given set of inputs to the hidden units.

        Args:
          data (list of list): The dataset on which to activate the machine. This should be a list
            of inputs to the machine, each of which should be a list of ints or floats of length
            equal to the number of hidden units in the machine.

        Returns:
          visible_activations (list of list of int): A sampled state of the visible units based on
            probabilities calculated from the given data.
          visible_probabilities (list of list of float): The probabilities of activation for the
            visible units based on the given data.

        '''

        weighted_data = [[sum([datum[j] * self.weights[i][j] for j in range(self.num_hidden)]) for i in range(self.num_visible)] for datum in data] 
        visible_probabilities = [[(1.0 / (1.0 + math.exp(value))) for value in weighted_datum] for weighted_datum in weighted_data]
        visible_activations = [[1 if prob > random.random() else 0 for prob in probs] for probs in visible_probabilities]
        return (visible_activations, visible_probabilities)

    def run_all(self, data):
        '''Activates the machine for a given set of inputs.

        Args:
          data (list of list): The dataset on which to activate the machine. This should be a list
            of inputs to the machine, each of which should be a list of ints or floats of length
            equal to the number of visible units in the machine.

        Returns:
          visible_activations (list of list of int): A sampled state of the visible units based on
            probabilities calculated from the given data.
          visible_probabilities (list of list of float): The probabilities of activation for the
            visible units based on the given data.

        '''

        return self.run_hidden_units(self.run_visible_units(data)[0])

    def daydream(self, limit = 10, initial_values = []):
        '''Repeatedly activates the machine, starting in a random state.

        Args:
          limit (int, optional): The number of daydream iterations to perform. defaults to 10.

        Returns:
          A list of length equal to the limit parameter containing the daydream results.

        '''

        if len(initial_values) > 0:
            current_activations = initial_values
        else:
            current_activations = [[random.random() for _ in range(self.num_visible)]]
        current_probabiltiies = []
        results = []
        for _ in range(limit):
            current_activations, current_probabilities = self.run_all(current_activations)
            results.append(current_activations[0])
        return results

    def _calculate_gradients(self, v, h):
        return self._matrix_mult(self._transpose(v), h)

    def _transpose(self, m):
        t = [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]
        return t

    def _matrix_mult(self, a, b):
        return [[sum([a[i][k] * b[k][j] for k in range(len(b))]) for j in range(len(b[0]))] for i in range(len(a))]

    def _format_weights(self, weights, num_visible, num_hidden):
        formatted_weights = [[weights[i][j] if j < len(weights[i]) else random.gauss(0, 0.25) for j in range(num_hidden)] if i < len(weights) else [random.gauss(0, 0.25) for j in range(num_hidden)] for i in range(num_visible)]
        return formatted_weights

if __name__ == '__main__':
    r = RBM(num_visible = 6, num_hidden = 2)
    training_data = [[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]]
    r.train(training_data, epochs = 1000)
    print(r.weights)
    dreams = r.daydream()
    print(dreams)