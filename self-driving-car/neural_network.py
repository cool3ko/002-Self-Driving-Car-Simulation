"""Neural network module for the car's brain.

Uses plain numpy rather than a training framework because the car is
never trained with backpropagation - it's evolved with a genetic
algorithm (mutation + crossover of weights), so no autograd is needed.
"""

import copy

import numpy as np


class Level:
    """A single fully-connected layer with a step activation."""

    def __init__(self, n_inputs, n_outputs):
        """Initialize a layer with random weights and biases in [-1, 1].

        Args:
            n_inputs: Number of inputs to the layer
            n_outputs: Number of outputs from the layer
        """
        self.weights = np.random.uniform(-1, 1, (n_inputs, n_outputs))
        self.biases = np.random.uniform(-1, 1, n_outputs)

    def feed_forward(self, inputs):
        """Compute the layer's output for a given input vector.

        Args:
            inputs: 1D array-like of length n_inputs

        Returns:
            np.ndarray: 1D array of 0/1 outputs (one per output neuron)
        """
        sums = np.asarray(inputs, dtype=float) @ self.weights + self.biases
        return (sums > 0).astype(float)


class Brain:
    """Neural network that controls the car's behavior.

    Input: sensor readings (one per ray)
    Output: 4 controls (forward, left, right, reverse)
    """

    def __init__(self, layer_sizes):
        """Initialize the neural network architecture.

        Args:
            layer_sizes: Sequence of layer widths, e.g. [5, 6, 4] for
                5 sensor inputs, one hidden layer of 6, and 4 outputs.
        """
        self.layer_sizes = list(layer_sizes)
        self.levels = [
            Level(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]

    def feed_forward(self, inputs):
        """Forward pass through the network.

        Args:
            inputs: Sensor readings

        Returns:
            np.ndarray: Output array (forward, left, right, reverse)
        """
        outputs = inputs
        for level in self.levels:
            outputs = level.feed_forward(outputs)
        return outputs

    def clone(self):
        """Return a deep copy of this brain."""
        return copy.deepcopy(self)

    @staticmethod
    def mutate(brain, amount=0.1):
        """Randomly perturb a brain's weights and biases in place.

        Args:
            brain: Brain instance to mutate
            amount: Maximum magnitude of the random perturbation
        """
        for level in brain.levels:
            level.weights = np.clip(
                level.weights + (np.random.rand(*level.weights.shape) * 2 - 1) * amount,
                -1, 1,
            )
            level.biases = np.clip(
                level.biases + (np.random.rand(*level.biases.shape) * 2 - 1) * amount,
                -1, 1,
            )
