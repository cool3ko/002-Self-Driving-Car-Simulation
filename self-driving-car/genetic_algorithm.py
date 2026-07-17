"""Genetic algorithm for evolving the car's neural network.

This module provides functions for:
- Creating and managing a population of brains
- Evaluating fitness based on distance traveled
- Selecting the best performers
- Breeding and mutating new generations
"""

import random

import numpy as np

from neural_network import Brain


def create_population(population_size, layer_sizes):
    """Create an initial population of randomly-initialized brains.

    Args:
        population_size: Number of brains to create
        layer_sizes: Network shape passed to Brain, e.g. [5, 6, 4]

    Returns:
        list: Population of Brain objects
    """
    return [Brain(layer_sizes) for _ in range(population_size)]


def evaluate_fitness(cars):
    """Sort cars by fitness, highest first.

    Args:
        cars: List of Car objects, each with a `.fitness` attribute
            (distance travelled before crashing or the run ending)

    Returns:
        list: Cars sorted by fitness (highest first)
    """
    return sorted(cars, key=lambda car: car.fitness, reverse=True)


def select_best(population, num_best):
    """Select the best performing entries from a fitness-sorted population.

    Args:
        population: List sorted by fitness (highest first)
        num_best: Number of best entries to select

    Returns:
        list: Top performing entries
    """
    return population[:max(1, num_best)]


def mutate(brain, amount=0.1):
    """Mutate a neural network brain in place.

    Args:
        brain: Brain instance to mutate
        amount: Maximum magnitude of the random perturbation
    """
    Brain.mutate(brain, amount)


def breed(parent1, parent2):
    """Create an offspring brain via uniform crossover of two parents.

    Args:
        parent1: First parent Brain
        parent2: Second parent Brain

    Returns:
        Brain: Offspring brain
    """
    child = parent1.clone()
    for child_level, level1, level2 in zip(child.levels, parent1.levels, parent2.levels):
        weight_mask = np.random.rand(*level1.weights.shape) < 0.5
        child_level.weights = np.where(weight_mask, level1.weights, level2.weights)

        bias_mask = np.random.rand(*level1.biases.shape) < 0.5
        child_level.biases = np.where(bias_mask, level1.biases, level2.biases)

    return child


def evolve_generation(cars, elite_count=2, mutation_amount=0.15):
    """Produce the next generation of brains from a finished population.

    The single fittest car is carried over unchanged (elitism), and the
    rest of the new population is bred from the top performers and then
    mutated.

    Args:
        cars: List of Car objects from the generation that just ended
        elite_count: How many of the top performers are eligible parents
        mutation_amount: Maximum magnitude of mutation applied to children

    Returns:
        list: Next generation of Brain objects, same size as `cars`
    """
    ranked = evaluate_fitness(cars)
    parents = [car.brain for car in select_best(ranked, elite_count) if car.brain]

    if not parents:
        return [car.brain for car in cars]

    population_size = len(cars)
    next_brains = [parents[0].clone()]

    while len(next_brains) < population_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = breed(parent1, parent2)
        mutate(child, mutation_amount)
        next_brains.append(child)

    return next_brains
