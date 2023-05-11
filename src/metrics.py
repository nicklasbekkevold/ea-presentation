import statistics
from typing import Callable

import numpy as np
import numpy.typing as npt


def average_fitness(
    population: npt.NDArray, fitness_function: Callable[[npt.NDArray], float]
) -> float:
    return statistics.fmean(list(map(fitness_function, population)))


def entropy(population):
    occurrences = np.sum(population, axis=0)
    probabilities = occurrences / len(population)
    return -sum([p * np.log2(p) if p != 0 else 0 for p in probabilities])
