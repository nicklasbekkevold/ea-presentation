import numpy as np

from src.metrics import (
    average_fitness,
    entropy,
)


def test_average_fitness() -> None:
    population = np.array([[False, False], [False, True], [True, False]])

    def fitness_function(chromosome):
        return int("".join(chromosome.astype(int).astype(str).tolist()), 2)

    assert average_fitness(population, fitness_function) == 1


def test_entropy() -> None:
    assert entropy(np.array([[False, False]])) == 0
    assert entropy(np.array([[False, False], [True, True]])) == 1
