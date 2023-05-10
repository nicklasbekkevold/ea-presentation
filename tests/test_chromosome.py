import numpy as np

from src.chromosome import (
    crossover,
    generate_chromosome,
    generate_population,
    mutate,
)


def test_generate_chromosome() -> None:
    chromosome = generate_chromosome(5)
    assert len(chromosome) == 5


def test_generate_population() -> None:
    population = generate_population(7, 11)
    assert len(population[0]) == 7
    assert len(population) == 11


def test_mutate() -> None:
    chromosome = generate_chromosome(13)

    mutated_chromosome = mutate(chromosome, mutation_rate=0)
    assert (chromosome == mutated_chromosome).all()

    mutated_chromosome = mutate(chromosome, mutation_rate=1)
    assert np.logical_xor(chromosome, mutated_chromosome).all()


def test_crossover() -> None:
    parent_a = generate_chromosome(13)
    parent_b = generate_chromosome(13)

    child_a, child_b = crossover(parent_a, parent_b, crossover_rate=0)
    assert (child_a == parent_a).all()
    assert (child_b == parent_b).all()

    child_a, child_b = crossover(parent_a, parent_b, crossover_rate=1)
    assert child_a[0] == parent_a[0]
    assert child_a[-1] == parent_b[-1]
    assert child_b[0] == parent_b[0]
    assert child_b[-1] == parent_a[-1]
    assert len(child_a) == len(parent_a)
    assert len(child_b) == len(parent_b)
