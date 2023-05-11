from typing import Callable

import numpy as np
import numpy.typing as npt

from . import parameters


def generate_chromosome(
    chromosome_length=parameters.CHROMOSOME_LENGTH,
) -> npt.NDArray:
    return np.random.choice(a=[False, True], size=(chromosome_length,))


def generate_population(
    chromosome_length=parameters.CHROMOSOME_LENGTH,
    population_size=parameters.POPULATION_SIZE,
) -> npt.NDArray:
    return np.array(
        [generate_chromosome(chromosome_length) for _ in range(population_size)]
    )


def mutate(
    chromosome: npt.NDArray,
    mutation_rate=parameters.MUTATION_RATE,
) -> npt.NDArray:
    """
    Stochastically performs bitflips for each gene in the chromosome with a probability
    equal to the `mutation_rate`
    """

    def stochastic_bit_flip(
        variable: bool,
    ) -> bool:
        return not variable if np.random.random() <= mutation_rate else variable

    return np.array(list(map(stochastic_bit_flip, chromosome)))


def crossover(
    chromosome_a: npt.NDArray,
    chromosome_b: npt.NDArray,
    crossover_rate=parameters.CROSSOVER_RATE,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Performs a one-point crossover with a probability equal to the `crossover_rate`
    """

    if np.random.random() <= crossover_rate:
        assert len(chromosome_a) == len(chromosome_b), "Chromosome lengths must match"
        cutoff_point = np.random.randint(1, len(chromosome_b) - 1)
        offspring_a = np.concatenate(
            (chromosome_a[:cutoff_point], chromosome_b[cutoff_point:])
        )
        offspring_b = np.concatenate(
            (chromosome_b[:cutoff_point], chromosome_a[cutoff_point:])
        )
        return offspring_a, offspring_b
    return chromosome_a, chromosome_b


def find_elite(
    population: npt.NDArray,
    fitness_function: Callable[[npt.NDArray], float],
    elite_size=parameters.ELITE_SIZE,
) -> npt.NDArray:
    return np.array(sorted(population, key=fitness_function, reverse=True)[:elite_size])


def tournament_selection(
    population: npt.NDArray,
    fitness_function: Callable[[npt.NDArray], float],
    tournament_size=parameters.TOURNAMENT_SIZE,
):
    tournament = population[np.random.choice(population.shape[0], tournament_size)]
    return np.array(sorted(tournament, key=fitness_function, reverse=True)[0])


def generational_step(
    population: npt.NDArray,
    fitness_function: Callable[[npt.NDArray], float],
    elite_size=parameters.ELITE_SIZE,
    tournament_size=parameters.TOURNAMENT_SIZE,
    crossover_rate=parameters.CROSSOVER_RATE,
    mutation_rate=parameters.MUTATION_RATE,
) -> npt.NDArray:
    offspring = find_elite(population, fitness_function, elite_size)
    for _ in range(len(population)):
        parent_a = tournament_selection(population, fitness_function, tournament_size)
        parent_b = tournament_selection(population, fitness_function, tournament_size)
        offspring_a, offspring_b = crossover(parent_a, parent_b, crossover_rate)
        offspring_a = mutate(offspring_a, mutation_rate)
        offspring_b = mutate(offspring_b, mutation_rate)
        np.append(offspring, [offspring_a, offspring_b])
    return offspring


def optimize(
    fitness_function: Callable[[npt.NDArray], float],
    generations=parameters.GENERATIONS,
    chromosome_length=parameters.CHROMOSOME_LENGTH,
    population_size=parameters.POPULATION_SIZE,
    elite_size=parameters.ELITE_SIZE,
    tournament_size=parameters.TOURNAMENT_SIZE,
    crossover_rate=parameters.CROSSOVER_RATE,
    mutation_rate=parameters.MUTATION_RATE,
    generation_hook=lambda x: None,
) -> npt.NDArray:
    population = generate_population(chromosome_length, population_size)
    for generation in range(generations):
        population = generational_step(
            population,
            fitness_function,
            elite_size,
            tournament_size,
            crossover_rate,
            mutation_rate,
        )
    return population
