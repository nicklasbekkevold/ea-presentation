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
