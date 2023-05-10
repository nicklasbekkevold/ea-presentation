import random

from . import parameters


def generate_chromosome(chromosome_length=parameters.CHROMOSOME_LENGTH) -> list[str]:
    return [random.choice([True, False]) for _ in range(chromosome_length)]


def generate_population(
    chromosome_length=parameters.CHROMOSOME_LENGTH,
    population_size=parameters.POPULATION_SIZE,
):
    return [generate_chromosome(chromosome_length) for _ in range(population_size)]


def mutate(chromosome, mutation_rate=parameters.MUTATION_RATE):
    mutated_chromosome = []
    for gene in chromosome:
        if random.random() <= mutation_rate:
            mutated_chromosome.append(not gene)
            continue
        mutated_chromosome.append(gene)
    return mutated_chromosome


def crossover(chromosome_a, chromosome_b, crossover_rate=parameters.CROSSOVER_RATE):
    assert len(chromosome_a) == len(chromosome_b), "Chromosome length must match"
    if random.random() <= crossover_rate:
        crossover_point = random.randrange(1, len(chromosome_b) - 1)
        offspring_a = chromosome_a[:crossover_point] + chromosome_b[crossover_point:]
        offspring_b = chromosome_b[:crossover_point] + chromosome_a[crossover_point:]
        return offspring_a, offspring_b
    return chromosome_a, chromosome_b
