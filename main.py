import numpy as np
import numpy.typing as npt

from src import ga, linear_regression, parameters


def hash(chromosome: npt.NDArray) -> int:
    return int("".join(chromosome.astype(int).astype(str).tolist()), 2)


def get_fitness_function() -> float:
    features, labels = linear_regression.load_data_set_from_csv()

    cache = {}

    def memoized_fitness_function(chromosome: npt.NDArray) -> float:
        if hash(chromosome) in cache:
            return cache[hash(chromosome)]
        fitness = linear_regression.compute_fitness(chromosome, features, labels)
        cache[hash(chromosome)] = fitness
        return fitness

    return memoized_fitness_function


def main():
    np.random.seed(parameters.SEED)

    population = ga.generate_population()
    fitness_function = get_fitness_function()
    print(ga.generational_step(population, fitness_function))


if __name__ == "__main__":
    main()
