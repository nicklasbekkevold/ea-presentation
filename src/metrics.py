import csv
import statistics
from typing import Callable

import numpy as np
import numpy.typing as npt

from . import results


def baseline_fitness(
    chromosome: npt.NDArray, fitness_function: Callable[[npt.NDArray], float]
) -> float:
    return fitness_function(chromosome)


def best_fitness(
    population: npt.NDArray, fitness_function: Callable[[npt.NDArray], float]
) -> float:
    return max(list(map(fitness_function, population)))


def average_fitness(
    population: npt.NDArray, fitness_function: Callable[[npt.NDArray], float]
) -> float:
    return statistics.fmean(list(map(fitness_function, population)))


def entropy(population) -> float:
    occurrences = np.sum(population, axis=0)
    probabilities = occurrences / len(population)
    return -sum([p * np.log2(p) if p != 0 else 0 for p in probabilities])


def create_metrics_hooks(
    baseline_chromosome: npt.NDArray,
    fitness_function: Callable[[npt.NDArray], float],
) -> tuple[Callable[[int, npt.NDArray], None], Callable[[], None]]:
    metrics = {
        "generation": [],
        "baseline": [],
        "best": [],
        "average": [],
        "entropy": [],
    }
    baseline = baseline_fitness(baseline_chromosome, fitness_function)

    def add_metrics(generation: int, population: npt.NDArray):
        metrics["generation"].append(generation)
        metrics["baseline"].append(baseline)
        metrics["best"].append(best_fitness(population, fitness_function))
        metrics["average"].append(average_fitness(population, fitness_function))
        metrics["entropy"].append(entropy(population))

    def save_metrics() -> None:
        results.create_result_folder()
        results.copy_parameters()
        with open(f"{results.get_current_result_folder()}/metrics.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(metrics.keys())
            writer.writerows(zip(*metrics.values()))

    return add_metrics, save_metrics
