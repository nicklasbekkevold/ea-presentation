import csv
import statistics
from typing import Any, Callable

import matplotlib.pyplot as plt
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


def plot_metrics(metrics: dict[str, list], path: str) -> None:
    fig, ax1 = plt.subplots()
    ax1.set_title("Genetic Algorithm")

    ax1.set_xlabel("generation")
    ax1.set_ylabel("fitness / -RMSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax1.plot(metrics["average"], color="tab:blue", label="Average")
    ax1.plot(metrics["baseline"], linestyle="-", color="k", label="Baseline")
    ax1.plot(metrics["best"], linestyle="--", color="k", label="Best")
    ax1.legend()

    ax2 = ax1.twinx()

    ax2.set_ylabel("entropy", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax2.plot(metrics["entropy"], color="tab:orange", label="Entropy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close()


def save_metrics(metrics: dict[str, list[Any]], path: str) -> None:
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))


def save_solution(chromosome: npt.NDArray, path: str) -> None:
    with open(path, "w") as text_file:
        text_file.write(results.print_solution(chromosome))


def create_hooks(
    baseline_chromosome: npt.NDArray,
    fitness_function: Callable[[npt.NDArray], float],
) -> tuple[Callable[[int, npt.NDArray], None], Callable[[npt.NDArray], None]]:
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

    def save_results(solution: npt.NDArray) -> None:
        results.create_result_folder()
        results.copy_parameters()
        result_folder = results.get_current_result_folder()

        plot_metrics(metrics, f"{result_folder}/ga.png")
        save_metrics(metrics, f"{result_folder}/metrics.csv")
        save_solution(solution, f"{result_folder}/solution.txt")

    return add_metrics, save_results
