import logging

import numpy as np

from src import ga, metrics, parameters
from src.linear_regression import create_fitness_function
from src.results import print_solution


def main():
    logging.basicConfig(
        format="%(levelname)s - %(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    np.random.seed(parameters.SEED)

    fitness_function = create_fitness_function()
    baseline = np.full((parameters.CHROMOSOME_LENGTH,), True)
    add_metrics, save_results = metrics.create_hooks(baseline, fitness_function)

    solution = ga.optimize(
        fitness_function, generational_hook=add_metrics, post_optimize_hook=save_results
    )

    logging.info(f"Solution found: {print_solution(solution)}")
    logging.info(f"Solution RMSE: {-fitness_function(solution)}")
    logging.info(f"Baseline RMSE: {-fitness_function(baseline)}")


if __name__ == "__main__":
    main()
