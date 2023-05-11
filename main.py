import numpy as np

from src import ga, metrics, parameters
from src.linear_regression import create_fitness_function, print_chromosome


def main():
    np.random.seed(parameters.SEED)

    fitness_function = create_fitness_function()
    baseline = np.full((parameters.CHROMOSOME_LENGTH,), True)
    add_metrics, save_metrics = metrics.create_metrics_hooks(baseline, fitness_function)

    solution = ga.optimize(
        fitness_function, generational_hook=add_metrics, post_optimize_hook=save_metrics
    )
    print(f"Solution found: {print_chromosome(solution)}")
    print(f"Solution RMSE:  {-fitness_function(solution)}")
    print(f"Baseline RMSE:  {-fitness_function(baseline)}")


if __name__ == "__main__":
    main()
