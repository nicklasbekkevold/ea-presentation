from typing import Callable

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_data_set_from_csv(path="data/data_set.csv") -> tuple[npt.NDArray, npt.NDArray]:
    data_set = np.array(np.loadtxt(path, delimiter=","))
    return data_set[:, :-1], data_set[:, -1]


def train(features: npt.NDArray, labels: npt.NDArray) -> LinearRegression:
    return LinearRegression().fit(features, labels)


def filter_data(data: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    return np.delete(data, np.where(~mask), axis=1)


def compute_negative_rmse(
    chromosome: npt.NDArray,
    features: npt.NDArray,
    labels: npt.NDArray,
) -> float:
    selected_features = filter_data(features, chromosome)
    x_train, x_test, y_train, y_test = train_test_split(
        selected_features, labels, test_size=0.2
    )
    model = train(x_train, y_train)
    y_pred = model.predict(x_test)
    return -mean_squared_error(y_test, y_pred, squared=False)


def print_chromosome(chromosome: npt.NDArray) -> str:
    return "".join(chromosome.astype(int).astype(str).tolist())


def hash(chromosome: npt.NDArray) -> int:
    return int(print_chromosome(chromosome), 2)


def create_fitness_function() -> Callable[[npt.NDArray], float]:
    features, labels = load_data_set_from_csv()

    cache = {}

    def memoized_fitness_function(chromosome: npt.NDArray) -> float:
        if hash(chromosome) in cache:
            return cache[hash(chromosome)]
        fitness = compute_negative_rmse(chromosome, features, labels)
        cache[hash(chromosome)] = fitness
        return fitness

    return memoized_fitness_function
