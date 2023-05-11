import numpy as np
import numpy.typing as npt
import pytest

from src.linear_regression import compute_fitness, filter_data, train


@pytest.fixture
def features() -> npt.NDArray:
    yield np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def labels() -> npt.NDArray:
    yield np.array([6, 15, 24])


def test_filter_data(features: npt.NDArray) -> None:
    mask = np.array([True, False, True])
    selected_features = filter_data(features, mask)
    assert (selected_features == np.array([[1, 3], [4, 6], [7, 9]])).all()


def test_train(features: npt.NDArray, labels: npt.NDArray) -> None:
    model = train(features, labels)
    assert np.isclose(model.intercept_, 0)
    np.testing.assert_allclose(model.coef_, np.array([1, 1, 1]))


def test_compute_fitness(features: npt.NDArray, labels: npt.NDArray) -> None:
    chromosome = np.array([True, True, True])
    assert np.isclose(compute_fitness(chromosome, features, labels), 0)
