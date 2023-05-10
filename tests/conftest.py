import numpy as np


def pytest_sessionstart(session):
    np.random.seed(42)
