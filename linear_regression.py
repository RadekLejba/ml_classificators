from copy import deepcopy
from typing import Optional, Tuple

import numpy as np


class LinearRegressionClassificator:
    converge_treshold: float = 0.01
    converged_at: Optional[int]
    dim1: int = 0
    dim2: int = 0

    def __init__(
        self,
        learning_rate: float,
        max_iter: int,
        initial_w: Optional[np.ndarray] = None,
    ) -> None:
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.b = 0.0
        self.w: np.ndarray = initial_w if initial_w else np.zeros(1)

        self.history: list = []
        self.converged_at = None

    def _set_dimensions(self, x: np.ndarray) -> None:
        self.dim1, self.dim2 = x.shape

    def _set_w(self) -> None:
        self.w = np.zeros(self.dim2)

    def model(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.w) + self.b

    def converge_test(self) -> None:
        try:
            result = abs(self.history[-2] - self.history[-1]) <= self.converge_treshold
            if result and not self.converged_at:
                self.converged_at = len(self.history)
        except IndexError:
            pass

    def calculate_cost(self, X: np.ndarray, y: np.ndarray) -> None:
        cost = 0.0
        for i in range(self.dim1):
            f_wb_i = np.dot(X[i], self.w) + self.b
            cost = cost + (f_wb_i - y[i]) ** 2
        cost = cost / (2 * self.dim1)

        self.history.append(cost)
        self.converge_test()

    def calculate_gradient(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        w_copy = deepcopy(self.w)
        b_copy = deepcopy(self.b)

        for i in range(self.dim1):
            prediction_error = self.model(x[i]) - y[i]
            for z in range(self.dim2):
                w_copy[z] = w_copy[z] + prediction_error * x[i, z]
            b_copy = b_copy + prediction_error

        return (b_copy / self.dim1), (w_copy / self.dim1)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._set_dimensions(x)
        self._set_w()

        for _ in range(self.max_iter):
            b, w = self.calculate_gradient(x, y)
            self.w -= self.learning_rate * w
            self.b -= self.learning_rate * b
            self.calculate_cost(x, y)
