from typing import Optional

import numpy as np

np.random.seed(1)


class LogisticRegressionClassificator:
    def __init__(
        self,
        learning_rate: float,
        regularization_factor: float,
        max_iter: int,
        initial_w: Optional[np.ndarray] = None,
    ):
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.b: float = 0.0
        self.w: np.ndarray = initial_w if initial_w else np.zeros(1)
        self.regularization_factor: float = regularization_factor
        self.history: list = []
        self.converged_at: Optional[int] = None
        self.converge_treshold: float = 0.001
        self.dim1: int = 0
        self.dim2: int = 0

    def _set_dimensions(self, x: np.ndarray) -> None:
        self.dim1, self.dim2 = x.shape

    def _set_w(self) -> None:
        self.w = np.zeros(self.dim2)

    def converge_test(self) -> None:
        try:
            result = abs(self.history[-2] - self.history[-1]) <= self.converge_treshold
            if result and not self.converged_at:
                self.converged_at = len(self.history)
        except IndexError:
            pass

    def model(self, x: np.ndarray):
        return np.dot(x, self.w) + self.b

    def sigmoid(self, x: np.ndarray):
        z = self.model(x)

        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: float) -> float:
        sigmoid = self.sigmoid(x)

        return (-y * np.log(sigmoid)) - ((1 - y) * np.log(1 - sigmoid))

    def calculate_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        cost = 0.0
        for i in range(self.dim1):
            cost += self.loss(X[i], y[i])

        cost = cost / self.dim1

        self.history.append(cost)
        self.converge_test()

        return cost

    def regularized_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.calculate_cost(X, y) + self.regular_term()

    def regular_term(self) -> float:
        return (self.regularization_factor / (2 * self.dim1)) * (np.sum(self.w**2))

    def compute_logistic_gradient(self, X, y):
        w_copy = np.zeros(self.dim2)
        b_copy = 0.0

        for i in range(self.dim1):
            sigmoid = self.sigmoid(X[i])
            err = sigmoid - y[i]
            for j in range(self.dim2):
                w_copy[j] += err * X[i, j]
            b_copy += err

        return b_copy / self.dim1, w_copy / self.dim1

    def compute_regularized_logistic_gradient(self, X: np.ndarray, y: np.ndarray):
        b, w = self.compute_logistic_gradient(X, y)

        for i in range(self.dim2):
            w[i] += (self.regularization_factor / self.dim1) * self.w[i]

        return b, w

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._set_dimensions(X)
        self._set_w()
        for _ in range(self.max_iter):
            b, w = self.compute_regularized_logistic_gradient(X, y)
            self.b -= self.learning_rate * b
            self.w -= self.learning_rate * w
            self.calculate_cost(X, y)

    def predict(self, x: np.ndarray) -> int:
        return 1 if self.model(x) >= 0.5 else 0
