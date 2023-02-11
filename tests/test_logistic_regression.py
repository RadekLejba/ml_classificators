from typing import Generator
import pytest
import numpy as np

from regression_models.logistic_regression import LogisticRegressionClassificator


class TestLogisticRegressionClassificator:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.x = np.random.rand(50, 3)
        self.y = np.array(list(self.fun(self.x)))

    def fun(self, features_list: np.ndarray) -> Generator:
        function = lambda x: 2 * x[0] + x[1] - x[2] + 1
        for element in features_list:
            yield 1 if (function(element) > 2) else 0

    def test_linear_regression_minimalize_cost(self):
        model = LogisticRegressionClassificator(
            learning_rate=0.1, max_iter=1000, regularization_factor=0.1
        )
        model.fit(self.x, self.y)

        for i in range(1, len(model.history)):
            assert model.history[i - 1] > model.history[i]
        assert model.converged_at is not None

    def test_linear_regression_learning_rate_to_high(self):
        model = LogisticRegressionClassificator(
            learning_rate=10, max_iter=1000, regularization_factor=1
        )
        model.fit(self.x, self.y)

        assert model.history[0] < model.history[-1]
        assert model.converged_at is None
