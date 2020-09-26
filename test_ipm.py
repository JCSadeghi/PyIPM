import pytest
import numpy as np
import PyIPM

np.random.seed(0)


@pytest.fixture
def x() -> np.ndarray:
    """
    Some input
    Returns: 2 Dimensional input array with 100 data points

    """
    return 5 * (np.random.rand(100, 2) - 0.5)


@pytest.fixture
def y(x: np.ndarray) -> np.ndarray:
    """
    Some 1 dimensional output which is a function of the input
    Args:
        x: Some 2 dimensional Input

    Returns:
        Output array

    """
    return x[:, 0] ** 2 + x[:, 1] ** 2 * np.random.rand(x.shape[0])


@pytest.fixture
def model(x: np.ndarray, y: np.ndarray) -> PyIPM.IPM:
    """
    Create a 2 degree IPM from training data
    Args:
        x: Input data
        y: Output data

    Returns:
        Trained IPM

    """
    model = PyIPM.IPM(polynomial_degree=2)

    model.fit(x, y)

    return model


class TestIPM:
    def test_prediction(self, model: PyIPM.IPM, x: np.ndarray):
        """
        Ensure that the predicted IPM bounds match stored values
        Args:
            model: Trained IPM
            x: Input data

        """
        upper_bound, lower_bound = model.predict(x)

        tolerance = 0.01

        assert(abs(upper_bound[0] - 1.15) < tolerance), "Upper bound does not match stored value"

        assert(abs(lower_bound[0] - 0.15) < tolerance), "Lower bound does not match stored value"

    def test_reliability(self, model: PyIPM.IPM):
        """
        Ensure model reliability matches stored value
        Args:
            model: Trained IPM

        """
        reliability = model.get_model_reliability()

        tolerance = 0.01

        assert(abs(reliability - 0.68) < tolerance), "Model reliability does not match stored value"
