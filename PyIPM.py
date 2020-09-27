import typing
from typing import Optional
from typing import Tuple

import cvxopt
import numpy as np
import scipy.misc
import scipy.optimize
import scipy.stats
import sklearn.preprocessing


# ==============================================================================
#     This class is a port of the MATLAB code IntervalPredictorModel from
#     OpenCossan
#
#     2018, Jonathan Sadeghi, COSSAN Working Group,
#     University~of~Liverpool, United Kingdom
#     See also:  http://cossan.co.uk/wiki/index.php/@IntervalPredictorModel
# ==============================================================================


class IPM:
    """
    A class to train and query Interval Predictor Models
    """

    def __init__(self, polynomial_degree: int = 1):
        """
        Constructor for the Interval Predictor Model
        Args:
            polynomial_degree: Integer representing the degree of the fitted polynomial. Default 1.
        """
        self.polynomial_degree = polynomial_degree
        if not isinstance(self.polynomial_degree, int):
            raise ValueError(
                "Polynomial Degree must be integer but is {}".format(
                    self.polynomial_degree
                )
            )
        self.n_features: Optional[int] = None
        self.n_data_points: Optional[int] = None
        self.input_scale: Optional[float] = None
        self.n_terms: Optional[int] = None
        self.param_vector: Optional[np.ndarray] = None

    def fit(self, training_input: np.ndarray, training_output: np.ndarray):
        """
        Fit the Interval Predictor Model to Training Data
        Args:
            training_input: Array of IPM training inputs, dims: (n_samples x n_input_dimensions)
            training_output: Array of IPM training outputs, dims: (n_samples)

        """
        self.n_features = typing.cast(int, training_input.shape[1])
        self.n_data_points = typing.cast(int, training_input.shape[0])

        if not training_output.shape == (self.n_data_points,):
            raise ValueError(
                "Number of input examples must equal number of output examples"
            )

        self.input_scale = np.mean(np.abs(training_input), axis=0)
        training_input = training_input / self.input_scale

        poly = sklearn.preprocessing.PolynomialFeatures(self.polynomial_degree)
        basis = poly.fit_transform(training_input)
        self.n_terms = typing.cast(int, basis.shape[1])

        basis_sum = np.mean(np.absolute(basis), axis=0)
        objective = np.concatenate((-basis_sum, basis_sum))

        constraint_matrix = np.zeros(
            (2 * self.n_data_points + self.n_terms, 2 * self.n_terms)
        )

        constraint_matrix[: self.n_data_points, : self.n_terms] = (
            -(basis - np.absolute(basis)) / 2
        )
        constraint_matrix[self.n_data_points : -self.n_terms, : self.n_terms] = (
            basis + np.absolute(basis)
        ) / 2
        constraint_matrix[: self.n_data_points, self.n_terms :] = (
            -(basis + np.absolute(basis)) / 2
        )
        constraint_matrix[self.n_data_points : -self.n_terms, self.n_terms :] = (
            basis - np.absolute(basis)
        ) / 2

        constraint_matrix[-self.n_terms :, : self.n_terms] = np.eye(self.n_terms)
        constraint_matrix[-self.n_terms :, self.n_terms :] = -np.eye(self.n_terms)

        b = np.zeros(((2 * self.n_data_points + self.n_terms), 1))
        b[: 2 * self.n_data_points, 0] = np.hstack((-training_output, training_output))

        sol = cvxopt.solvers.lp(
            cvxopt.matrix(objective), cvxopt.matrix(constraint_matrix), cvxopt.matrix(b)
        )

        self.param_vector = np.array(sol["x"])

    def predict(self, test_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions from trained IPM
        Args:
            test_input: Array of IPM test inputs, dims: (n_samples x n_input_dimensions)

        Returns:
            Tuple containing upper and lower bound for test input
        """
        if self.param_vector is None:
            raise RuntimeError("You must train IPM before predicting data!")

        if not test_input.shape[1] == self.n_features:
            raise ValueError("The provided test data has the wrong number of features")

        test_input = test_input / self.input_scale

        poly = sklearn.preprocessing.PolynomialFeatures(self.polynomial_degree)
        basis = poly.fit_transform(test_input)

        upper_bound = 0.5 * np.dot(
            np.hstack((basis - np.absolute(basis), basis + np.absolute(basis))),
            self.param_vector,
        )
        lower_bound = 0.5 * np.dot(
            np.hstack((basis + np.absolute(basis), basis - np.absolute(basis))),
            self.param_vector,
        )

        return upper_bound, lower_bound

    def get_model_reliability_old(self, confidence: float = 1 - 10 ** -6) -> float:
        """
        Compute the reliability of the trained IPM's prediction interval using a conservative method
        Args:
            confidence: the confidence with which the reliability is prescribed, float between 0 and 1

        Returns:
            reliability of the trained IPM's prediction interval, float between 0 and 1

        """
        if self.n_terms is None or self.n_data_points is None:
            raise RuntimeError("IPM must be trained")
        else:
            self.n_terms = typing.cast(int, self.n_terms)
            self.n_data_points = typing.cast(int, self.n_data_points)
        if confidence < 0 or confidence > 1:
            raise ValueError(
                "Invalid confidence parameter value, must be between 0 and 1 but is {}".format(
                    confidence
                )
            )
        else:
            return 1 - 2 * self.n_terms / ((self.n_data_points + 1) * (1 - confidence))

    def get_model_reliability(self, confidence: float = 1 - 10 ** -6) -> float:
        """
        Compute the reliability of the trained IPM's prediction interval
        Args:
            confidence: the confidence with which the reliability is prescribed, float between 0 and 1

        Returns:
            reliability of the trained IPM's prediction interval, float between 0 and 1

        """
        if confidence < 0 or confidence > 1:
            raise ValueError(
                "Invalid confidence parameter value, must be between 0 and 1 but is {}".format(
                    confidence
                )
            )
        else:
            x0 = scipy.optimize.bisect(
                lambda epsilon: self.beta(epsilon) - (1 - confidence), 0, 1
            )
            reliability = 1 - x0
            return reliability

    def beta(self, epsilon: float) -> float:
        """
        A helper function to compute model reliability
        """
        if self.n_terms is None:
            raise RuntimeError("IPM must be trained")
        else:
            self.n_terms = typing.cast(int, self.n_terms)
        d = 2 * self.n_terms
        return scipy.stats.binom.cdf(d - 1, self.n_data_points, epsilon)
