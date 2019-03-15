import numpy as np
import sklearn.preprocessing
import cvxopt

# ==============================================================================
#     This class is a port of the MATLAB code IntervalPredictorModel from
#     OpenCossan
#
#     2018, Jonathan Sadeghi, COSSAN Working Group,
#     University~of~Liverpool, United Kingdom
#     See also:  http://cossan.co.uk/wiki/index.php/@IntervalPredictorModel
# ==============================================================================


class PyIPM:
    def __init__(self, polynomial_degree=1):
        self.polynomial_degree = polynomial_degree
        assert (type(self.polynomial_degree) == int), 'polynomial_degree parameter must be integer'
        self.n_features = None
        self.n_data_points = None
        self.input_scale = None
        self.n_terms = None
        self.param_vector = None

    def fit(self, training_input, training_output):
        self.n_features = training_input.shape[1]
        self.n_data_points = training_input.shape[0]

        assert (training_output.shape == (
            self.n_data_points,)), 'Number of input examples must equal number of output examples'

        self.input_scale = np.mean(np.abs(training_input), axis=0)
        training_input = training_input / self.input_scale

        poly = sklearn.preprocessing.PolynomialFeatures(self.polynomial_degree)
        basis = poly.fit_transform(training_input)
        self.n_terms = basis.shape[1]

        basis_sum = np.mean(np.absolute(basis), axis=0)
        objective = np.concatenate((-basis_sum, basis_sum))

        constraint_matrix = np.zeros((2 * self.n_data_points + self.n_terms, 2 * self.n_terms))

        constraint_matrix[:self.n_data_points, :self.n_terms] = -(basis - np.absolute(basis)) / 2
        constraint_matrix[self.n_data_points:-self.n_terms, :self.n_terms] = (basis + np.absolute(basis)) / 2
        constraint_matrix[:self.n_data_points, self.n_terms:] = -(basis + np.absolute(basis)) / 2
        constraint_matrix[self.n_data_points:-self.n_terms, self.n_terms:] = (basis - np.absolute(basis)) / 2

        constraint_matrix[-self.n_terms:, :self.n_terms] = np.eye(self.n_terms)
        constraint_matrix[-self.n_terms:, self.n_terms:] = -np.eye(self.n_terms)

        b = np.zeros(((2 * self.n_data_points + self.n_terms), 1))
        b[:2 * self.n_data_points, 0] = np.hstack((-training_output, training_output))

        sol = cvxopt.solvers.lp(cvxopt.matrix(objective), cvxopt.matrix(constraint_matrix), cvxopt.matrix(b))

        self.param_vector = np.array(sol['x'])

        return self

    def predict(self, test_input):
        if self.param_vector is None:
            raise RuntimeError("You must train IPM before predicting data!")

        assert (test_input.shape[1] == self.n_features), 'The provided test data has the wrong number of features'

        test_input = test_input / self.input_scale

        poly = sklearn.preprocessing.PolynomialFeatures(self.polynomial_degree)
        basis = poly.fit_transform(test_input)

        upper_bound = 0.5 * np.dot(
            np.hstack((basis - np.absolute(basis), basis + np.absolute(basis))),
            self.param_vector)
        lower_bound = 0.5 * np.dot(
            np.hstack((basis + np.absolute(basis), basis - np.absolute(basis))),
            self.param_vector)

        return upper_bound, lower_bound

    def get_model_reliability(self, confidence=1 - 10 ** -6):
        if confidence < 0 or confidence > 1:
            print('Invalid confidence parameter value')
        else:
            return 1 - 2 * self.n_terms / ((self.n_data_points + 1) * confidence)
