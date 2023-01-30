import numpy as np

from numpy import linalg


class LinearRegression:
    """
    Linear regression class
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit a linear regression model using ordinary least squares
        """
        X_1 = np.vstack((X.T, np.ones(X.shape[0]))).T
        beta_hat = np.dot(np.dot(linalg.inv(np.dot(X_1.T, X_1)), X_1.T), y)
        self.w = beta_hat[:-1]
        self.b = beta_hat[-1]

    def predict(self, X: np.array) -> np.array:
        """
        Predict a feature matrix X using saved model parameters
        """
        y_hat = np.dot(self.w, X.T) + self.b
        return y_hat


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit a linear regressor using gradient descent.

        Arguments:
            X (np.ndarray): Input data
            y (np.ndarray): Target labels
            lr (float): Learning rate
            epochs (int): number of training epochs

        Returns:
            self
        """
        self.w = np.random.rand(X.shape[1])
        self.b = 0

        for i in range(epochs):
            y_pred = self.predict(X)
            resids = y_pred - y
            w_grad = np.mean(np.dot(resids, self.w))
            b_grad = resids

            self.w -= lr * w_grad
            self.b -= lr * b_grad

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return self.w.T @ X + self.b * np.ones(X.shape[0])
