from utils import sgn_label, sgn
from dataclasses import dataclass
from typing import Callable
import torch


@dataclass(repr=False)
class MultilabelKernelPerceptron:
    kernel: Callable
    labels: list
    epochs: int
    xs: torch.Tensor
    ys: torch.Tensor
    approach: str
    device: torch.device
    model: torch.Tensor = None

    def _fit_label_min(self, label, kernel_matrix):
        """
        The core implementation of the perceptron with One vs. All encoding.
        Given the label and the kernel matrix runs a kernel perceptron and returns
        the classifier with minimum error among all classifiers in the generated ensemble.
        """

        # one-vs-all encoding label transformation
        y_train_norm = sgn_label(self.ys, label)

        # Predictor coefficients
        alpha = torch.zeros(self.xs.shape[0], device=self.device)

        # Keep track of the score of the predictor
        alpha_score = torch.zeros(self.xs.shape[0], device=self.device)

        # Predictor with the lower training error
        alpha_min = alpha.detach().clone()

        # Keep track of the training error of such predictor
        alpha_error = int(torch.sum(sgn(alpha_score) != y_train_norm))
        alpha_min_error = alpha_error

        for epoch in range(self.epochs):
            for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
                alpha_update = sgn(torch.sum(alpha * y_train_norm * kernel_row)) != label_norm
                alpha[index] += alpha_update

                if not alpha_update:
                    continue

                alpha_score += label_norm * kernel_row
                alpha_error = int(torch.sum(sgn(alpha_score) != y_train_norm))

                if alpha_error < alpha_min_error:
                    alpha_min = alpha.detach().clone()
                    alpha_min_error = alpha_error

                if alpha_min_error == 0:
                    break

        return alpha_min

    def _fit_label_mean(self, label, kernel_matrix):
        """
        The core implementation of the perceptron with One vs. All encoding.
        Given the label and the kernel matrix runs a kernel perceptron and returns
        the mean classifier among all classifiers in the generated ensemble.
        """

        # one-vs-all encoding label transformation
        y_train_norm = sgn_label(self.ys, label)

        # Predictor coefficients
        alpha = torch.zeros(self.xs.shape[0], device=self.device)

        # Sum of the predictors in the ensemble
        alpha_sum = alpha.detach().clone()

        for epoch in range(self.epochs):
            for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
                alpha_update = sgn(torch.sum(alpha * y_train_norm * kernel_row)) != label_norm
                alpha[index] += alpha_update
                alpha_sum += alpha

        alpha_mean = alpha_sum / (self.epochs * kernel_matrix.shape[0])

        return alpha_mean

    def _fit_label_weight(self, label, kernel_matrix):
        """
        The core implementation of the perceptron with One vs. All encoding.
        Given the label and the kernel matrix runs a kernel perceptron and returns
        the classifier obtained by making an average of the classifiers
        in the generated ensemble weighed on their training error.
        """

        # one-vs-all encoding label transformation
        y_train_norm = sgn_label(self.ys, label)

        # Predictor coefficients
        alpha = torch.zeros(self.xs.shape[0], device=self.device)

        # Keep track of the score of such predictor
        alpha_score = torch.zeros(self.xs.shape[0], device=self.device)

        # Keep track of the training error of such predictor
        # To avoid extra calculations the error is kept absolute
        alpha_error = int(torch.sum(sgn(alpha_score) != y_train_norm))

        # Classifier of the weighted sum (initially zero)
        alpha_weighted_sum = (self.xs.shape[0] - alpha_error) * alpha

        # Sum of the accuracy for every classifier
        alpha_total_accuracy = self.xs.shape[0] - alpha_error

        for epoch in range(self.epochs):
            for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
                alpha_update = sgn(torch.sum(alpha * y_train_norm * kernel_row)) != label_norm
                alpha[index] += alpha_update

                if not alpha_update:
                    continue

                alpha_score += label_norm * kernel_row
                alpha_error = int(torch.sum(sgn(alpha_score) != y_train_norm))

                alpha_weighted_sum += (self.xs.shape[0] - alpha_error) * alpha
                alpha_total_accuracy += self.xs.shape[0] - alpha_error

        alpha_weighted_mean = alpha_weighted_sum / alpha_total_accuracy

        return alpha_weighted_mean

    def _fit_label_last(self, label, kernel_matrix):
        """
        The core implementation of the perceptron with One vs. All encoding.
        Given the label and the kernel matrix runs a kernel perceptron and returns
        the last classifiers added to the ensemble
        """

        # one-vs-all encoding label transformation
        y_train_norm = sgn_label(self.ys, label)

        # Predictor coefficients
        alpha = torch.zeros(self.xs.shape[0], device=self.device)

        for epoch in range(self.epochs):
            for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
                alpha[index] += sgn(torch.sum(alpha * y_train_norm * kernel_row)) != label_norm

        return alpha

    def fit(self):
        """
        Fits the model based on the training set.
        To do this runs a perceptron for each provided label.
        """

        kernel_matrix = self.kernel(self.xs, self.xs.T)
        self.model = torch.empty((len(self.labels), self.xs.shape[0]), device=self.device)

        for label in self.labels:
            if self.approach == "min":
                self.model[label] = self._fit_label_min(label, kernel_matrix)
            elif self.approach == "mean":
                self.model[label] = self._fit_label_mean(label, kernel_matrix)
            elif self.approach == "weight":
                self.model[label] = self._fit_label_weight(label, kernel_matrix)
            elif self.approach == "last":
                self.model[label] = self._fit_label_last(label, kernel_matrix)
            else:
                raise AttributeError(self.approach)

    def error(self, xs, ys, kernel_matrix=None):
        """
        Evaluates prediction error on the given data and label sets using the trained model.
        Is possible to provide a kernel matrix for repeated evaluations on the same set.
        Can be used to test training and test error alike.
        """

        if kernel_matrix is None:
            kernel_matrix = self.kernel(xs, self.xs.T)

        ys_norm = torch.stack([sgn_label(self.ys, label) for label in self.labels])
        scores = torch.zeros((self.model.shape[0], xs.shape[0]), device=self.device)

        for label, alpha in enumerate(self.model):
            # Compute the prediction score for each of the 10 binary classifiers
            scores[label] = torch.sum(kernel_matrix * ys_norm[label] * alpha, 1)

        # Classify using the highest score
        predictions = torch.argmax(scores, 0)

        # Compute error
        return float(torch.sum(predictions != ys) / ys.shape[0])
