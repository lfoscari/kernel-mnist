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
    device: torch.device
    model: torch.Tensor = None

    def __fit_label(self, label, kernel_matrix):
        """
        The core implementation of the perceptron with One vs. All encoding.
        Given the label and the kernel matrix runs a kernel perceptron and computes the misses-counter alpha.
        The procedure in incremental in the number of epochs.
        """

        # Can be shown that averaging the alpha vectors is equivalent to averaging the predictors
        alpha_means = torch.zeros((self.epochs, self.xs.shape[0]), device=self.device)
        y_train_norm = sgn_label(self.ys, label)

        for epoch in range(self.epochs):
            alpha = alpha_means[max(0, epoch - 1)]
            alpha_updates = torch.zeros((self.xs.shape[0], self.xs.shape[0]), device=self.device)

            for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
                alpha[index] += sgn(torch.sum(alpha * y_train_norm * kernel_row, 0)) != label_norm
                alpha_updates[index] = alpha

            alpha_means[epoch] = torch.mean(alpha_updates, 0)

        alpha_updates_error = torch.zeros(self.epochs, device=self.device)

        # Should I use the clusterized data or the original data for evaluating the training error?
        for epoch, alpha in enumerate(alpha_means):
            score = torch.sum(kernel_matrix * y_train_norm * alpha, 1)
            training_error = float(torch.sum(sgn(score) != y_train_norm)) / y_train_norm.shape[0]
            alpha_updates_error[epoch] = training_error

        lowest_error = torch.argmin(alpha_updates_error)
        return alpha_means[lowest_error]

    def fit(self):
        """
        Fits the model based on the training set.
        To do this runs a perceptron for each provided label.
        """

        self.model = torch.empty((len(self.labels), self.xs.shape[0]), device=self.device)
        kernel_matrix = self.kernel(self.xs, self.xs.T)

        for label in self.labels:
            self.model[label] = self.__fit_label(label, kernel_matrix)

    def error(self, xs, ys, model=None, kernel_matrix=None):
        """
        Evaluates prediction error on the given data and label sets using the fitted or given model.
        Is possible to provide a kernel_matrix for repeated evaluations on the same xs.
        Can be used to test training and test error alike.
        """

        if model is None:
            model = self.model

        if model is None:
            raise RuntimeError("You must fit or provide a model")

        if kernel_matrix is None:
            kernel_matrix = self.kernel(xs, self.xs.T)

        scores = torch.zeros((self.model.shape[0], xs.shape[0]), device=self.device)
        for label, alpha in enumerate(model):
            scores[label] = torch.sum(kernel_matrix * sgn_label(self.ys, label) * alpha, 1)

        predictions = torch.argmax(scores, 0)
        return float(torch.sum(predictions != ys) / ys.shape[0])
