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

        # Predictor coefficients
        alpha = torch.zeros(self.xs.shape[0], device=self.device)

        # Sum of the predictors in the ensemble
        alpha_sum = torch.zeros(self.xs.shape[0], device=self.device)

        # L'idea è di tenere il predittore con training error minimo, l'errore di questo predittore
        # e l'errore del predittore corrente, se quest'ultimo è minore dell'errore del predittore con trainin error minimo
        # allora dichiaro il predittore corrente come predittore di training error minimo

        # Prediction score for every element in the training set
        # previously p-score
        alpha_min_score = torch.zeros(self.xs.shape[0], device=self.device)

        # Predictor with the lower training error
        alpha_min = alpha

        current_score = alpha_min_score

        # one-vs-all encoding label transformation
        y_train_norm = sgn_label(self.ys, label)

        for epoch in range(self.epochs):

            for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
                alpha_update = sgn(torch.sum(alpha * y_train_norm * kernel_row)) != label_norm
                alpha[index] += alpha_update

                alpha_sum += alpha
                current_score += y_train_norm * kernel_row

                if update and torch.sum(sgn(alpha_min_score) != y_train_norm) > torch.sum(sgn(current_score) != y_train_norm):
                    alpha_min = alpha
                    alpha_min_score = current_score

                # if alpha_update and torch.sum(sgn(p_score) != y_train_norm) < torch.sum(sgn(p_score + y_train_norm * kernel_row) != y_train_norm):
                #     alpha_min = alpha
                
                # p_score += y_train_norm * kernel_row


        alpha_mean = alpha_sum / (self.epochs * kernel_matrix.shape[0])

        return alpha_min

    def fit(self):
        """
        Fits the model based on the training set.
        To do this runs a perceptron for each provided label.
        """

        self.model = torch.empty((len(self.labels), self.xs.shape[0]), device=self.device)
        kernel_matrix = self.kernel(self.xs, self.xs.T)

        for label in self.labels:
            # Compute a binary predictor for each label (ova encoding)
            self.model[label] = self.__fit_label(label, kernel_matrix)

    def error(self, xs, ys, kernel_matrix=None):
        """
        Evaluates prediction error on the given data and label sets using the fitted or given model.
        Is possible to provide a kernel_matrix for repeated evaluations on the same xs.
        Can be used to test training and test error alike.
        """

        if kernel_matrix is None:
            kernel_matrix = self.kernel(xs, self.xs.T)

        scores = torch.zeros((self.model.shape[0], xs.shape[0]), device=self.device)
        for label, alpha in enumerate(self.model):
            # Compute the prediction score for each of the 10 binary classifiers
            scores[label] = torch.sum(kernel_matrix * sgn_label(self.ys, label) * alpha, 1)

        # Classify using the highest score
        predictions = torch.argmax(scores, 0)

        # Compute error
        return float(torch.sum(predictions != ys) / ys.shape[0])
