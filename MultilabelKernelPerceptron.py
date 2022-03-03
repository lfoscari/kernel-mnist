from utils import sgn_label, sgn
from dataclasses import dataclass
from typing import Callable
import torch


@dataclass(repr=False)
class MultilabelKernelPerceptron:
    kernel: Callable
    labels: list
    epochs: int
    x_train: torch.Tensor
    y_train: torch.Tensor

    def __fit_label(self, label, kernel_matrix):
        alpha = torch.zeros(self.x_train.shape[0])
        y_train_norm = sgn_label(self.y_train, label)

        for _ in range(self.epochs):
            update = False

            for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
                alpha_update = sgn(torch.sum(alpha * y_train_norm * kernel_row, 0)) != label_norm
                update = alpha_update or update
                alpha[index] += alpha_update

            if not update:
                break

        return alpha

    def fit(self):
        self.model = torch.empty((len(self.labels), self.x_train.shape[0]))
        kernel_matrix = self.kernel(self.x_train, self.x_train.T)

        for label in self.labels:
            self.model[label] = self.__fit_label(label, kernel_matrix)

    def predict(self, xs, ys):
        kernel_matrix = self.kernel(xs, self.x_train.T)
        scores = [torch.sum(kernel_matrix * sgn_label(self.y_train, label) * alpha, 1)
                  for label, alpha in enumerate(self.model)]
        predictions = torch.max(torch.stack(scores), 0)[1]
        return float(torch.sum(predictions != ys) / ys.shape[0])
