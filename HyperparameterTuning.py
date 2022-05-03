from functools import partial

from utils import *
from MultilabelKernelPerceptron import MultilabelKernelPerceptron
from MNIST import label_set


def tune(xs, ys, reduction, approach):
    # Keeping the same split as MNIST
    validation_size = int(reduction * 1 / 7)

    x_train, y_train = xs[:-validation_size], ys[:-validation_size]
    x_val, y_val = xs[validation_size:], ys[validation_size:]

    results = []

    for epochs in EPOCHS:
        for degree in DEGREES:
            perceptron = MultilabelKernelPerceptron(
                partial(polynomial, degree=degree),
                label_set,
                epochs,
                x_train,
                y_train,
                approach,
                DEVICE
            )

            perceptron.fit()
            validation_error = perceptron.error(x_val, y_val)
            results.append((validation_error, epochs, degree))

    return min(results, key=lambda x: x[0])[1:]
