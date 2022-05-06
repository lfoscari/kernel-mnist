Naive implementation of the multilabel kernel perceptron for the MNIST dataset.
To speed-up computation a K-mean approach is used to reduce dimensionality
without loosing too much information. The experiments can be replicated by
running the following commands inside the cloned repository:

$ nix-shell
& ./experiments.py

When writing new tests or experimenting is imperative to set the seed for the
RNG in PyTorch, because otherwise the model will be scrambled. Simply add the
instruction torch.manual_seed(~~~).

Inside utils.py is possible to define the number and size of reductions and the
range of epochs and kernel degree. By default only the reduction to 200 examples
is used.

In the 'full' directory is possible to run a modified version of the algorithm
on the whole MNIST dataset.