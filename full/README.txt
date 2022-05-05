MultilabelKernelPerceptronDisk.py runs a modified version of the kernel
perceptorn algorithm proposed in MultilabelKernelPerceptron on the full MNIST
dataset. The execution takes quite a lot of time, roughly 4/5 hours.

Inside the file is possible to tweak the variables to save the results elsewhere
or change the order in which the approaches are tested, this to avoid having to
compute the kernel matrix more times than it's necessary.

To achieve a good accuracy the optimal polynomial degree and epochs are
extracted for ../results, in which the hyperparameter tuning and the test error
experiments helped determining the best combinations for each approach.

In the future would be smart to implement some kind of concurrent solution to
reduce execution time.
