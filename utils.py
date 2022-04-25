import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    DEVICE = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

SEED = hash("Ford... you're turning into a penguin. Stop it.") % 2 ** 32

TRAINING_SET_SIZE = 60_000
TEST_SET_SIZE = 10_000

EPOCHS = range(1, 11)
DEGREES = range(1, 7)

REDUCTIONS = [200, 1000, 1500]
APPROACHES = ["min", "mean"]

RESULTS_DIR = "./results"
RESULTS_TEMPLATE = {
    r: {
        a: {
            # "training_error": None
            "training_error_km": None,
            "test_error": None,
            "epochs": None,
            "degree": None
        } for a in ["min", "mean"]
    } for r in REDUCTIONS
}


def sgn_label(a, b):
    """
    Maps each label in the first argument to the 1 iff it is equal to the corresponding label in the second argument,
    -1 otherwise.
    """
    return (a == b) * 2 - 1


def sgn(a):
    """
    Maps each value of the argument to 1 iff is greater than zero, -1 otherwise.
    """
    return (a > 0) * 2 - 1


def polynomial(a, b, degree=5.):
    """
    Calculates the polynomial kernel.
    """
    return torch.float_power(a @ b + 1, degree)
