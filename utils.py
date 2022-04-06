import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    DEVICE = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

SEED = hash("Ford... you're turning into a penguin. Stop it.")

EPOCHS = range(1, 11)
DEGREES = range(1, 7)

RESULTS_TEMPLATE = {
    "epochs": {
        e: {
            "degree": {
                d: {
                    "training_time": None,
                    "training_error": None,
                    "test_error": None
                }
                for d in DEGREES
            }
        }
        for e in EPOCHS
    }
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


def polynomial(a, b, c=1., degree=5.):
    """
    Calculates the polynomial kernel.
    """
    return torch.float_power(a @ b + c, degree)


def save_to_csv(data, filepath):
    """
    Saves the test results, structured as a RESULT_TEMPLATE, in csv format to the specified filepath.
    """

    import csv

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["epochs", "degree", "training_time", "training_error", "test_error"]

        writer.writerow(header)

        for (e, a) in data["epochs"].items():
            for (d, b) in a["degree"].items():
                writer.writerow((e, d, b["training_time"], b["training_error"], b["test_error"]))

        print(f"CSV results saved in {csvfile.name}")
