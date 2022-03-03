import torch


def sgn_label(a, b):
    return (a == b) * 2 - 1


def sgn(a):
    return (a > 0) * 2 - 1


def polynomial(a, b, c=1., degree=5.):
    return torch.float_power(a @ b + c, degree)


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


def save_to_csv(data, file):
    import csv

    with open(f"./results/{file}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["epochs", "degree", "training_time", "training_error", "test_error"]

        writer.writerow(header)

        for (e, a) in data["epochs"].items():
            for (d, b) in a["degree"].items():
                writer.writerow((e, d, b["training_time"], b["training_error"], b["test_error"]))

        print(f"CSV results saved in {csvfile.name}")
