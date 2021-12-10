class Predictor():
	def fit(self): pass
	def predict(self, xs, ys): pass

	@staticmethod
	def sgn_label(a, b):
		return (a == b) * 2 - 1

	@staticmethod
	def sgn(a):
		return (a > 0) * 2 - 1

EPOCHS=range(1, 11)
DEGREES=range(1, 7)

RESULTS_WITH_DEGREE = {
	"epochs": {
		e: {
			"degree" : {
				d: {
					"training_time": None,
					"training_error": None,
					"test_error": None
				}
				for d in DEGREES
			}
		}
		for e in EPOCHS
	},
}

RESULTS = {
	"epochs": {
		e: {
			"training_time": None,
			"training_error": None,
			"test_error": None
		}
		for e in EPOCHS
	}
}