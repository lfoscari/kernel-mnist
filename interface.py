class Predictor():
	def fit(self): pass
	def predict(self, x_test, y_test): pass

	@staticmethod
	def sgn_label(a, b):
		return (a == b) * 2 - 1

	@staticmethod
	def sgn(a):
		return (a > 0) * 2 - 1