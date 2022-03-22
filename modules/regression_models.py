import numpy as np


class LR(object):
	"""docstring for LR"""
	def __init__(self):
		self._coef = []
		self._intercept = 1

	def _concatenate_for_intercept(self, X):
		X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

	def fit(self, X, y):
		self._concatenate_for_intercept(X)
		self._coef, *_ = np.linalg.lstsq(X, y, rcond=None) # least square method
		self._intercept = self._coef[0]

	def predict(self, test):
		predictions = []
		for row in test:
			yhat = sum(self._coef * row)
			predictions.append(yhat)
		return predictions
