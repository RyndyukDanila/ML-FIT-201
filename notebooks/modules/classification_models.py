import numpy as np


class KNN(object):
	"""description for KNN"""
	def __init__(self, k=1):
		self.k = k;


	@property
	def k(self):
		return self._k

	@k.setter
	def k(self, value):
		if value < 0:
			raise ValueError("K cannot be negative.")
		self._k = int(value)


	def _mode(self, values):
		vals, counts = np.unique(values, return_counts=True)
		index = np.argmax(counts)
		return vals[index]


	def fit(self, X, y):
		self.X = X
		self.y = y


	def predict(self, new_data):
		predictions = []

		for point in new_data:
			distances = np.linalg.norm(self.X - point, axis=1)

			nearest_neighbor_ids = distances.argsort()[:self.k]
			nearest_neighbor_classes = self.y.iloc[nearest_neighbor_ids]

			predictions.append(self._mode(nearest_neighbor_classes))

		return predictions