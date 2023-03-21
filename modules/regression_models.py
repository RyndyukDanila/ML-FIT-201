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

		
class DTR(object):
	"""docstring for DTR"""
	def __init__(self, mtd=5, mnr=1):
		self.mtd = mtd # Maximum Tree Depth
		self.mnr = mnr # Minimum Node Records
		self.tree = None

	@property
	def mtd(self):
		return self._mtd

	@mtd.setter
	def mtd(self, value):
		if value < 0:
			raise ValueError("Maximum Tree Depth cannot be negative.")
		self._mtd = int(value)

	@property
	def mnr(self):
		return self._mnr

	@mnr.setter
	def mnr(self, value):
		if value < 0:
			raise ValueError("Minimum Node Records cannot be negative.")
		self._mnr = int(value)
		
	# Split a dataset based on an attribute and an attribute value
	def _test_split(self, index, value, dataset):
		left, right = list(), list()
		for row in dataset:
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)
		return left, right
	 
	# Calculate the Gini index for a split dataset
	def _gini_index(self, groups, classes):
		# count all samples at split point
		n_instances = float(sum([len(group) for group in groups]))
		# sum weighted Gini index for each group
		gini = 0.0
		for group in groups:
			size = float(len(group))
			# avoid divide by zero
			if size == 0:
				continue
			score = 0.0
			# score the group based on the score for each class
			for class_val in classes:
				p = [row[-1] for row in group].count(class_val) / size
				score += p * p
			# weight the group score by its relative size
			gini += (1.0 - score) * (size / n_instances)
		return gini
	 
	# Select the best split point for a dataset
	def _get_split(self, dataset):
		class_values = list(set(row[-1] for row in dataset))
		b_index, b_value, b_score, b_groups = np.inf, np.inf, np.inf, None
		for index in range(len(dataset[0])-1):
			for row in dataset:
				groups = self._test_split(index, row[index], dataset)
				gini = self._gini_index(groups, class_values)
				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, row[index], gini, groups
		return {'index':b_index, 'value':b_value, 'groups':b_groups}
	 
	# Create a terminal node value
	def _to_terminal(self, group):
		outcomes = [row[-1] for row in group]
		return np.mean(outcomes)
	 
	# Create child splits for a node or make terminal
	def _split(self, node, max_depth, min_size, depth):
		left, right = node['groups']
		del(node['groups'])
		# check for a no split
		if not left or not right:
			node['left'] = node['right'] = self._to_terminal(left + right)
			return
		# check for max depth
		if depth >= max_depth:
			node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
			return
		# process left child
		if len(left) <= min_size:
			node['left'] = self._to_terminal(left)
		else:
			node['left'] = self._get_split(left)
			self._split(node['left'], max_depth, min_size, depth+1)
		# process right child
		if len(right) <= min_size:
			node['right'] = self._to_terminal(right)
		else:
			node['right'] = self._get_split(right)
			self._split(node['right'], max_depth, min_size, depth+1)
	 
	# Make a prediction with a decision tree
	def _prediction(self, node, row):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict):
				return self._prediction(node['left'], row)
			else:
				return node['left']
		else:
			if isinstance(node['right'], dict):
				return self._prediction(node['right'], row)
			else:
				return node['right']
	 
	# Build a decision tree
	def fit(self, data):
		root = self._get_split(data)
		self._split(root, self.mtd, self.mnr, 1)
		self.tree = root

	# Classification and Regression Tree Algorithm
	def predict(self, X):
		predictions = list()
		for row in X:
			prediction = self._prediction(self.tree, row)
			predictions.append(prediction)
		return predictions