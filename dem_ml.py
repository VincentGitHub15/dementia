import numpy as np
import csv

with open('donor_info_clean.csv') as f:
	array = csv.reader(f)
	data = []
	has_dementia = []

	# data = [row for row in array]

	for row in array:
		# taking all info except last column
		info = row[:-1]

		# data.append(info)
		data.append(info)


		#only including last column for dementia
		dementia = row[10]
		has_dementia.append(dementia)

	#removing headers
	data.pop(0)
	has_dementia.pop(0)

	# creating a nested list of integers for dementia
	array_dem = []
	for i in has_dementia:
		j = []
		j.append(int(i))
		# print(i)
		array_dem.append(j)

	# print(data)
	print(array_dem)

data = np.array(data, dtype=float)
# X = np.array(([3,5], [5,1], [10,2]), dtype=float)

class Neural_Net(object):

	def __init__(self):
		#Define Hyperparameters
		self.inputLayerSize = 10
		self.outputLayerSize = 1
		self.hiddenLayerSize = 11
		
		#Weights (parameters)
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
		
	def forward(self, data):
		#Propagate inputs though network
		self.z2 = np.dot(data, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		print(yHat)
		return yHat
		
	def sigmoid(self, z):
		#Apply sigmoid activation function to scalar, vector, or matrix
		return 1/(1+np.exp(-z))
	
Neural_Net().forward(data)
