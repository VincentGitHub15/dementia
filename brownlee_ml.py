import csv
import numpy as np
from random import seed
from random import random
from random import randrange
from math import exp


# getting data from CSV
with open('donor_info_clean.csv') as f:
	array = csv.reader(f)
	data = []

	for row in array:
		row_list=[]
		for i in row:
			row_list.append(int(i))
		data.append(row_list)
	# print(data)


# # Load a CSV file
# def load_csv(filename):
# 	dataset = list()
# 	with open(filename, 'r') as file:
# 		csv_reader = reader(file)
# 		for row in csv_reader:
# 			if not row:
# 				continue
# 			dataset.append(row)
# 	return dataset

# # Convert string column to float
# def str_column_to_float(dataset, column):
# 	for row in dataset:
# 		row[column] = float(row[column].strip())

# # Convert string column to integer
# def str_column_to_int(dataset, column):
# 	class_values = [row[column] for row in dataset]
# 	unique = set(class_values)
# 	lookup = dict()
# 	for i, value in enumerate(unique):
# 		lookup[value] = i
# 	for row in dataset:
# 		row[column] = lookup[row[column]]
# 	return lookup

# # Find the min and max values for each column
# def dataset_minmax(dataset):
# 	minmax = list()
# 	stats = [[min(column), max(column)] for column in zip(*dataset)]
# 	return stats

# # Rescale dataset columns to the range 0-1
# def normalize_dataset(dataset, minmax):
# 	for row in dataset:
# 		for i in range(len(row)-1):
# 			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	# print("cross val split", dataset_split)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
# *args = learn rate, iterations, hidden neurons
def evaluate_algorithm(dataset, back_prop, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = back_prop(train_set, test_set, *args)
		# print("predicted", predicted)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	# print("in train network", train)
	for epoch in range(n_epoch):
		sum_error=0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		# print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	# print("back prop fcn", test)
	# print("back prop fcn", train)
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)


# Test Backprop on Seeds dataset
# seed(1)
# # load and prepare data
# filename = 'seeds_dataset.csv'
# dataset = load_csv(filename)
# for i in range(len(dataset[0])-1):
# 	str_column_to_float(dataset, i)
# # convert class column to integers
# str_column_to_int(dataset, len(dataset[0])-1)
# # normalize input variables
# minmax = dataset_minmax(dataset)
# normalize_dataset(dataset, minmax)
# evaluate algorithm
dataset = [[78, 0, 0, 6, 56, 1, 3, 1, 6, 3, 1], [92, 1, 0, 8, 16, 1, 3, 1, 5, 3, 1], [97, 1, 0, 8, 0, 0, 3, 0, 3, 2, 0], [87, 1, 0, 8, 26, 1, 1, 1, 1, 1, 0], [97, 0, 0, 8, 62, 1, 3, 1, 5, 3, 0], [86, 1, 0, 9, 0, 0, 3, 0, 6, 3, 1], [89, 1, 0, 9, 0, 0, 1, 0, 3, 1, 1], [92, 1, 0, 9, 0, 0, 3, 0, 6, 3, 1], [92, 1, 0, 10, 22, 1, 1, 1, 3, 1, 1], [85, 1, 1, 10, 72, 1, 3, 1, 4, 2, 1], [81, 1, 0, 10, 0, 0, 0, 0, 1, 1, 0], [92, 1, 0, 10, 0, 0, 1, 0, 4, 1, 0], [86, 0, 0, 10, 9, 1, 1, 1, 4, 1, 0], [97, 0, 0, 11, 0, 0, 3, 0, 5, 3, 1], [86, 0, 0, 11, 26, 1, 0, 1, 4, 1, 1], [92, 0, 0, 11, 87, 1, 0, 1, 4, 0, 0], [88, 1, 0, 12, 0, 0, 0, 0, 1, 1, 1], [92, 1, 0, 12, 0, 0, 0, 0, 3, 1, 1], [92, 1, 0, 12, 0, 0, 2, 0, 3, 2, 1], [92, 1, 0, 12, 0, 0, 2, 0, 2, 1, 1], [100, 0, 0, 12, 0, 0, 3, 0, 5, 3, 1], [85, 1, 1, 12, 0, 0, 1, 0, 6, 2, 1], [88, 0, 1, 12, 0, 0, 0, 0, 3, 1, 1], [77, 1, 0, 12, 12, 1, 3, 1, 6, 3, 1], [87, 0, 0, 12, 83, 1, 0, 1, 0, 0, 1], [88, 1, 0, 12, 18, 1, 0, 1, 1, 1, 1], [88, 0, 0, 12, 81, 1, 0, 1, 4, 1, 1], [92, 1, 0, 12, 21, 1, 3, 1, 6, 3, 1], [100, 0, 0, 12, 89, 1, 2, 1, 6, 2, 1], [92, 0, 1, 12, 79, 1, 3, 1, 6, 3, 1], [89, 0, 0, 12, 13, 1, 3, 1, 6, 3, 1], [84, 1, 0, 12, 0, 0, 2, 0, 3, 2, 0], [89, 0, 0, 12, 0, 0, 1, 0, 5, 2, 0], [92, 0, 0, 12, 0, 0, 3, 0, 4, 2, 0], [92, 0, 0, 12, 0, 0, 2, 0, 5, 2, 0], [86, 0, 0, 12, 0, 0, 2, 0, 4, 2, 0], [92, 1, 0, 12, 68, 1, 1, 1, 1, 1, 0], [97, 1, 1, 12, 16, 1, 2, 1, 5, 2, 0], [83, 0, 0, 13, 33, 1, 2, 1, 4, 2, 1], [92, 0, 0, 13, 67, 1, 2, 1, 4, 2, 1], [97, 1, 0, 13, 0, 0, 2, 0, 1, 2, 0], [92, 1, 0, 13, 11, 1, 2, 2, 1, 2, 0], [97, 0, 0, 13, 78, 1, 3, 1, 6, 3, 0], [97, 1, 1, 13, 89, 1, 2, 1, 3, 2, 0], [82, 0, 0, 14, 0, 0, 2, 0, 2, 1, 1], [97, 0, 0, 14, 0, 0, 3, 0, 6, 3, 1], [92, 1, 1, 14, 0, 0, 3, 0, 6, 3, 1], [87, 0, 0, 14, 0, 0, 2, 0, 6, 2, 1], [92, 0, 0, 14, 47, 1, 3, 1, 4, 2, 1], [82, 0, 1, 14, 15, 1, 1, 1, 3, 1, 1], [92, 0, 1, 14, 3, 1, 1, 2, 5, 2, 1], [78, 0, 0, 14, 0, 0, 1, 0, 2, 1, 0], [86, 1, 0, 14, 0, 0, 0, 0, 2, 1, 0], [88, 1, 0, 14, 0, 0, 1, 0, 1, 1, 0], [92, 0, 0, 14, 0, 0, 0, 0, 3, 1, 0], [92, 0, 0, 14, 0, 0, 1, 0, 2, 1, 0], [92, 1, 0, 14, 0, 0, 1, 0, 3, 1, 0], [78, 1, 0, 14, 24, 1, 1, 2, 2, 1, 0], [86, 1, 0, 14, 20, 1, 1, 1, 3, 1, 0], [85, 1, 1, 14, 13, 1, 1, 1, 3, 1, 0], [88, 1, 1, 14, 86, 1, 0, 1, 3, 1, 0], [97, 1, 0, 15, 0, 0, 0, 0, 4, 1, 1], [100, 1, 0, 15, 10, 1, 3, 2, 5, 3, 1], [89, 0, 0, 15, 0, 0, 1, 0, 4, 1, 0], [92, 0, 0, 15, 0, 0, 1, 0, 4, 1, 0], [92, 0, 1, 15, 0, 0, 3, 0, 3, 2, 0], [89, 1, 0, 15, 22, 1, 1, 1, 3, 1, 0], [92, 0, 0, 15, 75, 1, 1, 1, 2, 1, 0], [85, 1, 1, 16, 0, 0, 3, 0, 5, 3, 1], [86, 0, 1, 16, 0, 0, 0, 0, 3, 1, 1], [88, 1, 1, 16, 0, 0, 3, 0, 5, 3, 1], [92, 1, 1, 16, 0, 0, 0, 0, 1, 1, 1], [100, 1, 1, 16, 0, 0, 3, 0, 4, 2, 1], [84, 1, 0, 16, 23, 1, 0, 2, 0, 0, 1], [89, 0, 0, 16, 6, 1, 1, 1, 2, 1, 1], [78, 1, 0, 16, 0, 0, 0, 0, 2, 0, 0], [78, 1, 0, 16, 0, 0, 2, 0, 3, 2, 0], [82, 0, 0, 16, 0, 0, 1, 0, 3, 1, 0], [87, 1, 0, 16, 0, 0, 0, 0, 1, 1, 0], [97, 0, 0, 16, 0, 0, 0, 0, 1, 1, 0], [97, 1, 0, 16, 0, 0, 0, 0, 2, 1, 0], [89, 0, 1, 16, 0, 0, 1, 0, 3, 1, 0], [79, 1, 0, 16, 13, 1, 2, 3, 3, 2, 0], [84, 1, 0, 16, 47, 1, 2, 1, 1, 2, 0], [92, 1, 0, 16, 15, 1, 0, 2, 6, 2, 0], [97, 0, 0, 16, 89, 1, 1, 1, 1, 1, 0], [100, 0, 0, 16, 89, 1, 1, 1, 4, 1, 0], [81, 1, 0, 16, 23, 1, 1, 2, 1, 1, 0], [100, 0, 0, 17, 0, 0, 3, 0, 5, 3, 1], [89, 0, 0, 17, 63, 1, 2, 1, 6, 2, 1], [88, 1, 1, 17, 7, 1, 2, 1, 6, 2, 1], [81, 1, 0, 17, 21, 1, 1, 1, 3, 1, 0], [87, 1, 0, 17, 80, 1, 3, 1, 6, 3, 0], [97, 1, 0, 17, 12, 1, 2, 1, 5, 2, 0], [79, 1, 0, 18, 0, 0, 0, 0, 2, 0, 1], [84, 1, 0, 18, 0, 0, 2, 0, 4, 2, 1], [78, 1, 0, 18, 0, 0, 1, 0, 2, 1, 0], [97, 0, 0, 18, 0, 0, 2, 0, 3, 2, 0], [100, 0, 0, 19, 89, 1, 2, 1, 5, 2, 1], [92, 1, 0, 21, 15, 1, 2, 1, 5, 2, 1], [92, 1, 0, 21, 88, 1, 1, 3, 3, 1, 1], [78, 1, 0, 21, 0, 0, 1, 0, 3, 1, 0], [87, 1, 0, 21, 0, 0, 0, 0, 0, 0, 0], [81, 1, 1, 21, 0, 0, 2, 0, 3, 2, 0], [78, 1, 0, 21, 12, 1, 0, 1, 0, 0, 0], [78, 1, 0, 21, 69, 1, 1, 2, 2, 1, 0], [97, 1, 0, 21, 18, 1, 1, 1, 3, 1, 0]]

train= [[79, 1, 0, 18, 0, 0, 0, 0, 2, 0, 1], [97, 0, 0, 18, 0, 0, 2, 0, 3, 2, 0], [92, 0, 1, 15, 0, 0, 3, 0, 3, 2, 0], [92, 1, 0, 9, 0, 0, 3, 0, 6, 3, 1], [89, 0, 0, 15, 0, 0, 1, 0, 4, 1, 0], [89, 0, 0, 17, 63, 1, 2, 1, 6, 2, 1], [97, 1, 1, 13, 89, 1, 2, 1, 3, 2, 0], [92, 0, 0, 15, 75, 1, 1, 1, 2, 1, 0], [97, 1, 0, 13, 0, 0, 2, 0, 1, 2, 0], [86, 1, 0, 14, 0, 0, 0, 0, 2, 1, 0], [87, 0, 0, 12, 83, 1, 0, 1, 0, 0, 1], [78, 1, 0, 16, 0, 0, 2, 0, 3, 2, 0], [92, 1, 0, 12, 0, 0, 2, 0, 3, 2, 1], [87, 1, 0, 8, 26, 1, 1, 1, 1, 1, 0], [92, 1, 0, 12, 68, 1, 1, 1, 1, 1, 0], [92, 0, 0, 14, 0, 0, 1, 0, 2, 1, 0], [92, 1, 0, 14, 0, 0, 1, 0, 3, 1, 0], [92, 1, 0, 12, 21, 1, 3, 1, 6, 3, 1], [78, 1, 0, 18, 0, 0, 1, 0, 2, 1, 0], [87, 1, 0, 21, 0, 0, 0, 0, 0, 0, 0], [97, 0, 0, 16, 0, 0, 0, 0, 1, 1, 0], [97, 0, 0, 11, 0, 0, 3, 0, 5, 3, 1], [92, 0, 1, 14, 3, 1, 1, 2, 5, 2, 1], [88, 1, 1, 14, 86, 1, 0, 1, 3, 1, 0], [92, 1, 0, 21, 15, 1, 2, 1, 5, 2, 1], [78, 0, 0, 14, 0, 0, 1, 0, 2, 1, 0], [92, 1, 0, 13, 11, 1, 2, 2, 1, 2, 0], [92, 1, 0, 12, 0, 0, 2, 0, 2, 1, 1], [78, 1, 0, 21, 69, 1, 1, 2, 2, 1, 0], [79, 1, 0, 16, 13, 1, 2, 3, 3, 2, 0], [87, 1, 0, 16, 0, 0, 0, 0, 1, 1, 0], [89, 0, 0, 16, 6, 1, 1, 1, 2, 1, 1], [88, 1, 1, 16, 0, 0, 3, 0, 5, 3, 1], [97, 0, 0, 16, 89, 1, 1, 1, 1, 1, 0], [89, 1, 0, 15, 22, 1, 1, 1, 3, 1, 0], [89, 0, 0, 12, 13, 1, 3, 1, 6, 3, 1], [97, 1, 0, 21, 18, 1, 1, 1, 3, 1, 0], [92, 0, 0, 15, 0, 0, 1, 0, 4, 1, 0], [97, 1, 0, 17, 12, 1, 2, 1, 5, 2, 0], [97, 1, 0, 16, 0, 0, 0, 0, 2, 1, 0], [92, 1, 0, 21, 88, 1, 1, 3, 3, 1, 1], [87, 1, 0, 17, 80, 1, 3, 1, 6, 3, 0], [100, 1, 1, 16, 0, 0, 3, 0, 4, 2, 1], [97, 1, 0, 8, 0, 0, 3, 0, 3, 2, 0], [86, 0, 0, 12, 0, 0, 2, 0, 4, 2, 0], [92, 0, 0, 13, 67, 1, 2, 1, 4, 2, 1], [84, 1, 0, 16, 47, 1, 2, 1, 1, 2, 0], [88, 1, 0, 14, 0, 0, 1, 0, 1, 1, 0], [82, 0, 1, 14, 15, 1, 1, 1, 3, 1, 1], [82, 0, 0, 16, 0, 0, 1, 0, 3, 1, 0], [88, 0, 1, 12, 0, 0, 0, 0, 3, 1, 1], [92, 0, 0, 14, 0, 0, 0, 0, 3, 1, 0], [81, 1, 1, 21, 0, 0, 2, 0, 3, 2, 0], [87, 0, 0, 14, 0, 0, 2, 0, 6, 2, 1], [78, 1, 0, 16, 0, 0, 0, 0, 2, 0, 0], [92, 1, 0, 16, 15, 1, 0, 2, 6, 2, 0], [85, 1, 1, 10, 72, 1, 3, 1, 4, 2, 1], [78, 0, 0, 6, 56, 1, 3, 1, 6, 3, 1], [85, 1, 1, 16, 0, 0, 3, 0, 5, 3, 1], [84, 1, 0, 12, 0, 0, 2, 0, 3, 2, 0], [97, 1, 1, 12, 16, 1, 2, 1, 5, 2, 0], [88, 1, 0, 12, 0, 0, 0, 0, 1, 1, 1], [86, 1, 0, 9, 0, 0, 3, 0, 6, 3, 1], [92, 0, 1, 12, 79, 1, 3, 1, 6, 3, 1], [85, 1, 1, 14, 13, 1, 1, 1, 3, 1, 0], [92, 0, 0, 12, 0, 0, 2, 0, 5, 2, 0], [84, 1, 0, 18, 0, 0, 2, 0, 4, 2, 1], [88, 1, 0, 12, 18, 1, 0, 1, 1, 1, 1], [89, 1, 0, 9, 0, 0, 1, 0, 3, 1, 1], [97, 1, 0, 15, 0, 0, 0, 0, 4, 1, 1], [78, 1, 0, 21, 12, 1, 0, 1, 0, 0, 0], [100, 0, 0, 17, 0, 0, 3, 0, 5, 3, 1], [92, 1, 0, 10, 0, 0, 1, 0, 4, 1, 0], [100, 1, 0, 15, 10, 1, 3, 2, 5, 3, 1], [97, 0, 0, 13, 78, 1, 3, 1, 6, 3, 0], [92, 0, 0, 11, 87, 1, 0, 1, 4, 0, 0], [81, 1, 0, 16, 23, 1, 1, 2, 1, 1, 0], [92, 1, 0, 12, 0, 0, 0, 0, 3, 1, 1], [92, 1, 1, 16, 0, 0, 0, 0, 1, 1, 1], [92, 0, 0, 14, 47, 1, 3, 1, 4, 2, 1], [82, 0, 0, 14, 0, 0, 2, 0, 2, 1, 1], [86, 0, 0, 11, 26, 1, 0, 1, 4, 1, 1], [86, 0, 0, 10, 9, 1, 1, 1, 4, 1, 0], [77, 1, 0, 12, 12, 1, 3, 1, 6, 3, 1], [78, 1, 0, 21, 0, 0, 1, 0, 3, 1, 0]]

test= [[92, 1, 0, 8, 16, 1, 3, 1, 5, 3, 1], [89, 0, 1, 16, 0, 0, 1, 0, 3, 1, 0], [83, 0, 0, 13, 33, 1, 2, 1, 4, 2, 1], [86, 1, 0, 14, 20, 1, 1, 1, 3, 1, 0], [85, 1, 1, 12, 0, 0, 1, 0, 6, 2, 1], [88, 1, 1, 17, 7, 1, 2, 1, 6, 2, 1], [81, 1, 0, 17, 21, 1, 1, 1, 3, 1, 0], [88, 0, 0, 12, 81, 1, 0, 1, 4, 1, 1], [92, 1, 1, 14, 0, 0, 3, 0, 6, 3, 1], [92, 1, 0, 10, 22, 1, 1, 1, 3, 1, 1], [100, 0, 0, 12, 0, 0, 3, 0, 5, 3, 1], [78, 1, 0, 14, 24, 1, 1, 2, 2, 1, 0], [92, 0, 0, 12, 0, 0, 3, 0, 4, 2, 0], [100, 0, 0, 16, 89, 1, 1, 1, 4, 1, 0], [100, 0, 0, 12, 89, 1, 2, 1, 6, 2, 1], [86, 0, 1, 16, 0, 0, 0, 0, 3, 1, 1], [97, 0, 0, 14, 0, 0, 3, 0, 6, 3, 1], [81, 1, 0, 10, 0, 0, 0, 0, 1, 1, 0], [100, 0, 0, 19, 89, 1, 2, 1, 5, 2, 1], [84, 1, 0, 16, 23, 1, 0, 2, 0, 0, 1], [89, 0, 0, 12, 0, 0, 1, 0, 5, 2, 0], [97, 0, 0, 8, 62, 1, 3, 1, 5, 3, 0]]

n_folds = 11
l_rate = 0.1
n_epoch = 500
n_hidden = 15

# back_propagation(train, test, l_rate, n_epoch, n_hidden)
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
