import numpy as np
import csv
from scipy import optimize

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
	# print(array_dem)

data = np.array(data, dtype=float)
# X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(array_dem, dtype=float)

class Neural_Net:

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

	def sigmoidPrime(self,z):
		#Gradient of sigmoid
		return np.exp(-z)/((1+np.exp(-z))**2)
	
	def costFunction(self, data, y):
		#Compute cost for given data,y, use weights already stored in class.
		self.yHat = self.forward(data)
		J = 0.5*sum((y-self.yHat)**2)
		return J
		
	def costFunctionPrime(self, data, y):
		#Compute derivative with respect to W and W2 for a given data and y:
		self.yHat = self.forward(data)
		
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)
		
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(data.T, delta2)  
		
		return dJdW1, dJdW2

	#Helper Functions for interacting with other classes:
	def getParams(self):
		#Get W1 and W2 unrolled into vector:
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params
	
	def setParams(self, params):
		#Set W1 and W2 using single paramater vector.
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
		
	def computeGradients(self, data, y):
		dJdW1, dJdW2 = self.costFunctionPrime(data, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

	def computeNumericalGradient(N, data, y):
		# print("N", N)
		paramsInitial = N.getParams()
		numgrad = np.zeros(paramsInitial.shape)
		perturb = np.zeros(paramsInitial.shape)
		e = 1e-4

		for p in range(len(paramsInitial)):
			#Set perturbation vector
			perturb[p] = e
			N.setParams(paramsInitial + perturb)
			loss2 = N.costFunction(data, y)
			
			N.setParams(paramsInitial - perturb)
			loss1 = N.costFunction(data, y)

			#Compute Numerical Gradient
			numgrad[p] = (loss2 - loss1) / (2*e)

			#Return the value we changed to zero:
			perturb[p] = 0
			
		#Return Params to original value:
		N.setParams(paramsInitial)

		return numgrad 




class trainer:
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

	
n = Neural_Net().computeNumericalGradient(data, y)
print(n)
# Neural_Net().forward(data)


# print("\nnormalize ",norm(grad-numgrad)/norm(grad+numgrad))
# compgrad = Neural_Network().computeGradients(self, data, y):
# print(compgrad)

# numgrad = computeNumericalGradient(Neural_Network(), data, y)
# grad = Neural_Network().computeGradients(data, y)

# print("numgrad ",numgrad)
# print("\ngrad ",grad)

# print("\nnormalize ",norm(grad-numgrad)/norm(grad+numgrad))

