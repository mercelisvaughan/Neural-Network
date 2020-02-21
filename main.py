import numpy.matlib
import numpy as np
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1


print("Random synaptic weights: ")
print(synaptic_weights)

# does only one iteration 
for iteration in range(1):

  input_layer = training_inputs
  output = sigmoid(np.dot(input_layer, synaptic_weights))

#print("outputs after training  =", output)
def dsigmoid(y):
  return y * (1.0 - y)
