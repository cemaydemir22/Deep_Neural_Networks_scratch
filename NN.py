import numpy as np
import random as rd
import math

class NeuralNetwork:
  def __init__(self,layer_sizes,activation_fun):
    """
      Initialization of layers can be done in the following way-> [a0,a1,...,ak,ak+1] where a0,..,ak+1 are positive integers,
      and this list shows that input layer has a0 nodes the network has k hidden nodes and the hidden layer ai where i=1,...,k has ai nodes and the output layer has ak+1 nodes.
      and the activation function can be initilalized as a list of strings where strings are elements of the set {"relu","softmax","sigmoid"} and softmax can only be used at the final layer
      activation_fun is of the form ["relu","sigmoid","softmax"] as an  example for a neural network of 4 layers (input+2hidden+output)

    """
    self.layer_sizes=layer_sizes
    self.activation_func=activation_fun
    self.bias=[]
    self.weights=[]
    self.activations=[None]*len(layer_sizes)
    self.net_values=[None]*(len(layer_sizes)-1)
    self.dWeights=[None]*(len(layer_sizes)-1)
    self.dBiases=[None]*(len(layer_sizes)-1)
    self.errors=[None]*(len(layer_sizes)-1)


    for i in range(len(layer_sizes) - 1):
            
            if activation_fun[i] == "relu":
                scale = np.sqrt(2 / layer_sizes[i])
            else:
                scale = np.sqrt(1 / layer_sizes[i])
                
            self.bias.append(np.zeros((layer_sizes[i+1], 1)))

            self.weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]) * scale)



  def forward_pass(self,input_data):
    self.activations[0]=input_data
    for i in range((len(self.layer_sizes))-1):
      self.net_values[i]=np.add(np.dot(self.weights[i],self.activations[i]),self.bias[i])
      if self.activation_func[i]=="sigmoid":
        self.activations[i+1]=self.sigmoid(self.net_values[i])

      elif self.activation_func[i]=="relu":
        self.activations[i+1]=self.relu(self.net_values[i])

      elif self.activation_func[i]=="softmax":
        self.activations[i+1]=self.softmax(self.net_values[i])
    return self.activations[-1]

  def sigmoid(self, values):
    return 1 / (1 + np.exp(-values))

  def relu(self, values):
    return np.maximum(0, values)

  def softmax(self, values):
    exp_vals = np.exp(values - np.max(values))
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)


  def calculate_loss(self,target):
    """ either uses categorical cross entropy  or MSE depending on the activation function of the last layer"""
    output=self.activations[-1]
    eps=1e-10
    loss=0
    if self.activation_func[-1]=="softmax":
      loss = -np.sum(target * np.log(output + eps))


    else:
      for i in range(len(self.activations[-1])):
        loss = 0.5 * np.sum((target - output)**2)


    return loss

  def derivative_relu(self, data):
        return np.where(data >= 0, 1, 0)

  def derivative_sigmoid(self, data):
        s = self.sigmoid(data)
        return s * (1 - s)

  def backpropagation(self,input_data,target_data,learning_rate):
    self.forward_pass(input_data)
    output=self.activations[-1]
    target=target_data
    loss=self.calculate_loss(target)

    if self.activation_func[-1]=="softmax":
      error_L=np.subtract(output,target)
      self.errors[-1]=error_L
    else:
       error=np.subtract(output-target)
       if self.activation_func[-1]=="relu":
         error_L=np.multiply(error,self.derivative_relu(self.net_values[-1]))
         self.errors[-1]=error_L
       else:
         error_L=np.multiply(error,self.derivative_sigmoid(self.net_values[-1]))
         self.errors[-1]=error_L

    for l in range(len(self.layer_sizes)-3,-1,-1):
      if self.activation_func[l]=="relu":
       error_l=np.multiply(np.matmul(np.transpose(self.weights[l+1]),self.errors[l+1]),self.derivative_relu(self.net_values[l]))
       self.errors[l]=error_l
      else:
        error_l=np.multiply(np.matmul(np.transpose(self.weights[l+1]),self.errors[l+1]),self.derivative_sigmoid(self.net_values[l]))
        self.errors[l]=error_l
    self.update_weights(learning_rate)

  def update_weights(self, learning_rate):
    # First, compute gradients for all layers.
    for l in range(len(self.weights)):
        self.dWeights[l] = np.matmul(self.errors[l], np.transpose(self.activations[l]))
        self.dBiases[l] = self.errors[l]
    # Then, update weights and biases.
    for l in range(len(self.layer_sizes) - 1):
        self.weights[l] = np.subtract(self.weights[l], learning_rate * self.dWeights[l])
        self.bias[l] = np.subtract(self.bias[l], learning_rate * self.dBiases[l])






