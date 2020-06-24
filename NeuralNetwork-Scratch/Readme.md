A simple two layer Neural Network from scratch to solve XOR problem, which is not linearly separable. 
Python numpy library has been used for generating arrays.  

Class **NeuralNet** should be instantiated first.  

Class methods could be used to get:  
 **get_weights**: Outputs weights and bias of fitted function  
 **get_steps**: Outputs number of steps to reach precision. n_steps is also returned from **fit** function itself.  

## Input
X = (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)    
Y = (-0.5), (0.5), (0.5), (-0.5)  

## Output  
Plot of cost function is saved at [Cost_NN.png](https://github.com/ishmukul/MachineLearning/blob/master/NeuralNetwork-Scratch/Cost_NN.png?raw=true).  
![Cos function](https://github.com/ishmukul/MachineLearning/blob/master/NeuralNetwork-Scratch/Cost_NN.png?raw=true)

Since seed number is fixed, output of Python script will remain same:  
N_neurons:2, N_steps:215, Output:[[-0.5  0.5  0.5 -0.5]]  
N_neurons:4, N_steps:96, Output:[[-0.5  0.5  0.5 -0.5]]  
N_neurons:6, N_steps:92, Output:[[-0.5  0.5  0.5 -0.5]]  
N_neurons:8, N_steps:87, Output:[[-0.5  0.5  0.5 -0.5]]  
N_neurons:10, N_steps:108, Output:[[-0.5  0.5  0.5 -0.5]]   

It seems that Layer with 8 neurons converges fastest.  

A simple 2 layer neural network can solve XOR problem easily, even with just 2 neurons.  
We will try for more hidden layers in next update.  
