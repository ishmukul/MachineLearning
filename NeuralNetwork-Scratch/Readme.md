A simple two layer Neural Network from scratch to solve XOR problem, which is not linearly separable. 
Python numpy library has been used for generating arrays.  

Classes are not used in this example since it was a simple 2 layer network. 
In next update, multi layer network will be implemented with class variables.  

## Input
X = (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)    
Y = (-0.5), (0.5), (0.5), (-0.5)  

## Output  
Plot of cost function is saved at Cost_NN.png.  

Since seed number is fixed, output of Python script will remain same:  
N_hidden_layers:2, N_steps:215, Output:[[-0.5  0.5  0.5 -0.5]]  
N_hidden_layers:4, N_steps:96, Output:[[-0.5  0.5  0.5 -0.5]]  
N_hidden_layers:6, N_steps:92, Output:[[-0.5  0.5  0.5 -0.5]]  
N_hidden_layers:8, N_steps:87, Output:[[-0.5  0.5  0.5 -0.5]]  
N_hidden_layers:10, N_steps:108, Output:[[-0.5  0.5  0.5 -0.5]]  

It seems that Layer 8 converges fastest.  

A simple 2 layer neural network can solve XOR problem easily, even with just 2 neurons.  
We will try for more hidden layers in next update.  