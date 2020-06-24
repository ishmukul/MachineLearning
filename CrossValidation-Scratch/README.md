# Cross validation techniques
A power technique for validation of model in supervised learning is Cross Validation (CV). 
CV consists of different methods out of which 3 are commonly used:   
a) Hold-out method: a portion of data is left for test.   
b) k-fold Cross Validation: data divided into k groups and (k-1) groups used for training.    
c) Leave Out One Cross Validation (LOOCV): K=n in this case.   

These methods have been tried by writing from scratch and comparing with sklearn toolkit.  
The idea is to get the working of these methods behind the scene.  

Algorithms from scratch took longer than sklearn, may be because of less optimization.

**Dataset: [BreastCancer.csv](https://github.com/ishmukul/MachineLearning/blob/master/data/BreastCancer.csv) in data folder.**  
    The data provides different features for identifying Cancer.

LOOCV is most time consuming as it loops over whole dataset.