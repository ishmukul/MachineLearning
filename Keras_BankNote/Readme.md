# Keras models for classification

Keras based neural network classifier application for Banknote authentication dataset.  

**Data Source**: [UCI website](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)  

**Data Set Information:**  

Data were extracted from images that were taken from genuine and forged banknote-like 
specimens. For digitization, an industrial camera usually used for print inspection was 
used. The final images have 400x 400 pixels. Due to the object lens and distance to the 
investigated object gray-scale pictures with a resolution of about 660 dpi were gained.
 Wavelet Transform tool were used to extract features from images.  

**Attribute Information:**

1. variance of Wavelet Transformed image (continuous)  
2. skewness of Wavelet Transformed image (continuous)  
3. curtosis of Wavelet Transformed image (continuous)  
4. entropy of image (continuous)  
5. class (integer)  

## Approaches  

Data was scaled with mean=0, sd=1, and split in hold-out of 85:15 ratio.  

Applied oversampling from [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/index.html) 
for addressing class imbalance issue.  

Four ML models have been trained in this work and tested on the hold-out set.      
1) Support Vector Machine: Gaussian Kernel.    
2) Random Forest Classifier:  Depth 5.  
3) Decision Tree Classifier:  Depth 5.  
4) Neural Network:  
Input Layer (ReLU) -> 2x Hidden Layer (ReLU) -> Drop layer -> Output Layer (Sigmoid).  
  There are options for different Loss and optimizer functions (commented in the code).  
  Keras model was run over 20 epochs in batch size=32.  

Accuracy metric were collected from :  
a) Confusion matrix  
b) F1 score  

## Output  

By comparing with different classifiers, it seems SVM and Neural Networks are producing similar results. It may be 
possible that data is separable and SVM works fine. Other classifiers are also giving more than 95% accuracy.   

**Output** of Python script (result may differ in another run):  
Time for Support Vector Machine: 0.004812 s, Accuracy: 100.00, and F1 score = 1.00  
Time for Random Forrest Classifier: 0.140958 s, Accuracy: 98.83, and F1 score = 0.99  
Time for Decision Tree Classifier: 0.003153 s, Accuracy: 97.38, and F1 score = 0.97  
Time for Keras Neural Network: 1.231440 s, Accuracy: 1.00, and F1 score = 1.00   

## Updates:  

1) Added K-fold split for cross validation check. Accuracy is still close to 100% in this dataset.
