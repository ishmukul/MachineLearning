## Unsupervised learning - KMeans algorithm

Kmeans algorithm used in unsupervised learning has been used on Old Faithful geyser dataset available on Kaggle,  'Faithful.csv'.  
<img src="https://github.com/ishmukul/MachineLearning/blob/master/KMeans-Scratch/plots/data_raw.png" alt="Raw data" width="300"/>

Algorithm has been written with scratch and the results are compared with sklearn toolkit.  

First I tried with sklearn. <img src="https://github.com/ishmukul/MachineLearning/blob/master/KMeans-Scratch/plots/cluster_kmeans_sklearn_scaled.png" alt="Clusters -sklearn" width="300"/>    
Then, from my implementation. <img src="https://github.com/ishmukul/MachineLearning/blob/master/KMeans-Scratch/plots/cluster_kmeans_scaled.png" alt="Clusters -sklearn" width="300"/>  


## File structure  
**[KMeans.py](https://github.com/ishmukul/MachineLearning/blob/master/KMeans-Scratch/KMeans.py)**:    
1) A class for kmeans algorithm implemented from scratch, named KMeansScratch.  
2) Results are identical to KMeans from sklearn.  
3) There is an option to generate plots at each step. Plots at each step for k=2 are saved in cluster_step folder.  
4) Added **[scree plot](https://github.com/ishmukul/MachineLearning/blob/master/KMeans-Scratch/plots/kmeans_scree_plot.png)** 
for finding optimal number of clusters which comes out to be 2.  
<img src="https://github.com/ishmukul/MachineLearning/blob/master/KMeans-Scratch/plots/kmeans_scree_plot.png" alt="Raw data" width="300"/>

