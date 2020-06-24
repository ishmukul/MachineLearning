## Unsupervised learning - KMeans algorithm

Kmeans algorithm used in unsupervised learning has been used on Old Faithful geyser dataset available on Kaggle,  'Faithful.csv'.  

Algorithm has been written with scratch and the results are compared with sklearn toolkit.  

First I tried with sklearn. Fig files saved as png.  
Going to write own implementation soon. Updates coming soon.

## File structure  
**KMeans.py**:  
1) A class for kmeans algorithm implemented from scratch, named KMeansScratch.  
2) Results are identical to KMeans from sklearn.  
3) There is an option to generate plots at each step. Plots at each step for k=2 are saved in cluster_step folder.  
4) Added **scree plot** for finding optimal number of clusters. 2 comes out to be nice number.