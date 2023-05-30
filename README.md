# Machine_Learning
Building machine Learning models using Pytorch
# K Means Clustering Algorithm: 
 1. Most commonly used for Exploratory Data Analysis when we encounter any new datafile. Also it's Unsupervised learning model which further more makes this algorithm to use more often. It's a iterative algorithm that works by partitioning the data points into k many distinct non-overlapping groups by minimizing the intercluster variation.
 2. Convergence guaranteed: Yes, it’s guaranteed to converge. K-means works by minimising the within cluster variation. Upon every iteration K-means find the cluster’s centre in such a way that the with-in cluster variation is minimised. So, upon a certain number of steps it finds the cluster centres that don’t change after new iteration irrespective of whether it belongs to global minima or local minima.
 3. Global Optima Always: No, We can’t always find global optima using the K-means algorithm. Since we areusing alternate Optimization where we optimise rxk keeping µk values as constant andvice-versa. It is minimising the within cluster variation. Depends on the initial clustercentres how the end clustering looks like. Cluster centres once formed, unable tomove within the clusters so very sensitive to initialization and we need to initialisewith different cluster centres and pick the one with least cost function.
  4. Upon generating a random data-set and using my model to predict the output; here are the results, ![image](https://github.com/LokeshSreenathJ/Machine_Learning/assets/115972450/ec784035-14ea-47a0-a6b0-dd3f4c11a536)

