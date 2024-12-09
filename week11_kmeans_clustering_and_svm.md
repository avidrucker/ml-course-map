# Week 11: SVM & KMeans Clustering
**Clustering (topic)**  
- (assertion) K-means clustering groups data into K clusters to minimize within-cluster sum of squares (inertia).  
- (assertion) Increasing the number of centroids typically decreases inertia since points are closer to their assigned cluster center.  
- (assertion) The elbow method involves plotting inertia against K and selecting the “elbow” point that balances a low inertia with a small number of clusters.

- (task) Implement K-means clustering on a given dataset and vary the number of clusters to observe changes in inertia and identify an appropriate K using the elbow method.

**Support Vector Classifier (SVC) (topic)** (as referenced in week 11 exercises)  
- (assertion) Hyper-parameters like C and kernel parameters must be selected carefully to ensure good generalization of the SVC model.  
- (task) Use GridSearchCV or similar hyper-parameter tuning methods to select the best C or kernel for SVC.
