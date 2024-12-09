# Week 9: Random Forest, Scaling, & Impurity
**Random Forest (topic)**  
- (assertion) A random forest is an ensemble of decision trees that reduces variance by aggregating predictions from many trees trained on bootstrapped samples.  
- (assertion) No pruning is done in random forests; instead, randomness in feature selection at splits and averaging over many trees helps avoid overfitting.  
- (assertion) Random forests can handle a wide variety of classification tasks and often outperform single decision trees in accuracy and robustness.

**Overfitting/Underfitting (topic)**  
- (assertion) Overfitting in decision trees can be detected by poor generalization on the test set and very high accuracy on the training set.  
- (assertion) Measures to avoid overfitting include limiting tree depth, pruning, or using ensembles like random forests.  
- (assertion) Scaling features and careful hyper-parameter tuning also help address overfitting and underfitting issues.

**Data Preprocessing (topic)**  
- (assertion) Scaling of features can affect decision boundaries, especially in distance-based or regularized models, and is often critical before feeding data into pipelines.  
- (task) Apply a StandardScaler to the dataset before training a model to improve performance and stability of decision boundaries.

**Evaluation Metrics (topic)**  
- (assertion) Impurity measures (like Gini or Entropy) guide splits in decision trees; lower impurity means purer subsets and better class separation.

**ML Pipelines (topic)**  
- (task) Incorporate scaling, PCA, and a classifier (e.g., DecisionTreeClassifier or SVC) into a single scikit-learn pipeline.  
- (task) Experiment with changing steps and hyper-parameters within the pipeline and observe their effect on model performance.
