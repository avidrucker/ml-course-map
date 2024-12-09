# Machine Learning Course Map

**Machine Learning Fundamentals (topic)**  
- Overview (topic)  
  - (assertion) Students should be able to use scikit-learn to create, train, and test various machine learning models.  
  - (assertion) Students should be able to produce plots and visualizations of data and model performance using matplotlib.  
  - (assertion) Students should understand the difference between supervised, unsupervised, and semi-supervised learning techniques.

- Naive Bayes Classification (topic)  
  - (assertion) Naive Bayes classifiers assume that features are conditionally independent given the class label.  
  - (assertion) Naive Bayes can handle categorical data by encoding features into numeric values and applying suitable probability models.  
  - (assertion) Naive Bayes is particularly effective for text classification problems due to its simple probabilistic approach.

- Linear Regression (topic)  
  - (assertion) Linear regression fits a linear model to minimize the sum of squared errors between predictions and actual values.  
  - (assertion) Common evaluation metrics for linear regression include Mean Squared Error (MSE) and R-squared (coefficient of determination).  
  - (assertion) Proper feature scaling and removal of outliers can improve the stability and accuracy of a linear regression model.

- Logistic Regression (topic)  
  - (assertion) Logistic regression models the probability of a binary outcome as a logistic function of input features.  
  - (assertion) Unlike linear regression, logistic regression outputs probabilities and is suitable for classification tasks.  
  - (assertion) The decision boundary in logistic regression is linear in the feature space.

- Gradient Descent (topic)  
  - (assertion) Gradient descent is an iterative optimization algorithm used to find a local minimum of a differentiable function.  
  - (assertion) In machine learning, gradient descent is commonly used to adjust model parameters to minimize a loss function.  
  - (assertion) Learning rate selection is crucial; too large a value can cause divergence, while too small slows convergence.

- Multi-variable Regression (topic)  
  - (assertion) Multi-variable (multivariate) regression extends linear regression to multiple input features simultaneously.  
  - (assertion) Feature scaling and normalization often improve the conditioning of the problem, leading to more stable parameter estimates.  
  - (assertion) Multicollinearity between features can reduce interpretability and stability of regression coefficients.

- Decision Trees (topic)  
  - (assertion) Decision trees recursively split the dataset based on feature values to create a hierarchy of decision rules.  
  - (assertion) Each internal node in a decision tree corresponds to a test on a feature, and each leaf node corresponds to a class or value prediction.  
  - (assertion) Decision trees can overfit if not pruned or constrained by parameters like max depth or minimum samples per split.
  - (task) Given a dataset, set max_depth=3 and fit a DecisionTreeClassifier, then visualize decision boundaries to understand how the model splits features.

- Random Forest (topic)  
  - (assertion) A random forest is an ensemble of decision trees, each trained on a random subset of features and samples.  
  - (assertion) Random forests generally improve over single decision trees by reducing variance and thus overfitting.  
  - (assertion) The final prediction of a random forest is typically made by averaging or majority voting over all trees.
  - (task) Fit a RandomForestClassifier with default parameters and report classification metrics.  
  - (assertion) Random forests can produce stable results with minimal tuning and handle complex decision boundaries.

- Overfitting/Underfitting (topic)  
  - (assertion) Overfitting occurs when a model learns patterns specific to the training data, failing to generalize to unseen data.  
  - (assertion) Underfitting occurs when a model is too simple to capture underlying trends in the data.  
  - (assertion) The bias-variance trade-off describes the balance between model complexity (variance) and generalization (bias).

- Evaluation Metrics (topic)  
  - (assertion) Accuracy measures the proportion of correct predictions but can be misleading if classes are imbalanced.  
  - (assertion) Precision and recall help evaluate the model’s performance on minority classes more effectively than accuracy alone.  
  - (assertion) The AUC (Area Under the ROC Curve) metric summarizes model performance across various threshold settings.
  - (task) Compute classification_report (precision, recall, f1-score, accuracy) for multiple models and interpret differences.  
  - (assertion) Classifier choice may depend on cost-sensitive considerations: finding all positives (high recall) vs. avoiding false positives (high precision).

- Data Preprocessing (topic)  
  - (assertion) Scaling features ensures that no single feature dominates the model due to differences in magnitude and units.  
  - (assertion) Imputing missing data allows models to use as many samples as possible without discarding incomplete rows.  
  - (assertion) Handling outliers by transformation or removal can improve model stability and reduce the impact of anomalous points.
  - (assertion) Stratification in train-test splitting ensures that class proportions are maintained, especially critical when classes are imbalanced.  
  - (task) Perform a stratified train-test split using `StratifiedKFold` or `train_test_split` with `stratify` parameter.

- Support Vector Classifier (SVC) (topic)  
  - (assertion) An SVC tries to find a hyperplane that maximizes the margin between classes.  
  - (assertion) Kernel functions allow SVC to model complex, non-linear decision boundaries.  
  - (assertion) Regularization parameters (like C) control the trade-off between a smooth decision boundary and correctly classifying training samples.
  - (task) Fit an SVC with rbf kernel and C=1000, visualize decision boundaries, and compare metrics (precision, recall, F1).

- Clustering (topic)  
  - (assertion) K-means clustering partitions data into K clusters, minimizing the within-cluster sum of squares.  
  - (assertion) The choice of K in K-means can be guided by methods like the elbow method, silhouette score, or domain knowledge.  
  - (assertion) Clustering is unsupervised, meaning it groups data based only on feature similarity, without class labels.

- K-nearest Neighbors (KNN) (topic)  
  - (assertion) KNN classifies a sample by looking at the classes of its K closest neighbors in feature space.  
  - (assertion) KNN is a lazy learner and can become computationally expensive as the dataset grows large.  
  - (assertion) Choosing K and the distance metric significantly affects KNN’s performance and stability.

- Model Selection & Hyper-parameter Tuning (topic)  
  - (assertion) Cross-validation helps estimate how well a model generalizes by training and evaluating on multiple folds of the data.  
  - (assertion) GridSearch automates the search for optimal hyper-parameters by exhaustively trying combinations from a predefined set.  
  - (assertion) Proper hyper-parameter tuning can drastically improve model performance while preventing overfitting.

- Dimensionality Reduction (topic)  
  - (assertion) Principal Component Analysis (PCA) projects data onto directions of maximum variance to reduce the dimensionality.  
  - (assertion) Dimensionality reduction can speed up training and reduce overfitting by removing noise and redundant features.  
  - (assertion) Retaining only a few principal components can preserve most variance while simplifying the model.

- ML Pipelines (topic)  
  - (assertion) A pipeline chains together preprocessing steps and a final estimator, ensuring a repeatable and clean workflow.  
  - (assertion) By incorporating transformations like scaling, PCA, or feature selection into a pipeline, data leakage can be prevented.  
  - (assertion) Pipelines make it easier to tune parameters for multiple steps simultaneously, improving the model selection process.
