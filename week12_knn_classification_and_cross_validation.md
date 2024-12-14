# Week 12: KNN Classification and Cross Validation

**K-Nearest Neighbors (KNN) Classification (topic)**  
- (assertion) KNN classification assigns a label to a sample based on the majority vote of its k nearest neighbors in the feature space.  
- (assertion) Changing the number of neighbors (k) can significantly affect model accuracy. For example, `k=5` and `k=10` may yield different accuracies.  
- (assertion) KNN does not build an explicit model; predictions rely on the entire training dataset at inference time.  
- (task) Implement KNN on the Iris dataset using scikit-learn, splitting data into train/test sets with stratification to preserve class distribution.  
- (task) Experiment with different `n_neighbors` values (e.g., 1, 5, 10) and compare the resulting accuracies on the test set.  
- (task) Generate a classification report and confusion matrix for KNN predictions to assess precision, recall, and f1-score for each class.

**Evaluation and Model Quality (topic)**  
- (assertion) Evaluating a model using only residuals (differences between predictions and actual values on the training set) does not indicate how well the model generalizes to unseen data.  
- (assertion) The holdout method (train/test split) provides a better indication of generalization than residual analysis but can still have high variance depending on the split.  
- (assertion) Cross validation further improves evaluation by averaging performance over multiple folds, reducing variance caused by a single train/test split.  
- (task) Compare model evaluation methods: residuals vs. holdout vs. cross validation.  
- (task) Compute accuracy or other metrics using a single holdout set and then using k-fold cross validation to see the difference in variance and confidence in the results.

**Cross Validation (topic)**  
- (assertion) K-fold cross validation partitions the dataset into k subsets (folds), and each fold is used once as a test set, with the remaining k-1 folds as training data.  
- (assertion) K-fold CV reduces variance in the performance estimate and ensures each data point serves as a test example exactly once.  
- (assertion) Leave-one-out cross validation (k=number of samples) is an extreme form of CV, often used when data is limited.  
- (task) Perform k-fold cross validation using scikit-learn’s `cross_val_score` on multiple classifiers (LogisticRegression, KNN, Naive Bayes, SVC, LinearSVC, RandomForestClassifier, DecisionTreeRegressor) to determine which model yields the best accuracy.  
- (task) Visualize algorithm comparison using boxplots of cross validation scores to understand model stability and performance.

**Data Preprocessing (topic)**  
- (assertion) Missing values in numerical features can be replaced by the median (or another statistic) to retain more data for modeling.  
- (assertion) Feature scaling (e.g., using MinMaxScaler) ensures all features contribute more equally to the model, especially important for distance-based methods like KNN and kernel-based methods like SVC.  
- (task) Clean and preprocess a dataset (e.g., diabetes data) by imputing missing values and applying scaling.  
- (task) Retrain classifiers on scaled and imputed data, then use cross validation to verify improved model performance.

**Hyperparameter Tuning with Cross Validation (topic)**  
- (assertion) Hyperparameters are user-defined parameters (e.g., number of neighbors in KNN, C and gamma in SVC) that are not learned from the data directly.  
- (assertion) GridSearchCV automates hyperparameter tuning by testing combinations of parameters and selecting the combination that yields the best cross validation score.  
- (task) Use GridSearchCV to find optimal hyperparameters for SVC (e.g., C, kernel, gamma) on the training data, using k-fold CV to evaluate each parameter setting.  
- (task) Apply the best found model to a test dataset or a new sample to confirm improved performance.  
- (task) Experiment with different parameters and compare how the chosen hyperparameters affect the final accuracy.

**What Are “CV Folds”? (question)**
- CV stands for Cross-Validation. In k-fold cross-validation, the dataset is split into k separate subsets (folds). Then for each iteration (out of k):
  - You train the model on k-1 folds.
  - You test the model on the remaining 1 fold.
- This process repeats k times so that each fold serves as a test set exactly once. The performance metric (e.g., accuracy, recall) is averaged over the k folds to get a more robust estimate of how the model generalizes.
- “CV folds” specifically refers to those k distinct partitions of the data used during cross-validation.

**What Does It Mean to “Mesh the Coordinate Space”? (question)**
- When plotting a decision boundary in a 2D feature space (e.g., features x and y), we often:
  1. Create a grid (mesh) of points that spans the minimum to maximum values of x and y.
  2. Use the trained model to predict a label/class for each point in that grid.
  3. Color or contour-plot the grid according to the predicted classes.
- This gives a continuous “map” of the classifier’s decisions over the entire 2D plane, showing where class boundaries lie.
- Meshing basically means systematically creating a dense, evenly spaced set of points in the coordinate space so the decision boundary can be visualized as a smooth contour.
