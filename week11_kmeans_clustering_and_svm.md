# Week 11: SVM & KMeans Clustering

**Clustering (topic)**  
- (assertion) K-means clustering groups data into K clusters to minimize within-cluster sum of squares (inertia).  
- (assertion) Increasing the number of centroids typically decreases inertia since points are closer to their assigned cluster center.  
- (assertion) The elbow method involves plotting inertia against K and selecting the “elbow” point that balances a low inertia with a small number of clusters.

- (task) Implement K-means clustering on a given dataset and vary the number of clusters to observe changes in inertia and identify an appropriate K using the elbow method.

**Support Vector Classifier (SVC) (topic)** (as referenced in week 11 exercises)  
- (assertion) Hyper-parameters like C and kernel parameters must be selected carefully to ensure good generalization of the SVC model.  
- (task) Use GridSearchCV or similar hyper-parameter tuning methods to select the best C or kernel for SVC.

---

# Study Prep

Below is a **machine-learning-teacher–style** outline of **Week 11** topics, tasks, and key points (factoids) gleaned from the reference material. Use this guide to know which skills and concepts you’re most likely to be tested on.

---

## 1. **Unsupervised Learning & k-Means Clustering**

### Core Concepts
1. **Unsupervised vs. Supervised**  
   - **Unsupervised:** Labels not known in advance. Goal is to find structure (clusters) in data.  
   - **Supervised:** Labels are provided, and the task is to learn a model that predicts those labels.

2. **What is Clustering?**  
   - Grouping similar data points so that points in the same cluster are more similar to each other than to points in other clusters.  
   - Applications: Market segmentation, customer grouping, social network analysis, astronomy data analysis, etc.

### k-Means Algorithm Specifics
1. **Algorithm Steps (k-Means)**  
   1. Choose **k** initial centroids (cluster centers), often randomly.  
   2. **Cluster assignment:** Assign each data point to the closest centroid (using Euclidean distance or sum of squared distances).  
   3. **Move centroids:** Recompute each centroid as the mean of all points assigned to that cluster.  
   4. Repeat until centroids stop moving (convergence).

2. **Computing Distances**  
   - Typically uses Euclidean distance (sum of squared differences).  
   - Minimizes the within-cluster sum of squares.

3. **Sensitivity to Initialization**  
   - Different initial centroids can lead to different final clusters.  
   - `init='k-means++'` is a common heuristic to improve initial centroid selection.

4. **Choosing k**  
   - Not always obvious.  
   - **Elbow method:** Plot the within-cluster sum of squares (inertia) vs. number of clusters; look for an “elbow” where adding more clusters yields diminishing returns.

5. **Iteration Details**  
   - Each iteration has two main steps: “cluster assignment” and “centroid update.”  
   - Convergence is reached when cluster assignments don’t change (or changes are below a threshold).

### Tasks & Hands-On Skills
- **Using `sklearn.cluster.KMeans`** to fit a model:
  - `KMeans(n_clusters=2, random_state=0).fit(X)`
  - Extracting `kmeans.cluster_centers_` and `kmeans.labels_`
  - Visualizing cluster assignments with matplotlib or seaborn
- **Performing multiple iterations** to watch cluster centers move.  
- **Experimenting with different values of k** (e.g., 2, 3, or more) and seeing how clusters change.  
- **Elbow method** to find an optimal number of clusters.

---

## 2. **Support Vector Machine (SVM) Exercises**

*(Although these SVM exercises might be from the same or additional references for Week 11, they are also worth noting.)*

### Core Concepts
1. **Linear Separability**  
   - Checking if data can be separated by a straight line (2D) or a hyperplane (higher dimensions).
2. **C Parameter (Regularization)**  
   - **High C** ~ less regularization → tries to perfectly classify training data (hard margin).  
   - **Low C** ~ more regularization → allows more misclassifications (soft margin).
3. **Kernels**  
   - `linear`, `rbf`, `poly`, etc.  
   - Kernel choice affects decision boundary shape.
4. **Drawing Decision Boundaries**  
   - Important for visualizing how the SVM separates the classes.

### Tasks & Hands-On Skills
- **Using `trainData.csv` or a similar dataset** to develop a Support Vector Classifier.  
- **Experimenting with multiple values of C**: 1, 100, 10,000 (1E4), 1E6, etc., and comparing how the decision boundary changes.  
- **Evaluating training accuracy** and **applying the same models to test data** to see how accurate the classification is.  
- Possibly using different kernels (`poly`, `rbf`) and checking their impact on the boundaries.

---

## 3. **Outlier Elimination Exercise (Regression Context)**

### Core Concepts
1. **Outliers**  
   - Extreme data points that might distort regression fits or other statistical models.
2. **Residual Error**  
   - Difference between actual label and predicted label.  
   - Outliers often have large absolute residuals.
3. **Iterative Outlier Removal**  
   - Fit a regression → compute residuals → remove top X% that are farthest from the regression line.  
   - Fit regression again on the “cleaned” dataset.

### Tasks & Hands-On Skills
- **Loading data** (e.g., `outliersData1.csv`)  
- **Fitting a linear regression** to all data; compute R².  
- **Identifying 10% of data with highest residual** (errors) and removing them.  
- **Refitting** the regression on the cleaned dataset.  
- Checking the difference in R² or other performance metrics after outlier removal.  
- **Writing a helper function** (`outlierCleaner`) that:
  - Accepts X, y, and y_pred along with the chosen threshold.  
  - Returns X_cleaned, y_cleaned after removing outliers.

---

## Highest Priorities & Likely Test/Quiz Focus

1. **k-Means Clustering**  
   - How the algorithm works step-by-step: initialization, assignment, centroid update.  
   - How to interpret cluster labels, cluster centers, and inertia.  
   - Practical usage in scikit-learn (constructing a `KMeans` model, accessing `.cluster_centers_`, `.labels_`, `.inertia_`).
   - The **Elbow method** for choosing k.

2. **Impact of Initialization**  
   - Understand that different initial centroids can lead to different clusters in k-means.  
   - How random state or specifying an initial array of centroids affects the result.

3. **Support Vector Machine (SVM)**  
   - Idea of linear vs. non-linear separability.  
   - Role of the **C parameter** and how it affects the margin.  
   - Possibly the effect of kernels (linear, polynomial, RBF).

4. **Outlier Handling (Regression)**  
   - Why outliers matter.  
   - The process of iterative removal (fit → residuals → remove → refit).  
   - Understand the difference in R² before and after outlier removal.

5. **General ML Best Practices**  
   - Data splits (train/test).  
   - Fitting models, interpreting model results (accuracy, R², etc.).  
   - Visualizing data and cluster boundaries / decision boundaries.

---

## Additional Factoids & Reminders

- **k-Means** is not guaranteed to find the global optimal solution; it can converge to local minima.  
- In scikit-learn, for better results, you can use `init='k-means++'` to improve centroid initialization.  
- For SVM, as `C` changes, you’ll see different complexities of decision boundaries. Hard margin SVM may overfit if data is not linearly separable.  
- Outlier removal is heuristic-dependent (e.g., removing 10% is arbitrary) and can sometimes remove legitimate data points. Always double-check the domain context.

---

### Final Takeaways
Week 11’s materials revolve around:
- **Unsupervised** learning with **k-means** (core algorithms, scikit-learn usage, elbow method).
- **Support Vector Classifiers** with different hyperparameters.
- **Outlier elimination** steps in a regression problem.

**Practice** writing code that:
1. Loads data (e.g., CSVs).
2. Fits the models (k-means, SVM, or linear regression).
3. Interprets the results (centers, labels, boundaries, R², error metrics).
4. Iterates or experiments with parameters (e.g., number of clusters, outlier fraction, C in SVM).
5. Visualizes results (scatter plots, decision boundaries, elbow curves).

Expect exam/quiz questions on:
- Detailed k-means steps and how to apply them.  
- SVM boundary changes with different C values.  
- How and why to remove outliers and the effect on regression fits.
