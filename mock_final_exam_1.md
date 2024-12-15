# **Multiple Choice Exam**

## **Section 1: Concepts & Theory**

### **Q1. (Naive Bayes Concept)**
(5 points)

Naive Bayes is considered “naive” because:
A. It assumes features are fully correlated, making it extremely accurate on any dataset.  
B. It uses a kernel trick to transform data into higher dimensions.  
C. It assumes each feature is conditionally independent given the class label.  
D. It always requires numeric features scaled to [0,1].

---

### **Q2. (Graphing & Visualization)**
(4 points)

Which of the following **matplotlib** plot(s) would be most helpful to visually **inspect outliers** in a regression context?
A. A scatter plot of `x_data` vs. `y_data` plus a regression line  
B. A bar chart of the regression coefficients  
C. A box plot or residual plot highlighting large errors  
D. A stacked bar chart of predicted vs. actual classes  

---

### **Q3. (K-Means & Cluster Count)**
(3 points)

How do we typically decide the **number of clusters (k)** in k-Means using the **Elbow Method**?
A. We pick the k that gives the **largest** within-cluster sum of squares (inertia).  
B. We pick the k corresponding to the first time the plot forms loops.  
C. We choose a random k.  
D. We look for a “bend” or “elbow” in the inertia plot, where further increases in k yield diminishing returns.

---

### **Q4. (k-NN & Cross-Validation)**
(3 points)

You want to use 10-fold cross-validation to tune the number of neighbors (k) in k-NN. Which step is **most** essential?
A. Fit k-NN on the entire dataset without cross-validation.  
B. Use **GridSearchCV** or manual loops over k in [1,3,5,...], evaluating accuracy via cross-validation for each k.  
C. Encode the target labels with OneHotEncoder.  
D. Plot the decision boundary for each k without measuring accuracy.

---

### **Q5. (Pipelines & Data Leakage)**
(5 points)

**Why** do scikit-learn pipelines help prevent data leakage?
A. They randomly shuffle the dataset multiple times.  
B. Pipelines fit preprocessing only on the training data, then apply those **same** transformations to the test set.  
C. They always ignore new or unknown categories in categorical data.  
D. They store data in a CSV file that is never read by the final estimator.

---

### **Q6. (PCA Variance)**
(5 points)

Which attribute in an sklearn PCA object tells you how much variance each principal component captures?
A. `pca.components_`  
B. `pca.singular_values_`  
C. `pca.explained_variance_ratio_`  
D. `pca.inverse_transform_`

---

## **Section 2: Coding & Analysis**

### **Q7. (Naive Bayes Classification)**  
(12 points)

**Select the correct snippet** to train a categorical Naive Bayes (e.g., `CategoricalNB` or `MultinomialNB`) model on a dataset with categorical features. The dataset is split into `X_train, X_test, y_train, y_test`. We need to **encode** the categorical features, then fit the model.  

A. 
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit_transform(X_train, y_train)
clf = MultinomialNB()
clf.fit(X_train, y_train)
```

B. 
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
X_train_encoded = enc.fit_transform(X_train)
X_test_encoded = enc.transform(X_test)
clf = MultinomialNB()
clf.fit(X_train_encoded, y_train)
```

C. 
```python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, X_test)
```

D. 
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()
X_train_encoded = lab.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train_encoded, y_train)
```

---

### **Q8. (Pipelines & Graphing)**  
(10 points)

We want a pipeline that:  
1) Imputes missing numeric values with median,  
2) Scales the numeric features,  
3) Uses **LogisticRegression(C=100)**,  
4) Finally we **plot** the decision boundary (assuming 2 numeric features).  

Which pipeline **code snippet** is correct to accomplish steps (1)–(3)? (We’ll handle plotting later.)

A. 
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=100))
])

pipeline.fit(X_train, y_train)
```

B. 
```python
pipeline = Pipeline([
    ("clf", LogisticRegression(C=100)),
    ("scaler", StandardScaler()),
    ("imputer", SimpleImputer(strategy="median"))
])
pipeline.fit_transform(X_train, y_train)
```

C. 
```python
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
    ("clf", LogisticRegression(C=100))
])
pipeline.fit(X_train, y_train)
```

D. 
```python
from sklearn.impute import SimpleImputer
pipeline = SimpleImputer(strategy="median")
pipeline.fit(X_train)
pipeline.transform(X_test)
```

---

### **Q9. (K-means with K=3 & Plot)**  
(8 points)

You have a 2D dataset in `someClusterData.csv`. Which snippet properly **fits** K-Means with 3 clusters and **plots** each data point colored by cluster? (No code for centroids needed in the snippet, just the cluster color plot.)

A. 
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("someClusterData.csv")
kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
labels = kmeans.fit(df)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=labels)
```

B.
```python
from sklearn.cluster import KMeans

df = pd.read_csv("someClusterData.csv")
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(df)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=labels)
plt.show()
```

C.
```python
df = pd.read_csv("someClusterData.csv")
kmeans = KMeans(n_clusters=3).transform(df)
labels = kmeans.predict(df)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=labels)
```

D.
```python
df = pd.read_csv("someClusterData.csv")
kmeans = KMeans(n_clusters=3)
preds = kmeans.fit_transform(df)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=preds)
```

---

## **Section 3: Interpretation of Results**

### **Q10. (Comparing Classifiers & Metrics)**
(10 points)

You have classification reports for a Decision Tree, an SVM, and a Random Forest on an **imbalanced** dataset (70% class 0, 30% class 1). Suppose your **goal** is to minimize **false negatives** for class 1. Which metric do you specifically want to maximize for class 1, and **why**?

A. **Accuracy**, because class imbalance does not matter if the overall accuracy is high.  
B. **Precision** for class 1, because that ensures fewer false positives.  
C. **Recall** for class 1, because you want to reduce missed actual positives (false negatives).  
D. **F1-score** for class 1, because it only measures false positives.

---

### **Q11. (PCA Variance Ratio)**
(5 points)

Your PCA yields an **explained_variance_ratio_** array = `[0.50, 0.30, 0.10, 0.10]`. You want to keep **at least 80% variance**. How many principal components do you need?
A. 1  
B. 2  
C. 3  
D. 4  

---

### **Q12. (Bias-Variance Trade-off)**
(10 points)

A random forest achieves 99% training accuracy but only 70% test accuracy. This discrepancy likely indicates:
A. The model is **underfitting**; it should be trained longer.  
B. The model has a high bias but low variance.  
C. The model is overfitting, memorizing training data rather than generalizing.  
D. The dataset must have zero variance in features.

---

## **Section 4: Extended Scenario / Pipelines**

### **Q13. (Naive Bayes + ColumnTransformer Pipeline)**  
(10 points)

You’re given a dataset `finalExamData.csv` with 10 numeric features and 5 categorical features for a **binary** classification (0/1). You want to:
- Impute numeric columns (median), scale numeric columns.  
- OneHotEncode categorical columns (`handle_unknown='ignore'`).  
- Then use PCA on numeric columns only, retaining 90% variance.  
- Finally, train a **Naive Bayes** classifier on the transformed data.

Which pipeline skeleton best captures that approach, **assuming** you define numeric_features and categorical_features lists?

A.
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90))    # PCA for numeric only
    ]), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("nb", GaussianNB())
])
```

B.
```python
clf = Pipeline([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("pca", PCA(n_components=0.90)),
    ("classifier", GaussianNB())
])
```

C.
```python
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("pca", PCA(n_components=0.90))   # PCA first
    ]), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])
clf = Pipeline([
    ("classifier", GaussianNB())
])
```

D.
```python
from sklearn.naive_bayes import GaussianNB

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

clf = GaussianNB()
clf.fit(X_train, y_train)
```

---

### **Q14. (Threshold Tuning)**  
(5 points)

If the Naive Bayes model’s recall for class 1 is too low, which action might help **increase recall** (though it might affect precision)?

A. Increase the PCA components to 99%.  
B. Decrease the numeric imputer strategy from median to mean.  
C. Lower the decision threshold for predicting class 1, accepting more positives.  
D. Convert the categorical features to numeric by random integer coding.

---

## **END OF EXAM**

**Instructions**: Provide your chosen answers (e.g. Q1: C, Q2: A, …). Then, for at least three of the questions, write **one or two sentences** explaining why you picked that answer, focusing on your reasoning or code analysis.
