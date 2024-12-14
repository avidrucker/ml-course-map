Below is a **study guide** for **Week 14: Machine Learning Pipelines**. This document will help you remember the big ideas, core workflows, and best practices for building and applying ML pipelines in scikit-learn.

---

# **Week 14 Study Guide: Machine Learning Pipelines**

## **1. Introduction to Pipelines**

### **Why Use Pipelines?**
- **Streamlined Workflow**: Combine pre-processing steps (e.g., imputation, scaling, encoding) and modeling into a single object.  
- **Reduced Errors**: Enforces a consistent sequence of transformations for both **training** and **testing**.  
- **Hyperparameter Tuning**: Integration with **GridSearchCV** or **RandomizedSearchCV** for the entire pipeline (i.e., you can tune preprocessing steps and model parameters together).
- **Reusable & Modular**: Re-run the same pipeline on new data without repeating code or risking mistakes in manual transformations.

### **Key Concepts**
- `Pipeline(steps=[('step_name', transformer), ('step_name', model)])`
- Each "step" is either a preprocessing transformer (e.g., `StandardScaler`) or a final estimator (e.g., `DecisionTreeClassifier`, `SVC`, etc.).
- The **last** step is always the estimator (classifier or regressor). All steps before it must be **transformers** that implement `fit` and `transform`.

---

## **2. Common Preprocessing Tasks**

1. **Imputing Missing Values**  
   - `from sklearn.impute import SimpleImputer`  
   - Strategies include `mean`, `median`, `most_frequent`.  
   - Example usage:  
     ```python
     imputer = SimpleImputer(strategy='median')
     ```
2. **Outlier Detection / Removal**  
   - Not directly built-in as a standard pipeline step but can be done with custom transformers or third-party solutions.  
   - E.g., creating a function/class that filters out outliers and then use `Pipeline` for consistent application.
3. **Scaling / Normalization**  
   - `StandardScaler()` (z-score standardization)  
   - `MinMaxScaler()` (scales each feature to [0,1])  
4. **Encoding Categorical Features**  
   - **OrdinalEncoder** for ordinal categories.  
   - **OneHotEncoder** for nominal categories.  
   - Handle unknown categories with `handle_unknown="ignore"` if you expect new categories at inference time.
5. **Dimensionality Reduction**  
   - **PCA** to reduce feature space.  
   - Tends to be placed after scaling (so the pipeline order is typically `[('scaler', StandardScaler()), ('pca', PCA(...))]`).

---

## **3. Building a Pipeline in Scikit-Learn**

**Syntax Example**:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', DecisionTreeClassifier())
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
```

**Key Takeaways**:
- **Order matters**: Put transformations in the sequence that makes sense (e.g., imputation first, then scaling, then PCA, then model).  
- `pipeline.predict()` automatically applies **all** transformation steps (fit on training data) to new data, then calls the final estimator’s `.predict`.

---

## **4. ColumnTransformer for Mixed Feature Types**

### **Motivation**
- Real-world datasets often contain **numeric** and **categorical** features.  
- Different preprocessing pipelines might be needed for each subset of columns.

### **ColumnTransformer**:
```python
from sklearn.compose import ColumnTransformer

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
```
- **Explanation**: This means “apply `numeric_transformer` to columns `age` and `fare`” and “apply `categorical_transformer` to columns `embarked`, `sex`, and `pclass`.”

### **Combine with Model**:
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
```
- **Advantage**: A single pipeline that handles all numeric/categorical preprocessing plus final classification.

---

## **5. Example Workflows**

### **Example 1: Iris Dataset with PCA + Classifier**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=4
)

pipeline = Pipeline([
    ('scler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', DecisionTreeClassifier())
])
pipeline.fit(X_train, y_train)
print("Test Accuracy:", pipeline.score(X_test, y_test))
```
- Steps:
  1. **Scale** features,  
  2. **Reduce** to 2 principal components,  
  3. **Train** a `DecisionTreeClassifier`.

### **Example 2: Titanic Dataset (Mix of Numeric & Categorical)**
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))  # e.g., 0.799
```
- Steps:
  1. **Impute** missing numeric data with median,  
  2. **Scale** numeric features,  
  3. **One-Hot Encode** categorical columns,  
  4. **Logistic Regression** model.

### **Example 3: Wine Data with ColumnTransformer**  
- Similar approach: numeric columns use `MinMaxScaler`, categorical columns use `OneHotEncoder`.  
- Final estimator is either `LogisticRegression` or `SVC`.

---

## **6. Pipeline Tips & Best Practices**

1. **Consistent Order**: Preprocessing steps must come before the estimator. For instance, if you do PCA, make sure it’s after scaling.  
2. **Custom Transformers**: You can create your own Python class to implement a `fit` and `transform` method (useful for outlier removal or custom feature engineering). Then place it in the pipeline.  
3. **Hyperparameter Tuning**:  
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {
       'clf__C': [0.1, 1, 10, 100],
       'clf__kernel': ['rbf', 'linear']
   }
   search = GridSearchCV(pipeline, param_grid, cv=5)
   search.fit(X_train, y_train)
   ```
   - Notice how parameter names are specified: `'stepname__parametername'`.  
   - This seamlessly tunes both pipeline steps and final model hyperparameters.

4. **Column Names**: Make sure to pass the correct **list of column names** to the correct transformers in `ColumnTransformer`.  
5. **Handling Unknown Categories**: `OneHotEncoder(handle_unknown="ignore")` helps avoid errors when new categories appear in test data.  
6. **Performance**: Using a pipeline is often more efficient and repeatable than manual steps for your ML workflow.

---

## **7. Putting It All Together**

- **Define** your numeric transformer pipeline (imputation, scaling, etc.).  
- **Define** your categorical transformer pipeline (one-hot or ordinal encoding).  
- **Use** `ColumnTransformer` if you have both numeric and categorical data.  
- **Wrap** it all in a final **Pipeline** with an estimator (e.g., LogisticRegression, SVC, DecisionTree).  
- **Train** (fit) on `X_train`, `y_train`.  
- **Evaluate** (`.score(X_test, y_test)` or other metrics like `.predict()`, `accuracy_score`).  
- Consider **hyperparameter tuning** or cross-validation (e.g., `GridSearchCV`) to refine your pipeline.

---

# **Key Takeaways**

1. **Pipelines** allow consistent, clean transformations from raw data to final predictions.  
2. **ColumnTransformer** splits numeric and categorical features into different transformation routes.  
3. Preprocessing steps such as **imputation, scaling, encoding** are crucial in real-world data.  
4. Always **fit** transformations on **training data only** (the pipeline does this automatically under the hood), then **transform** the test data.  
5. Pipelines integrate seamlessly with scikit-learn’s **model selection** and **hyperparameter tuning**.

---

**Use these pipelines to ensure your entire machine learning process is reproducible, consistent, and less error-prone.** After mastering pipelines, you’ll have a powerful template for building more advanced workflows that scale to complex real-world projects.
