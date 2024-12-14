# **Week 14’s Machine Learning Pipelines quiz**

---

## **1. Why do we use an sklearn `Pipeline` for data preprocessing and model training?**
A. It automatically converts all numerical features to strings for consistency  
B. It ensures a consistent and reproducible sequence of transformations and model fitting  
C. It forces the user to always use PCA before classification  
D. It randomly shuffles the steps in no specific order  

---

## **2. Which of the following best describes a **typical** pipeline structure in scikit-learn?**
A. `[("clf", LogisticRegression()), ("scaler", StandardScaler())]`  
B. `[("pca", PCA()), ("clf", LogisticRegression()), ("scaler", StandardScaler())]`  
C. `[("transformer", StandardScaler()), ("clf", LogisticRegression())]`  
D. `[("clf", DecisionTreeClassifier()), ("clf2", SVC())]`  

---

## **3. When building a `Pipeline` with multiple steps, the **last step** must always be**:**
A. A dimensionality reduction method (PCA)  
B. A model estimator (classifier or regressor)  
C. A data imputer for missing values  
D. A standard scaler  

---

## **4. If you have **both numeric and categorical** features that require different preprocessing steps, which sklearn tool helps you **process each subset** of columns differently?**
A. `GridSearchCV`  
B. `ColumnTransformer`  
C. `make_pipeline`  
D. `OneHotEncoder`  

---

## **5. When building a pipeline for classification, which line of code is correct for **fitting** and **scoring** the pipeline on test data?**
A. 
```python
pipeline.transform(X_train, y_train)
pipeline.fit(X_test, y_test)
print(pipeline.score(X_train, y_train))
```
B. 
```python
pipeline.fit(X_train, y_train)
pipeline.predict(X_test, y_test)
print(accuracy_score(X_test, y_test))
```
C. 
```python
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
```
D. 
```python
pipeline.transform(X_test, y_test)
pipeline.fit_score(X_test, y_test)
```

---

## **6. In a pipeline step such as** 
```python
("num", numeric_transformer, ["age", "fare"])
```
**within a `ColumnTransformer`, what does `numeric_transformer` represent?**
A. The final model  
B. A numeric pipeline (e.g., imputation + scaling) applied to columns `"age"` and `"fare"`  
C. A name for the final classifier  
D. A string label that references a hidden default transformer  

---

## **7. What is the key benefit of wrapping a pipeline inside **`GridSearchCV`**?**
A. It prevents overfitting by ignoring the pipeline steps  
B. It runs only one parameter setting, ignoring cross-validation  
C. It can optimize hyperparameters for both the **model** and **preprocessing steps** together  
D. It automatically selects the best random seed for reproducibility  

---

## **8. In the Titanic example pipeline:**
```python
Pipeline([
  ("preprocessor", preprocessor),
  ("classifier", LogisticRegression())
])
```
**What does `preprocessor` generally do?**
A. Performs missing-value imputation, scaling of numeric features, and one-hot encoding of categorical features  
B. Uses PCA to reduce the Titanic dataset to 2 components  
C. Trains a logistic regression model on the raw Titanic data  
D. Selects only the numeric columns and discards categorical columns  

---

## **9. Which step is **not** commonly part of a preprocessing pipeline for numeric features?**
A. SimpleImputer with a `median` or `mean` strategy  
B. StandardScaler or MinMaxScaler  
C. OneHotEncoder  
D. Outlier removal (potentially with a custom transformer)  

---

## **10. After creating a pipeline with:
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("clf", SVC())
])
```
**and fitting on the training set, how do we get predictions on the test set?**
A. `pipeline.transform(X_test)`  
B. `pipeline.fit_transform(X_test)`  
C. `pipeline.score(X_test, y_test)`  
D. `pipeline.predict(X_test)`  

---

# Answers

---

## **Answer Explanations**

### **1. Why do we use an sklearn `Pipeline`…?**  
**Your answer:** B. *It ensures a consistent and reproducible sequence of transformations and model fitting.*  
**Correctness:** **B** is correct. Pipelines are used to streamline the workflow of data preprocessing and model training in a single object, ensuring consistent transformations and avoiding manual mistakes.

---

### **2. Which describes a typical pipeline structure?**  
**Your answer:** C. `[("transformer", StandardScaler()), ("clf", LogisticRegression())]`  

**Explanation of Each Letter**:  
- Option **A**: `[("clf", LogisticRegression()), ("scaler", StandardScaler())]` would try to run the classifier **before** scaling—this is out of the usual order.  
- Option **B**: `[("pca", PCA()), ("clf", LogisticRegression()), ("scaler", StandardScaler())]` has the scaler last, which is also unusual. Typically, you’d scale **before** PCA.  
- Option **C**: `[("transformer", StandardScaler()), ("clf", LogisticRegression())]` **is** a standard approach: scaling first, then the final classifier. This is a typical pipeline order.  
- Option **D**: `[("clf", DecisionTreeClassifier()), ("clf2", SVC())]` has two classifiers in sequence, which is not typical (only the final step is usually the estimator).  

**Does the pipeline order matter?**  
**Yes**, absolutely. The pipeline order must match the logical sequence of data transformations. For example, you wouldn’t want to do PCA **before** scaling or do classification **before** imputation. Each step depends on the outputs from the previous step. If you switch them around, your model might break or yield meaningless results.  

---

### **3. The last step in a pipeline must be:**  
**Your answer:** B. *A model estimator (classifier or regressor).*  
**Correctness:** That’s correct. All preceding pipeline steps should be transformers. The final step is the **estimator** that produces predictions.

---

### **4. Processing mixed numeric & categorical features:**  
**Your answer:** B. `ColumnTransformer`  
**Correctness:** Correct. `ColumnTransformer` allows different pipelines (for numeric vs. categorical) to be merged into a single pipeline.

---

### **5. Fitting & scoring the pipeline on test data:**  
**Your answer:** B.  
But let’s check each choice carefully:

- **(A)** is incorrect: we normally call `pipeline.fit(X_train, y_train)` not `pipeline.transform` on training data first.  
- **(B)** `pipeline.fit(X_train, y_train)` followed by `pipeline.predict(X_test)` is a *typical approach*, but then you’d manually compute accuracy or some metric (like `accuracy_score(y_test, preds)`).
  
  However, the question specifically references “**which line of code is correct for fitting and scoring** on test data?” The standard approach is indeed:  
  ```python
  pipeline.fit(X_train, y_train)
  pipeline.score(X_test, y_test)
  ```
  or  
  ```python
  pipeline.predict(X_test)
  ...
  accuracy_score(y_test, preds)
  ```
- **(C)** is actually the more direct code snippet to do everything in two lines:  
  ```python
  pipeline.fit(X_train, y_train)
  print(pipeline.score(X_test, y_test))
  ```
  This is the canonical approach for scikit-learn pipeline usage.  
- If you chose **B**, you might have overlooked that `.predict(X_test, y_test)` is not valid syntax (`predict` only takes `X_test`). If the question’s code example was slightly off, the *best practice* is (C).

**So (C)** is generally the best single line to both fit and then `.score()` the pipeline. 
Your choice of B might be an “almost correct” approach *if* you interpret it as `pipeline.predict(X_test)` then manually compute accuracy. But the given snippet `predict(X_test, y_test)` is not standard usage. 

---

### **6. In a `ColumnTransformer` step like `("num", numeric_transformer, ["age", "fare"])`:**  
**Your answer:** B. *A numeric pipeline (e.g., imputation + scaling) applied to columns “age” and “fare”.*  
**Correctness:** Precisely. That pipeline is the transformation pipeline for numeric columns.

---

### **7. Benefit of wrapping a pipeline in `GridSearchCV`**  
**Your answer:** C. *It can optimize hyperparameters for both the model and preprocessing steps.*  
**Correctness:** Exactly. One of the biggest strengths of scikit-learn pipelines is that your grid search can tune parameters in all pipeline steps.

---

### **8. In the Titanic example, what does “preprocessor” do?**  
**Your answer:** A. *Performs missing-value imputation, scaling of numeric features, and one-hot encoding…*  
**Correctness:** That is correct.

---

### **9. Which step is **not** part of numeric preprocessing?**  
**Your answer:** C. *OneHotEncoder.*  
**Correctness:** Right, one-hot encoding is for **categorical** variables, not numeric.

---

### **10. After fitting a pipeline with `scaler`, `pca`, `clf`, how do we get predictions?**  
**Your answer:** D. *`pipeline.predict(X_test)`.*  
**Correctness:** Exactly. `pipeline.predict(X_test)` runs all transformations on `X_test` then calls the model’s `.predict()`.

---

## **What Does “clf” Stand For?**

- **“clf”** is simply **short for “classifier.”** It’s a common naming convention in scikit-learn examples.  
- When you see code like:  
  ```python
  clf = Pipeline([
      ("scaler", StandardScaler()),
      ("pca", PCA(n_components=2)),
      ("clf", DecisionTreeClassifier())
  ])
  ```
  The variable name `clf` on the **left** side (`clf = Pipeline(...)`) is just a **Python variable** referencing the entire pipeline object.  
- Inside the **Pipeline steps** you might also see `('clf', DecisionTreeClassifier())`. That **‘clf’** is a step **label** within the pipeline. The pipeline steps require a string name for each transformation/estimator, and `'clf'` is a common label for “classifier step.”  

### **Why name the pipeline “clf” sometimes and other times “pipeline”?**
- In Python, you can name a variable anything. If the final step is a classifier, many people like using `clf` as a short, convenient name.  
- If the final step were a regressor, we might name the pipeline `reg`. If it’s more general, some people just call it `pipeline`.  
- **Naming** is purely **convention** and has no functional difference. The pipeline object is the same; the name “clf” doesn’t do anything special programmatically. It’s a matter of code readability.

**In short**: 
- `clf` as a **variable name** (e.g., `clf = Pipeline([...])`) is just a Python reference for that pipeline object.  
- `('clf', DecisionTreeClassifier())` is the pipeline **step name** (the label for the final step). You could call it `('final_model', DecisionTreeClassifier())` or `('estimator', DecisionTreeClassifier())`; it’s entirely up to you.

---

### **Does Pipeline Order Matter?**  
**Absolutely.** The transformations happen in the exact sequence you specify. For example:  
1. Impute missing values.  
2. Scale numeric features.  
3. Perform PCA.  
4. Fit a final model (SVC, LogisticRegression, DecisionTree, etc.).  

Reversing them can break the logic or yield meaningless results. So **yes**, pipeline order is crucial.

---

## **Takeaways**:
- You answered most questions in line with best practices.  
- The usage of **`clf`** is a **convention** meaning “classifier.” It’s not mandatory but widely used in scikit-learn documentation and examples.  
- The name `'clf'` in pipeline steps vs. `clf` as a Python variable is essentially the same label, but used in two different contexts: step naming vs. variable naming. Both are purely for readability and have no special effect on how the pipeline runs.
