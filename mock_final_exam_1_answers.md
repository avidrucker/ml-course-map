Below is the **Answer Key** for the multiple-choice exam, along with **1–2 sentence explanations** per question.

---

## **Section 1: Concepts & Theory**

### **Q1. (Naive Bayes Concept)**
**Correct Answer: C**  
**Explanation:** Naive Bayes is called “naive” because it assumes each feature is conditionally independent given the class label, which often simplifies computations but doesn’t always hold in reality.

---

### **Q2. (Graphing & Visualization)**
**Correct Answer: C**  
**Explanation:** A **box plot** or **residual plot** highlights points that deviate significantly from typical values or from the model’s predictions, making outliers easier to spot in a regression context.

---

### **Q3. (K-Means & Cluster Count)**
**Correct Answer: D**  
**Explanation:** With the **Elbow Method**, you plot the inertia (within-cluster sum of squares) against `k` and look for a “bend” or “elbow,” where increasing `k` further yields diminishing returns in variance reduction.

---

### **Q4. (k-NN & Cross-Validation)**
**Correct Answer: B**  
**Explanation:** Tuning `k` in k-NN typically involves cross-validation over a range of `k` values (e.g., 1 to 20). Using **GridSearchCV** or a manual loop with cross-validation systematically evaluates each `k` value for model performance.

---

### **Q5. (Pipelines & Data Leakage)**
**Correct Answer: B**  
**Explanation:** Pipelines in scikit-learn ensure transformations (e.g., scaling or encoding) are **fit only on the training data** and then applied to the test data, preventing data leakage and preserving a fair evaluation.

---

### **Q6. (PCA Variance)**
**Correct Answer: C**  
**Explanation:** `pca.explained_variance_ratio_` is an array that indicates how much variance each principal component captures, helping us decide how many components to retain.

---

## **Section 2: Coding & Analysis**

### **Q7. (Naive Bayes Classification)**  
**Correct Answer: B**  
**Explanation:** We must **fit and transform** `X_train` with OneHotEncoder, **transform** `X_test` with the same encoder, and then train `MultinomialNB` on the encoded training data. Option B does exactly that.

---

### **Q8. (Pipelines & Graphing)**
**Correct Answer: A**  
**Explanation:** The pipeline properly orders steps: imputer → scaler → logistic regression. Options B and C either have the wrong order or omit key steps, and D doesn’t chain the transformations into one pipeline.

---

### **Q9. (K-means with K=3 & Plot)**
**Correct Answer: B**  
**Explanation:** Using **`kmeans.fit_predict(df)`** returns cluster labels for each row, which we can then pass to `plt.scatter(..., c=labels)`. This is the standard approach for clustering and plotting the results.

---

## **Section 3: Interpretation of Results**

### **Q10. (Comparing Classifiers & Metrics)**
**Correct Answer: C**  
**Explanation:** **Recall for class 1** measures the fraction of actual positives correctly identified. Minimizing false negatives means maximizing recall for class 1.

---

### **Q11. (PCA Variance Ratio)**
**Correct Answer: B**  
**Explanation:** With `[0.50, 0.30, 0.10, 0.10]`, the first two components capture `0.50 + 0.30 = 0.80` (80%). So **2** components suffice to reach the 80% variance goal.

---

### **Q12. (Bias-Variance Trade-off)**
**Correct Answer: C**  
**Explanation:** A big gap between high training accuracy (99%) and lower test accuracy (70%) usually indicates **overfitting**: the model memorizes the training set rather than generalizing well.

---

## **Section 4: Extended Scenario / Pipelines**

### **Q13. (Naive Bayes + ColumnTransformer Pipeline)**
**Correct Answer: A**  
**Explanation:** Option A sets up the **ColumnTransformer** so numeric columns are imputed, scaled, and PCA-transformed, while categorical columns are one-hot encoded, all before training a **Naive Bayes** classifier. It matches the stated requirement precisely.

---

### **Q14. (Threshold Tuning)**
**Correct Answer: C**  
**Explanation:** Lowering the decision threshold for predicting class 1 typically **increases recall** at the expense of precision, which is often acceptable if we want to reduce false negatives for class 1.

---

**End of Answer Key**
