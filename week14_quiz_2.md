---

# **Quiz: Encoding Categorical Data**

### **1. Which **encoding** is typically used for **non-ordinal categorical** **features**?**  
A. LabelEncoder  
B. OneHotEncoder  
C. StandardScaler  
D. PCA  

---

### **2. What is the main **difference** between `LabelEncoder` and `OneHotEncoder`?**  
A. They are exactly the same but from different modules  
B. LabelEncoder is only for numeric features; OneHotEncoder is only for images  
C. LabelEncoder outputs a single column with integer codes, whereas OneHotEncoder outputs multiple binary columns  
D. OneHotEncoder works only with ordinal data, while LabelEncoder works with nominal data  

---

### **3. In the breast cancer dataset code, we had lines**:
```python
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(X_train)
X_train_encoded = onehot_encoder.transform(X_train)
X_test_encoded = onehot_encoder.transform(X_test)
```
**Why do we call `.transform(X_test)` separately (rather than `.fit_transform(X_test)`)?**  
A. To keep the same categories learned from `X_train` so the test data encoding is consistent  
B. Because `.fit_transform` is computationally faster  
C. Because `.transform(X_test)` automatically tunes hyperparameters  
D. Because `.fit_transform` is illegal in scikit-learn  

---

### **4. If your dataset has missing values **in categorical columns**, which approach might be more appropriate **than dropping rows**?**  
A. Scale them with StandardScaler  
B. Use `LabelEncoder` only  
C. Impute missing values with the most frequent category or a placeholder (e.g. `NaN_category`)  
D. Convert them to numeric columns automatically  

---

### **5. After **LabelEncoding** your binary class target in a breast cancer dataset, the model accuracy is 65.45%. Which of the following best describes this result?**  
A. The model can’t be used because accuracy is below 70%  
B. The model predictions have about a 65% chance of matching the true labels on test data  
C. The accuracy is measuring the correlation coefficient of the features  
D. The model is guaranteed to have 65% accuracy on future unseen data  

---

# Answers

## **1. Which encoding is typically used for non-ordinal categorical features?**  
**Your answer:** **B. OneHotEncoder**  
**Correctness:** **B** is correct.  
**Explanation:**  
- **OneHotEncoder** is the standard choice for **nominal (non-ordinal)** categorical features, producing binary indicator columns for each category.  
- **LabelEncoder** is often reserved for encoding target labels or for features that are truly ordinal, but it only produces a single integer column that might mislead ML algorithms if the categories have no inherent order.

---

## **2. What is the difference between `LabelEncoder` and `OneHotEncoder`?**  
**Your answer:** **C. LabelEncoder outputs a single column with integer codes, whereas OneHotEncoder outputs multiple binary columns.**  
**Correctness:** **C** is correct.  
**Explanation:**  
- **LabelEncoder** turns each category into an integer code (e.g., 0,1,2,...).  
- **OneHotEncoder** creates one binary column per category (e.g., `[1, 0, 0]` for “blue,” `[0, 1, 0]` for “red,” etc.).  
- They serve different purposes, especially for **nominal** vs. **ordinal** data.

---

## **3. Why do we call `.transform(X_test)` (rather than `.fit_transform(X_test)`) with OneHotEncoder?**  
**Your answer:** **A. To keep the same categories learned from `X_train` so the test data encoding is consistent.**  
**Correctness:** **A** is correct.  
**Explanation:**  
- You fit the OneHotEncoder (or any transformer) on **training data** so it learns the categories. Then you **only** call `.transform(X_test)` to apply the exact same encoding structure to the test data.  
- If you used `.fit_transform(X_test)`, it might learn different categories or indexing, which breaks alignment with the training set encoding.

---

## **4. If your dataset has missing values in categorical columns, which approach might be more appropriate than dropping rows?**  
**Your answer:** **C. Impute missing values with the most frequent category or a placeholder**  
**Correctness:** **C** is correct.  
**Explanation:**  
- Dropping rows with missing values can reduce your dataset significantly and cause bias.  
- **Imputation** replaces missing categories with a sensible default (like “unknown” or the most frequent category).  
- This preserves data and can be automated (e.g., `SimpleImputer(strategy='most_frequent')`) within a pipeline.

---

## **5. After LabelEncoding your binary class target in a breast cancer dataset, the model accuracy is 65.45%. Which statement best describes this result?**  
**Your answer:** **B. The model predictions have about a 65% chance of matching the true labels on test data.**  
**Correctness:** **B** is correct.  
**Explanation:**  
- An accuracy of 65.45% means that **on the test set** (with its true labels), the model predictions match the real labels about 65% of the time.  
- It doesn’t guarantee the same accuracy on future unseen data, nor is it un-useable or a correlation measure. It’s simply the fraction of correct predictions in that test sample.
