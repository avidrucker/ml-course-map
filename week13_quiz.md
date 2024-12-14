## Questions

### 1. Why do we typically **perform PCA** on a dataset?
A. To replace cross-validation as a validation technique  
B. To generate synthetic labels for supervised learning  
C. To reduce the dimensionality of the feature space while retaining most of the variance  
D. To automatically tune hyperparameters for kNN  

---

### 2. In the provided notebook, what does **`PCA(n_components=0.85)`** specifically do?
A. It forces the PCA model to create 1 principal component exactly  
B. It picks the number of components necessary to capture 85% of the variance in the data  
C. It sets the maximum number of principal components to 85  
D. It randomly assigns 85% of features to be principal components  

---

### 3. After splitting the Iris dataset with `train_test_split(X, y, test_size=0.2, random_state=142)`, how many samples from the original 150 go into the **training set**?
A. 120 samples  
B. 100 samples  
C. 80 samples  
D. 142 samples  

*(Hint: the Iris dataset has 150 samples total; 20% is used for testing.)*

---

### 4. Which function was used in the notebook to measure **accuracy** of the kNN classifier on the test set?
A. `confusion_matrix`  
B. `classification_report`  
C. `accuracy_score`  
D. `precision_score`  

---

### 5. By default, the **Iris dataset** has 4 features: Sepal Length, Sepal Width, Petal Length, and Petal Width. After a successful **PCA transform** to capture 85% variance, the transformed dataset had shape `(n_samples, n_components)`. Which statement is true about the PCA transformation in the notebook?
A. It resulted in exactly 1 principal component  
B. It resulted in exactly 2 principal components  
C. It retained all 4 features (no reduction occurred)  
D. It added additional synthetic features to the dataset  

---

### 6. After PCA transformation, the data has fewer features. **Which step in the notebook ensures** that both the training and test sets are reduced to the **same number** of principal components?
A. Fitting PCA separately on X_train and X_test  
B. Fitting PCA on X_train and using the same PCA model to transform X_test  
C. Fitting PCA on the entire dataset, ignoring train/test split  
D. Fitting PCA repeatedly for every batch  

---

### 7. What is the purpose of **`pca.explained_variance_ratio_`**?
A. It stores the confusion matrix produced by PCA  
B. It returns the top features of the dataset  
C. It shows how much variance each principal component captures  
D. It shows the classification accuracy of the PCA model  

---

### 8. **Why do we evaluate precision, recall, and F1-score for each species** in the Iris classification task?
A. Because each species has different features, making accuracy unreliable  
B. To get a more detailed view of how well the model performs per class, beyond just overall accuracy  
C. Because there is no accuracy metric available in scikit-learn  
D. Because the dataset is too large to rely on accuracy alone  

---

### 9. **Comparing the kNN results before and after PCA** (retaining 85% variance), which outcome best describes the expected result?
A. The PCA-transformed model always has 100% accuracy  
B. The PCA-transformed model has drastically worse performance because of lost data  
C. The PCA-transformed model’s performance could be slightly lower or similar, but might be beneficial if it reduces overfitting or speeds up training  
D. The PCA-transformed model’s performance is guaranteed to be higher than the original  

---

### 10. If you wanted to **reconstruct** the full 4-feature data from the PCA-transformed data (where only 85% variance is kept), which function would you call?
A. `pca.inverse_transform(...)`  
B. `pca.inverse_components_(...)`  
C. `pca.explained_variance_ratio_(...)`  
D. `pca.recover_features(...)`  

---

## Answers

---

### 1. Why do we typically **perform PCA** on a dataset?  
**Your answer:** C. To reduce the dimensionality of the feature space while retaining most of the variance  
**Correct answer:** C  
**Explanation:** Principal Component Analysis is primarily used to lower the number of features (dimensionality) while preserving most of the dataset’s variance.

---

### 2. In the provided notebook, what does **`PCA(n_components=0.85)`** specifically do?  
**Your answer:** B. It picks the number of components necessary to capture 85% of the variance in the data  
**Correct answer:** B  
**Explanation:** Setting `n_components=0.85` tells PCA to automatically determine how many principal components are needed so that 85% of the original variance is retained.

---

### 3. How many samples go into the **training set** after `train_test_split(X, y, test_size=0.2, random_state=142)` on 150 samples?  
**Your answer:** A. 120 samples  
**Correct answer:** A  
**Explanation:** With 20% (30 samples) for the test set, the training set indeed has 120 samples remaining.

---

### 4. Which function was used to measure **accuracy** of the kNN classifier on the test set?  
**Your answer:** C. `accuracy_score`  
**Correct answer:** C  
**Explanation:** The notebook shows `accuracy_score(y_test, y_pred)`, which computes the fraction of correct predictions.

---

### 5. After a PCA transform retaining 85% variance, the notebook’s transformed dataset shape was `(n_samples, 1)`. Which statement is true?  
**Your answer:** A. It resulted in exactly 1 principal component  
**Correct answer:** A  
**Explanation:** The code snippet notes that around 85% of the variance can be captured by 1 principal component for this particular dataset. This is specific to the Iris dataset.

---

### 6. Which step ensures **both the training and test sets** are reduced to the same number of principal components?  
**Your answer:** B. Fitting PCA on X_train and using the same PCA model to transform X_test  
**Correct answer:** B  
**Explanation:** You fit PCA only on the training set (i.e., `pca.fit(X_train)`) and then apply the same transform (`pca.transform(X_test)`). That way both sets end up in the same principal component space.

---

### 7. What is the purpose of **`pca.explained_variance_ratio_`**?  
**Your answer:** C. It shows how much variance each principal component captures  
**Correct answer:** C  
**Explanation:** The array returned by `pca.explained_variance_ratio_` indicates the percentage of variance explained by each principal component, in descending order.

---

### 8. **Why do we evaluate precision, recall, and F1-score for each species** in the Iris classification task?  
**Your answer:** B. To get a more detailed view of how well the model performs per class, beyond just overall accuracy  
**Correct answer:** B  
**Explanation:** While overall accuracy is useful, precision/recall/F1 per class provides a more granular assessment of performance, which is especially insightful in multi-class settings.

---

### 9. **Comparing the kNN results before and after PCA** (retaining 85% variance), which outcome best describes the expected result?  
**Your answer:** C. The PCA-transformed model’s performance could be slightly lower or similar, but might be beneficial if it reduces overfitting or speeds up training  
**Correct answer:** C  
**Explanation:** PCA can lead to a slight drop or sometimes slight improvement in performance. The main advantage is dimensionality reduction, which can aid interpretability or reduce overfitting.

---

### 10. If you wanted to **reconstruct** the full 4-feature data from the PCA-transformed data, which function would you call?  
**Your answer:** A. `pca.inverse_transform(...)`  
**Correct answer:** A  
**Explanation:** `pca.inverse_transform` maps data from principal component space back to the original feature space.
