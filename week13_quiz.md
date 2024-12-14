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


Feel free to provide your answers (e.g., 1: A, 2: B, …). After you answer, I can provide detailed feedback on which were correct, which were incorrect, and why.
