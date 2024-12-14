## **Quiz on Week 11 Fundamentals**

### **1. Support Vector Classifier and Linear Separability**
In the `trainData.csv` SVM exercise, we saw that the two-class data was not linearly separable. Which of the following statements best describes why the data is “not linearly separable”?

A. The data has only one feature, and we need at least two features for linear separability.  
B. There’s no single straight line (hyperplane in 2D) that perfectly splits the two classes without misclassification.  
C. The SVM can’t converge on the dataset due to improper parameters.  
D. Because the dataset is too large for a linear classifier to handle.

---

### **2. SVM’s `C` Parameter**
You used `SVC(kernel='rbf', C=1000)` for classification. What happens in general as the value of `C` is **increased**?

A. The model becomes more tolerant of misclassifications (softer margins).  
B. The model tries harder to correctly classify all training points, leading to harder margins.  
C. The kernel switches automatically from RBF to linear.  
D. The algorithm runs fewer iterations.

---

### **3. Drawing an SVM Decision Boundary**
In the solutions, a 2D decision boundary was plotted using a meshgrid (`xx`, `yy`) and then calling `clf.predict` on every point in that mesh. Why do we do this?

A. It’s the only way to compute the model’s accuracy.  
B. We must fill in missing values in the dataset using mesh points.  
C. It helps visualize where the classifier changes labels across the feature space.  
D. It ensures the SVM gets more training data.

---

### **4. k-Means Clustering Movement (Center Convergence)**
In the `KMeans-plottingClusterMovement` solution, the code iterates `max_iter` from 1 to 6 and logs the cluster centers at each iteration. Which of the following best describes **why** the cluster centers move each iteration?

A. k-Means randomly picks a new centroid location each iteration.  
B. After each assignment step, each centroid is updated to the mean of the data points assigned to it.  
C. The cluster centers always move in exactly the same direction, independent of the data.  
D. The cluster centers move because PCA is being applied behind the scenes.

---

### **5. Initialization of k-Means**
When using `init=np.array([[100,100], [400,200]])`, why might the final clusters differ compared to using the default initializer?

A. k-Means always converges to the same solution, so the initializer doesn’t matter.  
B. The default initializer uses only a single centroid.  
C. Different initial centroids can lead to different local optima for the cluster assignments.  
D. The array of `[100,100]` and `[400,200]` is invalid and will cause an error.

---

### **6. Elbow Method (Conceptual)**
The elbow method plots the **inertia** (within-cluster sum of squares) against the number of clusters *k*. What are you typically looking for in that plot to decide on a good **k**?

A. A point where the plot starts forming loops.  
B. The first cluster count that yields zero inertia.  
C. A “bend” or “elbow” in the plot, where adding more clusters doesn’t drastically reduce inertia further.  
D. The cluster count that maximizes inertia.

---

### **7. Outlier Removal Strategy**
In the outliers exercise, 10% of the data points (with the highest residual errors) were removed before refitting the linear regression. What was the **effect** on the regression’s R² score?

A. It went down significantly, showing that outlier removal hurt the model.  
B. It increased substantially (from ~0.52 to ~0.95).  
C. It stayed exactly the same.  
D. It could not be computed after outlier removal.

---

### **8. Residual Error in Outliers Exercise**
The function `outlierCleaner` computed `error = np.abs(y - y_predicted)` and removed points with the largest errors. Why does removing high-error points often *improve* the final linear regression’s R²?

A. Because removing any data always guarantees better performance.  
B. Because outliers can skew the linear regression fit, and removing them lets the model fit the majority of data better.  
C. Because the shape of the linear regression line never changes.  
D. Because R² is unaffected by the distance of outliers.

---

### **9. KMeans vs. SVM**
Which statement best distinguishes **k-Means** from **SVC** (Support Vector Classifier) in these notebooks?

A. Both algorithms are supervised learning methods.  
B. k-Means requires labeled data, while SVC does not.  
C. k-Means is unsupervised (clusters data without prior labels), while SVC is supervised (needs known labels for training).  
D. They are the same algorithm, just implemented differently.

---

### **10. Predicting Test Data with SVM**
In the SVM solution code, you predicted on `X_test` and compared the predictions with `y_test`. The accuracy was around **0.94**. What does this accuracy value represent?

A. 94% of the data points in `X_test` were assigned to the correct cluster.  
B. 94% of the test data points were correctly classified into their true labels.  
C. The model’s R² value is 0.94 for the test set.  
D. The average distance from the decision boundary is 0.94.

---

## Answers

---

### **1. Support Vector Classifier and Linear Separability**
**Question:** Why is the data not linearly separable in `trainData.csv`?  
**Your answer:** **B** – *There’s no single straight line (hyperplane in 2D) that perfectly splits the two classes without misclassification.*  
**Correct answer:** **B**  
**Explanation:** That’s correct. “Not linearly separable” means you can’t draw one straight line that classifies all points correctly.

---

### **2. SVM’s `C` Parameter**
**Question:** What happens as `C` **increases**?  
**Your answer:** **B** – *The model tries harder to classify all training points correctly (harder margins).*  
**Correct answer:** **B**  
**Explanation:** A higher `C` penalizes misclassifications more strongly, leading to a model that attempts to create a “hard” margin (potentially overfitting).

---

### **3. Drawing an SVM Decision Boundary**
**Question:** Why do we use a meshgrid and predict on every point in that grid?  
**Your answer:** **C** – *It helps visualize where the classifier changes labels across the feature space.*  
**Correct answer:** **C**  
**Explanation:** Exactly. Generating a dense grid and calling `predict` for each point is a standard way to plot the decision boundary in 2D.

---

### **4. k-Means Clustering Movement (Center Convergence)**
**Question:** Why do cluster centers move each iteration in k-Means?  
**Your answer:** **B** – *After the assignment step, each centroid is updated to the mean of the data points assigned to it.*  
**Correct answer:** **B**  
**Explanation:** The cluster centers shift to the average location of points in each cluster after every iteration.

---

### **5. Initialization of k-Means**
**Question:** Why might using `init=np.array([[100, 100], [400, 200]])` lead to different final clusters than default init?  
**Your answer:** **C** – *Different initial centroids can lead to different local optima for the cluster assignments.*  
**Correct answer:** **C**  
**Explanation:** k-Means can converge to different solutions depending on initial centroid positions.

---

### **6. Elbow Method (Conceptual)**
**Question:** On the distortion vs. number-of-clusters plot, what’s the key pattern we look for?  
**Your answer:** **C** – *A “bend” or “elbow” in the plot, where adding more clusters doesn’t drastically reduce inertia.*  
**Correct answer:** **C**  
**Explanation:** Correct—the “elbow” indicates a good trade-off between cluster count and within-cluster variance.

---

### **7. Outlier Removal Strategy**
**Question:** After removing 10% outliers, how did R² change in the regression exercise?  
**Your answer:** **B** – *It increased substantially (from ~0.52 to ~0.95).*  
**Correct answer:** **B**  
**Explanation:** Removing severe outliers allowed the regression line to fit the remaining data much better, boosting R² from about 0.52 to ~0.95.

---

### **8. Residual Error in Outliers Exercise**
**Question:** Why does removing high-error points often improve the final R²?  
**Your answer:** **B** – *Because outliers can skew the linear regression fit, removing them allows the model to fit the majority better.*  
**Correct answer:** **B**  
**Explanation:** Exactly. Large outliers hurt the regression fit. Removing them generally improves how well the model fits the bulk of points.

---

### **9. KMeans vs. SVM**
**Question:** Which statement best distinguishes k-Means from SVC?  
**Your answer:** **C** – *k-Means is unsupervised, while SVC is supervised.*  
**Correct answer:** **C**  
**Explanation:** Spot on. k-Means needs no labels; SVM/SVC requires labeled training data.

---

### **10. Predicting Test Data with SVM**
**Question:** With an accuracy near 0.94, what does that accuracy represent?  
**Your answer:** **B** – *94% of the test data points were correctly classified into their true labels.*  
**Correct answer:** **B**  
**Explanation:** That’s precisely what an accuracy of 0.94 means for a classifier.
