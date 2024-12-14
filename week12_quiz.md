### Week 12 Quiz Questions

1. **What is the main purpose of using k-fold cross-validation in model development?**  
   A. To reduce the size of the dataset  
   B. To systematically tune hyperparameters and reduce overfitting  
   C. To speed up training time  
   D. To visualize decision boundaries effectively  

2. **In the KNN example with `n_neighbors` from 1 to 20 (in steps of 2), how do we decide which `k` value is the "optimal number of neighbors"?**  
   A. By randomly picking the smallest `k`  
   B. By plotting a decision boundary for each `k` and inspecting it visually  
   C. By choosing the `k` that yields the highest average accuracy across the CV folds  
   D. By choosing the `k` that yields the lowest average accuracy across the CV folds  

3. **When using sklearn’s `train_test_split(X, y, test_size=0.2, random_state=42)`, what does `test_size=0.2` represent?**  
   A. 20% of the samples go to the training set, 80% to the test set  
   B. 20% of the samples go to the test set, 80% to the training set  
   C. 42% of the samples go to the test set, 58% to the training set  
   D. 20% of the features are used  

4. **After finding the optimal `k` in the first part (`test_size=0.2, random_state=42`), the best model's accuracy on the test set was printed. Which sklearn function was used for that?**  
   A. `recall_score`  
   B. `accuracy_score`  
   C. `classification_report`  
   D. `confusion_matrix`  

5. **Why do we mesh the coordinate space (using `np.meshgrid` and `np.c_`) before plotting the KNN decision boundary?**  
   A. To generate synthetic training data points  
   B. To create a grid that covers the attribute space so we can predict classes across the entire range of features  
   C. To do dimensionality reduction on the data  
   D. To randomly shuffle the data points before cross-validation  

6. **In the diabetes dataset (Pima Indians), which performance metric was chosen as the primary one, and why?**  
   A. Accuracy, because the classes are perfectly balanced  
   B. Precision, because it reduces false negatives  
   C. Recall, because minimizing false negatives is critical for medical diagnoses  
   D. F1-score, because it balances the dataset distribution  

7. **Which hyperparameters were tuned for the SVC model with GridSearchCV in the diabetes example?**  
   A. `n_neighbors`, `cv`, and `p`  
   B. `kernel`, `degree`, and `C`  
   C. `learning_rate`, `n_estimators`, and `max_depth`  
   D. `fit_intercept`, `solver`, and `max_iter`  

8. **In the second KNN experiment (with `train_test_split(X, y, test_size=0.1, random_state=4)`), what differs compared to the first experiment (test_size=0.2, random_state=42)?**  
   A. The number of neighbors used by KNN  
   B. The portion of data reserved for testing  
   C. The classification algorithm changes from KNN to SVC  
   D. The training data is now loaded incorrectly  

9. **What does `make_scorer(recall_score, average='binary')` do in the GridSearchCV context?**  
   A. Tells GridSearchCV to use recall as the metric instead of accuracy  
   B. Calculates the confusion matrix for binary classification  
   C. Forces the SVC to output probabilities rather than labels  
   D. Replaces cross-validation with a different sampling method  

10. **Comparing the final results of the Logistic Regression vs. SVC on the diabetes dataset, how do we interpret the recall scores?**  
   A. The higher the recall score, the fewer false negatives the model produces  
   B. The higher the recall score, the more false negatives the model produces  
   C. The recall score is irrelevant for imbalanced datasets  
   D. The recall score is the same thing as precision

---

**Here is the correctness of your answers**, followed by an explanation for each:

---

### Answers

1. **B** – Correct  
   - *Reason:* k-fold cross-validation is indeed used to systematically tune hyperparameters and reduce overfitting (among other benefits).

2. **C** – Correct  
   - *Reason:* We pick the `k` that gives the highest average accuracy (or highest chosen metric) across all cross-validation folds.

3. **B** – Correct  
   - *Reason:* `test_size=0.2` means 20% of the samples go into the test set, and 80% remain for training.

4. **C** – **Incorrect**  
   - *Correct answer:* **B** (`accuracy_score`)  
   - *Reason:* The code explicitly shows `test_accuracy = accuracy_score(y_test, y_pred)`. Although `classification_report` is also useful, the numeric accuracy printed came from `accuracy_score`.

5. **B** – Correct  
   - *Reason:* Creating a mesh grid over the `(x, y)` range is a way to systematically sample points in feature space so we can predict their classes and plot the decision boundary.

6. **C** – Correct  
   - *Reason:* The Pima Indians Diabetes dataset is imbalanced, and minimizing false negatives is critical for medical screening, so **Recall** is the chosen metric.

7. **B** – Correct  
   - *Reason:* The grid search tuned `kernel`, `degree` (for the polynomial kernel), and `C`.

8. **B** – Correct  
   - *Reason:* The second experiment changes the portion of data reserved for testing from 20% to 10% (and a different random seed), nothing else.

9. **A** – Correct  
   - *Reason:* `make_scorer(recall_score, average='binary')` tells GridSearchCV to use **recall** instead of accuracy as the scoring metric.

10. **A** – Correct  
   - *Reason:* The higher the recall score, the fewer false negatives the model produces (recall = TP / (TP + FN)).  

---

### Additional Explanations

#### What Are “CV Folds”?

- **CV** stands for **Cross-Validation**. In **k-fold cross-validation**, the dataset is split into *k* separate subsets (folds). Then for each iteration (out of *k*):
  - You train the model on *k-1* folds.
  - You test the model on the remaining 1 fold.
- This process repeats *k* times so that each fold serves as a test set exactly once. The performance metric (e.g., accuracy, recall) is averaged over the *k* folds to get a more robust estimate of how the model generalizes.  
- **“CV folds”** specifically refers to those *k* distinct partitions of the data used during cross-validation.

#### What Does It Mean to “Mesh the Coordinate Space”?

- When **plotting a decision boundary** in a 2D feature space (e.g., features `x` and `y`), we often:
  1. Create a grid (mesh) of points that spans the minimum to maximum values of `x` and `y`.
  2. Use the trained model to predict a label/class for each point in that grid.
  3. Color or contour-plot the grid according to the predicted classes.  
- This gives a continuous “map” of the classifier’s decisions over the entire 2D plane, showing where class boundaries lie. 
- **Meshing** basically means systematically creating a dense, evenly spaced set of points in the coordinate space so the decision boundary can be visualized as a smooth contour.
