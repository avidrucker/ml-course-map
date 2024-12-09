# week 6: logistic regression

- (assertion) Logistic regression models the probability of a class by applying a sigmoid function to a linear combination of inputs, ensuring outputs remain within [0,1].
- (assertion) Logistic regression is used for classification problems where the output is categorical and probabilities are required, unlike linear regression which predicts continuous values.
- (assertion) A decision boundary in logistic regression is determined by selecting a threshold (often 0.5) on the predicted probabilities, classifying samples above the threshold as one class and below as another.
- (task) Implement logistic regression in a Jupyter Notebook using scikit-learn:
  - load and preprocess data,
  - encode categorical variables if needed,
  - fit a LogisticRegression model,
  - predict, and
  - evaluate the classification metrics.
- (assertion) Adjusting the threshold for logistic regression predictions shifts the decision boundary, allowing control over the trade-off between precision and recall.
