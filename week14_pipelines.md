# Week 14: ML Pipelines  
- (assertion) A pipeline chains together preprocessing steps and a final estimator, ensuring a repeatable and clean workflow.  
- (assertion) By incorporating transformations like scaling, PCA, or feature selection into a pipeline, data leakage can be prevented.  
- (assertion) Pipelines make it easier to tune parameters for multiple steps simultaneously, improving the model selection process.

- How to determine if stratification is necessary (topic)  
  - (assertion) Stratification ensures that each train/test split maintains the same proportion of each class as in the full dataset.  
  - (assertion) Without stratification, splits might inadvertently produce class imbalance in the training or test sets.  
  - (assertion) Stratification is particularly important for datasets with highly imbalanced class distributions.

- Jupyter Notebook code writing & data comprehension (topic)  
  - (assertion) Understanding the shape and distribution of features is essential before model training.  
  - (assertion) Exploratory data analysis, including plotting histograms or boxplots, helps identify outliers and skewed distributions.  
  - (assertion) Writing clean, commented Jupyter notebooks improves reproducibility and collaboration.

- (task) Configure a pipeline object to include data scaling, PCA, and a classifier.  
- (task) Evaluate multiple pipelines with different classifiers to find the best model for your dataset.  
- (task) Incorporate cross-validation into a pipeline to tune hyper-parameters.  
- (task) Perform exploratory data analysis (e.g., plotting distributions) within a Jupyter notebook to understand data patterns and anomalies.  
- (task) Determine if stratification should be applied by checking class distributions and applying `StratifiedKFold` if necessary.
- (task) Adjust pipeline steps and parameters (e.g., `n_components` in PCA, `C` in SVC) and observe changes in model performance.
