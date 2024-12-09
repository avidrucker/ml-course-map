PCA (Principal Component Analysis) (topic)
- (assertion) PCA finds orthogonal directions (principal components) of maximum variance in the data.
- (assertion) The first principal component explains the largest portion of the variance, subsequent components explain decreasing amounts.
- (assertion) PCA can reduce data dimensionality while retaining most variance, helping to avoid overfitting and improve computational efficiency.
- (assertion) Dropping lower-variance principal components approximates the original data but may lose some detail.
- (task) Use PCA with n_components=2 to visualize high-dimensional data in 2D.
- (task) Use PCA to project data onto the first principal component and reconstruct the data from this single dimension. Compare the reconstructed data to the original to see information loss.
- (task) Integrate PCA into a pipeline with scaling and a chosen classifier. Evaluate how PCA impacts model performance and training time.
- (task) Plot explained_variance_ratio_ to decide how many components to retain for balancing dimensionality reduction with variance retention.
