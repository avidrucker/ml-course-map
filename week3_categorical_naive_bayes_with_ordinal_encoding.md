# Week 3: Categorical Naive Bayes with Ordinal Encoding
**Topics**: CategoricalNB, Ordinal encoding for categorical features

- (assertion) CategoricalNB models categorical features by estimating probabilities of feature values per class directly.
- (assertion) OrdinalEncoder transforms categorical features into integer codes, enabling ML models to process them.
- (assertion) Even after encoding, the order assigned by OrdinalEncoder does not imply a true numerical order of categories.

**Tasks**:
- Use OrdinalEncoder to convert categorical features (like "rainy", "sunny") into numeric codes.
- Implement CategoricalNB on a weather dataset with categorical features: fit, predict, and evaluate accuracy.
- Experiment with different training/test splits to assess model stability.
