Pre-Milestone 
- cleaned data
- made common extractors for data
- bin naive bayes into several buckets
- very high bias in features
- tried 

11/25
- Reimplement naive bayes using scikit-learn
- Choice between Multinomial and Gaussian.
- Chooes MultinomialNB
- Accuracy of < 0.01 with all features
- Residual analysis >> good performance

11/26
- Smoothed out a some feature values. Create dummy values for some. Reduce some multinomial features to binomials (these should originally be at most trinomial, but there were different values for error codes)
- drug1Type feature produces high bias, so omitting it as a feature 
- Produced training / testing error graph; weird spikes between 0-5000 features,
gets more evened out btwn 10k-20k, and after 20k, both rise a bit, but there's a consistent "gap" between training and test data after 10k.

11/27
- Met up. Introduced idea of PCA for visualizing distribution of features.
- Best performance still 48%. 
- Looked at residuals - found lots of variation with 

11/28
- Best set of features ("isReferred", "seenBefore", and "pastVisitsLastYear") yield at best 50% accuracy
- Decision trees no better than MultinomialNB (46.7%).
- 