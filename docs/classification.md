Key
---

SL = slightly lowered results
L = lowered results 
SH = slightly improved results
H = improved results

Pre-Milestone 
- cleaned data
- made common extractors for data
- bin naive bayes into several buckets
- very high bias in features

11/25
- Reimplement naive bayes using scikit-learn
- Choice between Multinomial and Gaussian.
- Chose MultinomialNB
- Accuracy of < 0.01 with all features
- Residual analysis >> good performance
- When plotted predictions, found very high bias toward 80

11/26
- Smoothed out a some feature values. Create dummy values for some. Reduce some multinomial features to binomials (these should originally be at most trinomial, but there were different values for error codes)
- drug1Type feature produces high bias towards > 70 minutes for MultinomialNB. This implies that drug1Type 
is not a great feature to use (likely is dependent on some other feature)
- Produced training / testing error graph; weird spikes between 0-5000 features,
gets more evened out btwn 10k-20k, and after 20k, both rise a bit, but there's a consistent "gap" between training and test data after 10k.
- After doing better feature processing / selection, found very high bias toward 10-20 bucket

11/27
- Introduced idea of PCA for visualizing distribution of features + getting better performance
- Best performance still 48%
- Residuals didn't offer much information
- Predictions fell into 3 classes: {10-20, 50-60, 60-70}
- Need to find features that separate 10-20 for 20-30

11/28
- Best set of features ("isReferred", "seenBefore", and "pastVisitsLastYear") yield at best 50% accuracy
- Decision trees no better than MultinomialNB (46.7%)
- Segregating labels to bucket sizes of 25 instead of 10 yielded 0.727 accuracy, probably due to the fact that all buckets fall into the 0-25 group
- Combining isReferred and seenBefore into one multinomial feature produces slightly less accurate but more varied results
- Try to restrict labelings. We will make weaker predictions for > 70 minutes.
- Clsutering features into groupings didn't add too much to predictions
- Adding kmeans in front to smooth continuous features only and then adding the centroid assignment as a feature lowers accuracy
- Ensemble methods achieve highest training set result, Naive Bayes achieves best testing set result

TODO:
- Try grouping together payment + drug type features
- Kmeans on more variations of continuous features:

	- (weight, height, temperature) - seemed to separate y's most; these were handpicked.(SL)
	- (...) try the predefined cluster of features from Yannis's kmeans code. (SL)
	- try redefining clusters
	- 

12/6 
- Use PCA to combine linearly related featuers such as drug1Type. Doing so gives predictions of 

12/7 
- Try to even out number of labels (i.e. reduce ones that are 10 randomly). Results didn't help

12/9
- Try to get bigger dataset. Difficult because the mappings are off and there aren't consistent set of features between 2003 to 2010
- Tried RFC's with 20 to 100 plots. Accuracy was the same (0.51044 accuracy on training data,  0.48484 accuracy on test data). Average prediction difference was 8.06 minutes
- Finallly tried logistic regression. We get accuracy of 0.50535, an improvement over Naive Bayes but slightly lower than Random Forests. Interestingly, we are mispredicting even though we're doing a pretty good job
- Logistic regression with weight tuning (give less weight to the 10's); with weights `{10: 0.5}`, we got accuracy of 0.49953.
- What's going on with the underlying data???? It's so inherently noisy because...why? hard to separate out features. My guess is that most features are not that great predictors by themselves of time with md, and the rest of the features don't cluster well.

12/10
- Maybe a better idea would be to predict physician office type?
- We have currently a mix of office types. Ok, taking out office types definitely gives an improvement. We get higher accuracy when we predict on training data vs predicting on test data when we filter out by officeSettings (non-private practice). (.53 accuracy for NB and RFC)
- For private practice, get no improvement for test or training

| smthg               |	Multinomial NB 	|	Random Forest Classifier   |	Logistic Regression 	| 
----------------------|-----------------|----------------------------------|----------------------------|
| Accuracy (percent)  | 0.485 		| 	0.478 			   |   	0.476  			|
| Avg Error (minutes) |	7.30    	| 	7.56 			   |			7.38 	|



