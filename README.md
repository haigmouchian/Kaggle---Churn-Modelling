# Kaggle---Churn-Modelling

## Information

In this notebook I explore a Kaggle dataset which collects employee data from a company (Age, Gender, Salary, CreditScore, etc). 

The main objective is to predict whether or not the employee has left the company or not (aka Churning Rate).

The problem is a **binary Classification** one:
* 0: No churn
* 1: Churn

It is also an **imbalanced dataset**:
* 0: 79% of the data
* 1: 21% of the data
Due to this, **resampling techniques** were used.

The main scoring metric used was **ROC-AUC**.

All the code until the modelling (EDA, Data prep, Feature Engineering) was done by me.

In the 1st iteration I have tested the following things:
* Different features: some with binned nummerical variables, some with Normalizing transformation (i.e.: Box-Cox)
* Different resampling techniques in order to deal with the imabalanced label: Undersampling, Oversampling, SMOTE.
* Different models for classification: Naive-Bayes, Logistic Regression, SVC, Random Forest, XGBoost

**One important thing**: Resampling was done during CV with imblearn pipeline, and NOT before. Performing resampling before CV leads to overfitting, as the validation dataset may contain duplicated samples from the training dataset. One should be careful when doing this.

There was not a clear winner in neither the resampling and modelling techniques. Oversampling with Random Forest was marginally better so that is the one thas has been finally used.

Although the final training AUC was 0.86, the final testing AUC was just 0.79. As a conclusion, the model seems to be overfitting and this issue should be addressed in the following iterations. Age was the most important feature.

In the 2nd iteration, I tried to improve on the score from the 1st iteration using feature engineering techniques. I tested the following approaches:
* Recursive Feature Elimination (RFE) with Random Forest
* Principal Component Analysis (PCA)
* The Lasso Regression

All of these techniques were used within the same pipeline as the 1st iteration:
* Standard Scaler
* Oversampling
* Random Forest model
* I also used the same transformed dataset from the 1st iteration in order to get comparable results.

The goal of these techniques is to extract the important features and drop the irrelevant ones, in order to improve performance.

The best technique was RFE. In my opinion PCA didn't work so well because there is not much multi-collinearity between features, and Lasso didn't work so well because there is not much linear correlation between the features and the label.

With the RFE + RF approach, I was able to drop 4 of the 15 features without loss of performance, having 11 features instead of the original 15. After fine tuning the hyperparameters via RandomizadSearchCV, I trained the model with the entire Train set and made predictions on the Test set. The improvement over the 1st iteration is negligible, so the model is still overfitting. However, it is important to note that with RFE we reached the same performance using 11 features instead of 15 (27% reduction). This might not be such an important thing here as there are only 10000 samples. But a simpler model would be easier to scale and consumes less time and computing power.


## Aknowledgements
* General information and modelling (Option 2 from 'AUC Performance measurement using 2 different codes'): https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/
* Modelling and AUC plotting (Option 1 from 'AUC Performance measurement using 2 different codes'): https://imbalanced-learn.org/stable/auto_examples/applications/plot_over_sampling_benchmark_lfw.html
* Model tuning with RandomizedSearchCV: https://stackoverflow.com/questions/61453795/using-sklearns-randomizedsearchcv-with-smote-oversampling-only-on-training-fold
* Feature Importances: https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html
