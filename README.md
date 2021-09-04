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

In this 1st iteration I have tested the following things:
* Different features: some with binned nummerical variables, some with Normalizing transformation (i.e.: Box-Cox)
* Different resampling techniques in order to deal with the imabalanced label: Undersampling, Oversampling, SMOTE.
* Different models for classification: Naive-Bayes, Logistic Regression, SVC, Random Forest, XGBoost

**One important thing**: Resampling was done during CV with imblearn pipeline, and NOT before. Performing resampling before CV leads to overfitting, as the validation dataset may contain duplicated samples from the training dataset. One should be careful when doing this.

There was not a clear winner in neither the resampling and modelling techniques. Oversampling with Random Forest was marginally better so that is the one thas has been finally used.

Although the final training AUC was 0.86, the final testing AUC was just 0.79. As a conclusion, the model seems to be overfitting and this issue should be addressed in the following iterations. Age was the most important feature.


## Aknowledgements
* General information and modelling (Option 2 from 'AUC Performance measurement using 2 different codes'): https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/
* Modelling and AUC plotting (Option 1 from 'AUC Performance measurement using 2 different codes'): https://imbalanced-learn.org/stable/auto_examples/applications/plot_over_sampling_benchmark_lfw.html
* Model tuning with RandomizedSearchCV: https://stackoverflow.com/questions/61453795/using-sklearns-randomizedsearchcv-with-smote-oversampling-only-on-training-fold
* Feature Importances: https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html
