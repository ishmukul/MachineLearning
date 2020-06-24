# ----------------------
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer

# -----------------------
# Import Data sets
DataSet = pd.read_csv('data/BreastCancer.csv')
DataSet = DataSet.dropna()
# DataSet.describe()  # Diagnostic
x_columns = np.r_[1:10]
X = DataSet.iloc[:, x_columns].values  # X feature vector
# X = np.c_[np.ones(x.shape[0]), x]  # Add 1s as column vector; Not required for internal functions
y = DataSet.iloc[:, len(DataSet.columns) - 1].values  # y values
# y.shape = len(y),1  # Used to shape the y array.
# print('Shape of X feature vector is ', X.shape)  # Diagnostic
# print('Shape of y vector is ', y.shape)  # Diagnostic

# Changing shape of y is giving a warning in in-built function of Logistic Regression.

# -----------------------
# Timing discussion
start_time = timer()
# Applying Logistic regression model as Classifier to DataSet
clf = LogisticRegression()  # Taken from sklearn
clf.fit(X, y)  # Fit model
end_time = timer()
print("==============================================")
print('Printing fit parameters and accuracy on whole DataSet')
print('intercept:', clf.intercept_)
print('coefficient:', clf.coef_)
print('Accuracy with fit on whole data is %f percent' % (100 * clf.score(X, y)))
print('Time taken %f seconds' % (end_time-start_time))
print('==============================================\n\n')

# -----------------------------
# Splitting data in training and test set.

# VALIDATION SET APPROACH (Hold out method):
# Split data into training set and validation set.
# Steps are:
# 1) Split data into 50% training set.
# 2) Fit training data with classifier.
# 3) Use fitted model to predict **validation set**
# 4) Find out Mean square error (MSE) for the fit.

# Split DataSet into training + test data
start_time = timer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  # sklearn function to split data
# x_train.shape  # Diagnostic

clf_train = LogisticRegression()
clf_train.fit(X_train, y_train)

print('==============================================')
print('Printing fit parameters and accuracy with 50% split on DataSet')
print('intercept:', clf_train.intercept_)
print('coefficient:', clf_train.coef_)
print('Accuracy with 50:50 Hold-Out on training set is %f percent' % (100 * clf_train.score(X_train, y_train)))

# Accuracy check I am using following metric for accuracy:
# 1) Predict y_test_pred values from x_test set.
# 2) Calculate mean square error for (y_pred-y_value)$^2$
# 3) Number in step 3 actually gives how many predictions were wrong, since y has only 0 or 1 value.
#       Therefore, in case of any wrong prediciton, square of (0-1) or (1-0) gives only 1.
# 4) Take ratio of (y_test - wrong pred)/(y_test)

y_test_pred = clf_train.predict(X_test)
mse = ((y_test_pred - y_test) ** 2).sum()
accuracy = (len(y_test) - mse) / len(y_test)
print('Accuracy with 50:50 Hold-Out on test set using Mean Square Error (MSE) is %f percent ' % (100 * accuracy))

cm = confusion_matrix(y_test, y_test_pred)  # Create Confusion Matrix for Evaluation
accuracy = cm.trace() / cm.sum()  # Accuracy with confusion matrix

end_time = timer()
print('Accuracy with 50:50 Hold-Out on test set using Confusion Matrix (CM) is %f percent ' % (100 * accuracy))
print('Time taken %f seconds' % (end_time-start_time))
print('==============================================\n\n')

# --------------------------------------------
# --------------------------------------------
# CROSS VALIDATION - k fold technique
# Divide data into k subsets and train on data excluding 1 subset.
# In other words, make 1 subset (n/k) as test/validation DataSet.

# Using sklearn functions
start_time = timer()
k = 10
clf_kfold = LogisticRegression()
kfold_predicitons = cross_val_predict(clf_kfold, X, y, cv=k)
kfold_scores = cross_val_score(clf_kfold, X, y, cv=k)
# print(kfold_predicitons.mean())
accuracy = kfold_scores.mean()
end_time = timer()
print('==============================================')
print('Testing with k-fold method')
print('Number of folds used is %d ' % k)
print('The accuracy from k-fold (sklearn) method is %f ' % (100 * accuracy))
print('Time taken %f seconds' % (end_time-start_time))

# Trying writing from scratch
start_time = timer()
ss = int(X.shape[0] / k)
accuracy_kfold = []  # Empty array
accuracy_kfold_cm = []  # Empty array
for i in range(k):
    # x_kfold_ss = k-fold subset(ss) form X data
    # i * ss = start of index in loop
    # (i+1) * ss = end of loop
    #  ss is just length divided by k folds
    x_kfold_test = DataSet.iloc[i * ss: (i + 1) * ss, x_columns].values  # Test subset
    x_kfold_train = DataSet.iloc[np.r_[0: i * ss, (i+1) * ss: k * ss], x_columns].values  # Training subset (
    # larger in size). It is combination of two subsets;
    # 1) Previous to test set, starting from Zero (0: i * ss),
    # 2) After test set till the end k ((i+1) * ss: k * ss)
    y_kfold_test = DataSet.iloc[i * ss: (i + 1) * ss, len(DataSet.columns) - 1].values  # Test Y
    y_kfold_train = DataSet.iloc[np.r_[0: i * ss, (i+1) * ss: k * ss], len(DataSet.columns) - 1].values  # Training Y
    # print(x_kfold_test[1], y_kfold_test[1])  # Diagnostic Check if data is actually changing
    # print(x_kfold_train.shape, x_kfold_test.shape)  # Diagnostic check shapes
    clf_kfold = LogisticRegression()  # Logistic Regression
    clf_kfold.fit(x_kfold_train, y_kfold_train)  # Fitting using Logistic Regression
    y_kfold_pred = clf_train.predict(x_kfold_test)  # Prediction using Fit model
    mse = ((y_kfold_pred - y_kfold_test) ** 2).sum()  # Mean Square Error
    accuracy1 = (len(y_kfold_test) - mse) / len(y_kfold_test)  # Accuracy
    accuracy_kfold.append(accuracy1)  # Filling of Accuracy array

    # Checking through confusion matrix
    cm = confusion_matrix(y_kfold_test, y_kfold_pred)  # Create Confusion Matrix for Evaluation
    accuracy_cm = cm.trace() / cm.sum()  # Accuracy with confusion matrix
    accuracy_kfold_cm.append(accuracy_cm)
end_time = timer()
print('Accuracy with k-fold (scratch) on test set using MSE is %f percent ' % (100 * np.average(accuracy_kfold)))
print('Accuracy with k-fold (scratch) on test set using CM is %f percent ' % (100 * np.average(accuracy_kfold_cm)))
print('Time taken %f seconds' % (end_time-start_time))
print('==============================================\n')


# --------------------------------------------
# --------------------------------------------
# LOO-CROSS VALIDATION - k = 1
# Divide data into k subsets and train on data excluding 1 subset.
# In other words, make 1 subset (n/k) as test/validation DataSet.

# print(DataSet.index)  # Diagnostic

# Using sklearn functions
# k = X.shape[0]
# clf_kfold = LogisticRegression()
# kfold_predicitons = cross_val_predict(clf_kfold, X, y, cv=k)
# kfold_scores = cross_val_score(clf_kfold, X, y, cv=k)
# # print(kfold_predicitons.mean())
# accuracy = kfold_scores.mean()
# print('==============================================')
# print('Testing with k-fold method')
# print('Number of folds used is %d ' % k)
# print('The accuracy from k-fold (sklearn) method is %f ' % (100 * accuracy))

# Trying writing from scratch
start_time = timer()
k = X.shape[0]
ss = int(X.shape[0] / k)
accuracy_kfold = []  # Empty array
accuracy_kfold_cm = []  # Empty array
print('==============================================')
print('Testing with LOOCV method')

for i in range(k):
    # x_kfold_ss = k-fold subset(ss) form X data
    # i * ss = start of index in loop
    # (i+1) * ss = end of loop
    #  ss is just length divided by k folds
    x_kfold_test = DataSet.iloc[i * ss: (i + 1) * ss, x_columns].values  # Test subset
    x_kfold_train = DataSet.iloc[np.r_[0: i * ss, (i+1) * ss: k * ss], x_columns].values  # Training subset (larger in size)
    y_kfold_test = DataSet.iloc[i * ss: (i + 1) * ss, len(DataSet.columns) - 1].values  # Test Y
    y_kfold_train = DataSet.iloc[np.r_[0: i * ss, (i+1) * ss: k * ss], len(DataSet.columns) - 1].values  # Training Y
    # print(x_kfold_test[1], y_kfold_test[1])  # Diagnostic Check if data is actually changing
    # print(x_kfold_train.shape, x_kfold_test.shape)  # Diagnostic check shapes
    # print(y_kfold_train.shape, y_kfold_test.shape)  # Diagnostic check shapes
    clf_kfold = LogisticRegression()  # Logistic Regression
    clf_kfold.fit(x_kfold_train, y_kfold_train)  # Fitting using Logistic Regression
    y_kfold_pred = clf_train.predict(x_kfold_test)  # Prediction using Fit model
    mse = ((y_kfold_pred - y_kfold_test) ** 2).sum()  # Mean Square Error
    accuracy1 = (len(y_kfold_test) - mse) / len(y_kfold_test)  # Accuracy
    accuracy_kfold.append(accuracy1)  # Filling of Accuracy array

    cm = confusion_matrix(y_kfold_test, y_kfold_pred)  # Create Confusion Matrix for Evaluation
    accuracy_cm = cm.trace() / cm.sum()  # Accuracy with confusion matrix
    accuracy_kfold_cm.append(accuracy_cm)  # Filling of Accuracy array
end_time = timer()
print('Accuracy with k-fold (scratch) on test set using MSE is %f percent ' % (100 * np.average(accuracy_kfold)))
print('Accuracy with k-fold (scratch) on test set using CM is %f percent ' % (100 * np.average(accuracy_kfold_cm)))
print('Time taken %f seconds' % (end_time-start_time))
print('==============================================\n')
