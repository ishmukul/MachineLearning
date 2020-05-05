# See Readme on main page for details
# Keras based neural network classifier application for Banknote authentication dataset.


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score

from timeit import default_timer as timer

import logging

# Following lines are to suppress warning on GPU.
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# Defining an accuracy function based on Confusion matrix
def acc(yy_real, yy_pred):
    cm = confusion_matrix(yy_real, yy_pred)
    return cm.trace() / cm.sum()


# Import data using Pandas
data = pd.read_csv("data/data_banknote_authentication.txt")
X = data.iloc[:, :4].to_numpy()
Y = data.iloc[:, 4].to_numpy()

# # Scaling data such that mean=0 and standard deviation is 1
scale = StandardScaler()
X_scale = scale.fit_transform(X)
#
# Train Test spit using sklearn
X_train, X_test, Y_train, Y_test = tts(X_scale, Y, test_size=0.25, random_state=42)

# =======================================
# Using sklearn classifiers
# Support Vector Classifier
start_time = timer()
ker = 'rbf'
clf_svc = SVC(gamma='auto', kernel=ker)  # Kernels available: rbf = Gaus, linear, poly,
clf_svc.fit(X_train, Y_train)
Y_pred_svm = clf_svc.predict(X_test)
cm_svm = confusion_matrix(Y_test, clf_svc.predict(X_test))
acc_svm = 100 * acc(Y_test, Y_pred_svm)
f1_svm = f1_score(Y_test, Y_pred_svm)
# print('%0.4f' % clf_svc._gamma)
print(cm_svm)
# plot_confusion_matrix(clf_svc, X_test, y_test)
end_time = timer()
time_svm = end_time - start_time

# =======================================
# Random forest classifier
start_time = timer()
clf_rfc = RandomForestClassifier(max_depth=5, random_state=0)
clf_rfc.fit(X_train, Y_train)
Y_pred_rfc = clf_rfc.predict(X_test)
cm_rfc = confusion_matrix(Y_test, clf_rfc.predict(X_test))
acc_rfc = 100 * acc(Y_test, Y_pred_rfc)
f1_rfc = f1_score(Y_test, Y_pred_rfc)
print(cm_rfc)
# plot_confusion_matrix(clf_rfc, X_test, y_test)
end_time = timer()
time_rfc = end_time - start_time

# =======================================
# Decision Tree Classifier
start_time = timer()
clf_dtc = DecisionTreeClassifier(max_depth=5, random_state=0)
clf_dtc.fit(X_train, Y_train)
Y_pred_dtc = clf_dtc.predict(X_test)
cm_dtc = confusion_matrix(Y_test, clf_dtc.predict(X_test))
acc_dtc = 100 * acc(Y_test, Y_pred_dtc)
f1_dtc = f1_score(Y_test, Y_pred_dtc)
print(cm_dtc)
# plot_confusion_matrix(clf_dtc, X_test, y_test)
end_time = timer()
time_dtc = end_time - start_time

# =======================================
# =======================================
# Using a neural network - Keras
# Initiate model Keras

start_time = timer()
model = Sequential()

# Add layers to the model
model.add(Dense(units=10, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=10, activation='relu'))  # First hidden layer
model.add(Dense(units=10, activation='relu'))  # Second hidden layer
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(units=1, activation='sigmoid'))  # Output layer

# Optimizer algorithms
# opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # Stochastic Gradient Descent
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
# opt = keras.optimizers.Adagrad(learning_rate=0.01)
opt = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
# opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)


# Loss functions, several options available
# loss = 'mean_squared_error'
# loss = 'mean_absolute_error'
loss = 'binary_crossentropy'

# Compile model
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

# Fit model; Verbose=0 will not show print messages
model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)

# Calculate score and values
score = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)

Y_pred_keras = (model.predict(X_test) > 0.5).astype(int)
cm_keras = confusion_matrix(Y_test, Y_pred_keras)
acc_keras = acc(Y_test, Y_pred_keras)
f1_keras = f1_score(Y_test, Y_pred_keras)
print(cm_keras)
end_time = timer()
time_keras = end_time - start_time

# =======================================
# =======================================
# Print final comparisons of different classifiers
print('Time for Support Vector Machine: %f s, Accuracy: %0.2f, and F1 score = %0.2f' % (time_svm, acc_svm, f1_svm))
print('Time for Random Forrest Classifier: %f s, Accuracy: %0.2f, and F1 score = %0.2f' % (time_rfc, acc_rfc, f1_rfc))
print('Time for Decision Tree Classifier: %f s, Accuracy: %0.2f, and F1 score = %0.2f' % (time_dtc, acc_dtc, f1_dtc))
print('Time for Keras Neural Network: %f s, Accuracy: %0.2f, and F1 score = %0.2f' % (time_keras, score[1], f1_keras))

# Output:
# By comparing with different classifiers, it seems SVM and Neural Networks are producing similar results. It may be
# possible that data is separable and SVM works fine. Other classifiers are also giving more than 95% accuracy.
#
# Time for Support Vector Machine: 0.004812 s, Accuracy: 100.00, and F1 score = 1.00
# Time for Random Forrest Classifier: 0.140958 s, Accuracy: 98.83, and F1 score = 0.99
# Time for Decision Tree Classifier: 0.003153 s, Accuracy: 97.38, and F1 score = 0.97
# Time for Keras Neural Network: 1.231440 s, Accuracy: 1.00, and F1 score = 1.00
