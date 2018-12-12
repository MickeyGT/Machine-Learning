# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, y_train)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'rbf', random_state = 0)
classifierSVM.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, y_train)

# Predicting the Test set results
y_predLR = classifierLR.predict(X_test)
y_predKNN = classifierKNN.predict(X_test)
y_predSVM = classifierSVM.predict(X_test)
y_predRF = classifierRF.predict(X_test)

# Making the Confusion Matrixes
from sklearn.metrics import confusion_matrix
cmLR = confusion_matrix(y_test, y_predLR)
cmKNN = confusion_matrix(y_test, y_predKNN)
cmSVM = confusion_matrix(y_test, y_predSVM)
cmRF = confusion_matrix(y_test, y_predRF)

# Visualising the Training set results
X_set, y_set = X_train, y_train
# Visualising the Test set results
X_set, y_set = X_test, y_test


X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

from matplotlib.colors import ListedColormap
fig,axarr = plt.subplots(2, 2,sharex=True,sharey=True)
axarr[0, 0].contourf(X1, X2, classifierLR.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    axarr[0, 0].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
               c = ListedColormap(('blue', 'purple'))(i), label = j)
axarr[0, 0].set_title('Logistic Regression')
axarr[0, 1].contourf(X1, X2, classifierKNN.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    axarr[0, 1].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'purple'))(i), label = j)
axarr[0, 1].set_title('K Nearest Neighbours')
axarr[1, 0].contourf(X1, X2, classifierSVM.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    axarr[1, 0].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
               c = ListedColormap(('blue', 'purple'))(i), label = j)
axarr[1, 0].set_title('Kernel SVM')
axarr[1, 1].contourf(X1, X2, classifierRF.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    axarr[1, 1].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'purple'))(i), label = j)
axarr[1, 1].set_title('Random Forest')

fig.text(0.5, 0.04, 'Age', ha='center')
fig.text(0.04, 0.5, 'Estimated Salary', va='center', rotation='vertical')