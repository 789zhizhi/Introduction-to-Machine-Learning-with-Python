import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
mglearn.plots.plot_linear_regression_wave()
##############dataset
X,y = mglearn.datasets.make_wave(n_samples = 60)
X_train ,X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
#w,b parajmeters
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("train set score:{:.2f}".format(lr.score(X_train, y_train)))
print("test set score:{:.2f}".format(lr.score(X_test, y_test)))
#sample dataset with linerregression performace not well, underfit
#using more complex dateset
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
#overfit!!!!!!!!!!!!!1
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#Ridge regression uses regularization to reduce overfit
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
###############
#Lasso is a kind of liner-regression with regularization
#If we set alpha too low, however, we again remove the effect of regularization and end
#up overfitting, with a result similar to LinearRegression:
from sklearn.linear_model import Lasso
import numpy as np
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

