from pyexpat.model import XML_CQUANT_OPT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import decisionTrees

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train)
print("accuracy on training set:{:.3f}".format(gbrt.score(X_train,y_train)))#1
print("accuracy on test set:{:.3f}".format(gbrt.score(X_test,y_test)))#0.958
#######overfitted
# To reduce overfit‐ting, we could either apply stronger pre-pruning 
# by limiting the maximum depth or lower the learning rate
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

###visualize the importance of feature
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
decisionTrees.plot_feature_importances_cancer(gbrt)
##########predicion probability
print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))
print("Predicted probabilities:\n{}".format(
 gbrt.predict_proba(X_test[:6])))
 