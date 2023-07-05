# PCA :Principal Component Analysis can do Dimensionality Reduction
# PCA(Principal Components Analysis)即主成分分析，也称主分量分析或主成分回归分析法，
# 是一种无监督的数据降维方法。首先利用线性变换，将数据变换到一个新的坐标系统中；
# 然后再利用降维的思想，使得任何数据投影的第一大方差在第一个坐标(称为第一主成分)上，
# 第二大方差在第二个坐标(第二主成分)上。这种降维的思想首先减少数据集的维数，
# 同时还保持数据集的对方差贡献最大的特征，最终使数据直观呈现在二维坐标系

from re import S
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np
# two-demosional dataset as example
mglearn.plots.plot_pca_illustration()
plt.show()

cancer = load_breast_cancer()
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
#print(cancer.target[:6])
print(cancer.data.shape)   #569*30
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]   # binary classfication
ax = axes.ravel()
for i in range(30): # dataset consists of 30 features
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc="best")
fig.tight_layout()
#plt.show()
# However, this plot doesn’t show us anything about the interactions between variables
# and how these relate to the classes. Using PCA, we can capture the main interactions
# and get a slightly more complete picture. We can find the first two principal components, 
# and visualize the data in this new two-dimensional space with a single scatterv plot.
# first we scale our data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
# transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

# plot first vs. second principal component, colored by class
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()
# Each row in components_ corresponds to one principal component, and they are sorted 
# by their importance (the first principal component comes first, etc.). The columns
# correspond to the original features attribute of the PCA in this example, “mean
# radius,” “mean texture,” and so on. Let’s have a look at the content of components_
# We can also visualize the coefficients using a heat map
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
 cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.show()
