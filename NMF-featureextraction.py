# feature extraction with nmf
import mglearn
import matplotlib.pyplot as plt
import featureExtraction
mglearn.plots.plot_nmf_illustration()
plt.show()

from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(featureExtraction.X_train)
X_train_nmf = nmf.transform(featureExtraction.X_train)
X_test_nmf = nmf.transform(featureExtraction.X_test)
fix, axes = plt.subplots(3, 5, figsize=(15, 12),
 subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(featureExtraction.image_shape))
    ax.set_title("{}. component".format(i))
plt.show()
# return component=3/7 face-photo
import numpy as np
compn = 3
inds = np.argsort(X_train_nmf[:,compn])[::-1] #使用[::-1],可以建立X从大到小的索引。
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
 subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(featureExtraction.X_train[ind].reshape(featureExtraction.image_shape))
compn = 7
# sort by 7th component, plot first 10 images
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
 subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(featureExtraction.X_train[ind].reshape(featureExtraction.image_shape))
plt.show()

S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel("Time")
plt.ylabel("Signal") # three sources
plt.show()
# mix data into a 100-dimensional state
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("Shape of measurements: {}".format(X.shape))
#recover with nmf
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("Recovered signal shape: {}".format(S_.shape))
# rfecover with pca
from sklearn.decomposition import   PCA
pca = PCA(n_components=3) 
H = pca.fit_transform(X)
models = [X, S, S_, H]
names = ['Observations (first three measurements)',
 'True sources',
 'NMF recovered signals',
 'PCA recovered signals']
fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
 subplot_kw={'xticks': (), 'yticks': ()})
for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
plt.show()
