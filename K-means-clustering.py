from cgi import print_form
from posixpath import split
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np

#dataset_people_faces
people = fetch_lfw_people(data_home = "/home/cmk/work/test/ML-with-Pythom/dataset",download_if_missing=False)
mask = np.zeros(people.target.shape, dtype=np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255

mglearn.plots.plot_kmeans_algorithm()
plt.show()
# boundary 
mglearn.plots.plot_kmeans_boundaries()
plt.show()

X, y = make_blobs(random_state=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("cluster membership :{}\n".format(kmeans.labels_))
# we can also see more or fewer cluster centers
fig, axes =plt.subplots(1, 2, figsize=(10, 5))
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
plt.show()

# kmeans on face recognification coimparing with PCA and NMF
X_train, X_test, y_train, y_test = train_test_split(
 X_people, y_people, stratify=None, random_state=0)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))# 将降维后的数据转换成原始数据
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)] # 找出聚类中心
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)# 矩阵乘法运算 ,components:H_matrix
# get reconstructed_feature of X on the above
# ============comparison============
image_shape = people.images[0].shape
fig, axes = plt.subplots(3, 5, figsize=(8, 8),
 subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Extracted Components")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(
 axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
    ax[0].imshow(comp_kmeans.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
    ax[2].imshow(comp_nmf.reshape(image_shape))
axes[0, 0].set_ylabel("kmeans")
axes[1, 0].set_ylabel("pca")
axes[2, 0].set_ylabel("nmf")
print("===========above pac-kmeans-nmf-face-extraction=======")
plt.show()
fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()},
 figsize=(8, 8))
fig.suptitle("Reconstructions")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
 axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca,
 X_reconstructed_nmf):
    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_kmeans.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape))
    ax[3].imshow(rec_nmf.reshape(image_shape))
axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")
print("===============reconstracted comprison between pac/kmeans/nmf/orignal")
plt.show()
# An interesting aspect of vector quantization using k-means is that we can use many
# more clusters than input dimensions to encode our data. Let’s go back to the
# two_moons data. Using PCA or NMF, there is nothing much we can do to this data, as
# it lives in only two dimensions. Reducing it to one dimension with PCA or NMF
# would completely destroy the structure of the data. But we can find a more expressive
# representation with k-means, by using more cluster centers
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60,
 marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
print("Cluster memberships:\n{}".format(y_pred))
plt.show()