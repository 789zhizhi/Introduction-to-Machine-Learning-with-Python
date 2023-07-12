from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

X, y = make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()
# DBSCAN does not allow predictions on
# new test data, so we will use the fit_predict method to perform clustering and
# return the cluster labels in one step:
clusters = dbscan.fit_predict(X)
print("Cluster memberships:\n{}".format(clusters))
# -1, which stands for noise
# visualization
mglearn.plots.plot_dbscan()
plt.show()
# 附近(eps)点的数量 ≥min_samples，则当前点与其附近点形成一个簇
#  Finding a good setting for eps is sometimes easier
#  after scaling the data using StandardScaler or MinMaxScaler,
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#=================comparison between kmeans/agglomerative/dbscan
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
fig, axes = plt.subplots(1, 4, figsize=(15, 3),
 subplot_kw={'xticks': (), 'yticks': ()})
# make a list of algorithms to use
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
 DBSCAN()]
# create a random cluster assignment for reference
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
# plot random assignment
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
 cmap=mglearn.cm3, s=60)
axes[0].set_title("Random assignment - ARI: {:.2f}".format(
 adjusted_rand_score(y, random_clusters)))
for ax, algorithm in zip(axes[1:], algorithms):
    # plot the cluster assignments and cluster centers
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
    cmap=mglearn.cm3, s=60)
    ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,
    adjusted_rand_score(y, clusters)))
plt.show()

# The results for eps=7 look most interesting, with many small clusters.on the face dataset
people = fetch_lfw_people(data_home = "/home/cmk/work/test/ML-with-Pythom/dataset",download_if_missing=False)
mask = np.zeros(people.target.shape, dtype=np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255
image_shape = people.images[0].shape
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)   # propercessing data X_people

dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)   #clusters
print("clusters = {}\n".format(labels))
# output is clusters = [-1 -1 -1 ... -1 -1  0] . -1 represents noise
# # visualize the face of eps=7
# for cluster in range(max(labels) + 1):
#     mask = labels == cluster  # mask = ture or false
#     # print("mask= {}\n".format(mask))
#     # print("cluster= {}\n".format(cluster))
#     # print("X_people[mask] = {}\n".format(X_people[mask]))
#     n_images = np.sum(mask) #the number of images
#     fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
#     subplot_kw={'xticks': (), 'yticks': ()})
#     for image, label, ax in zip(X_people[mask], y_people[mask], axes):
#         ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
#         ax.set_title(people.target_names[label].split()[-1])
# # plt.show() #picture is bigger than 2^16,so can't show the picture
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people,
 y_people, people.target_names)
plt.show()