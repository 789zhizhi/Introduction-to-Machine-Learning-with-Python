import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_lfw_people
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
