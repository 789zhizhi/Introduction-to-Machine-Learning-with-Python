import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
mglearn.plots.plot_agglomerative_algorithm()
plt.show()
#  agglomerative clustering cannot make predictions for new data points.
from sklearn.cluster import AgglomerativeClustering
X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
# shows an overlay of all the possible clusterings 
mglearn.plots.plot_agglomerative()
plt.show()
# it can't show the hierchical cluster more than two features, but 
# dendrogram can do it ,and it is included in scipy ,not sklearn
