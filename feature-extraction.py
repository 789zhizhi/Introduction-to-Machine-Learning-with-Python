# We will give a very simple application of feature extraction on images using PCA, by
# working with face images from the Labeled Faces in the Wild dataset
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
#people = fetch_lfw_people(min_faces_per_person=20, resize=0.7
people = fetch_lfw_people(data_home = "/home/cmk/work/test/ML-with-Pythom/dataset",download_if_missing=False)

image_shape = people.images[0].shape
fix, axes = plt.subplots(2, 5, figsize=(15, 8),
 subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
#plt.show()
print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))
print("people.target.shape: {}".format(people.target.shape))
print("people.data.shape: {}".format(people.data.shape))
# print(people.data.head(5))
# # count how often each target appears
# counts = np.bincount(people.target)
# # print counts next to target names
# for i, (count, name) in enumerate(zip(counts, people.target_names)):
#     print("{0:25} {1:3}".format(name, count), end=' ')
#     if (i + 1) % 3 == 0:
#         print()
#class 5749
#print(people.data[2])
mask = np.zeros(people.target.shape, dtype=np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255.
# face recognition 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# split the data into training and test sets
print(X_people.shape,y_people[:3])
X_train, X_test, y_train, y_test = train_test_split(
 X_people, y_people, stratify=None, random_state=0)
# build a KNeighborsClassifier using one neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test))) # 0.04
# accuracy is bad
# use pca get feature from original data
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("X_train_pca.shape: {}".format(X_train_pca.shape))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("Test set accuracy: {:.2f}".format(knn.score(X_test_pca, y_test))) # 0.06
