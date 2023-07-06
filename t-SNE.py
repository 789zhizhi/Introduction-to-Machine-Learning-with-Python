# t-SNE for visualization
# dataste is 0-9 digits
from re import T
from unicodedata import digit
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
digits = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':(), 'yticks': ()}) 
# don't embody the coordinate
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
plt.show()
# the code above shows the digit-pictures from 0 to 9
# Let’s use PCA to visualize the data reduced to two dimensions. 
# We plot the first two principal components, and color each dot by its class
pca = PCA(n_components=2) # two dimension
pca.fit(digits.data)
# transform the digits data onto the first two principal components
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
 "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max()) # limit
for i in range(len(digits_pca.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
    color = colors[digits.target[i]],
    fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()
# Let’s apply t-SNE to the same dataset, and compare the results.
# As t-SNE does not support transforming new data, the TSNE class has no transform method.
# Instead, we can call the fit_transform method, which will build the model and immediately 
# return the transformed data 
from sklearn.manifold import TSNE
tsne = TSNE(random_state=43)
digits_tsne = tsne.fit_transform(digits.data)
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max()+1)
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max()+1)
for i in range(len(digits_pca.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
    color = colors[digits.target[i]],
    fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
plt.show()