# kmeans example
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, y = make_blobs(n_samples=300, centers=4, random_state=42)
k = KMeans(n_clusters=4, random_state=42)
k.fit(X)
print('centers', k.cluster_centers_)
