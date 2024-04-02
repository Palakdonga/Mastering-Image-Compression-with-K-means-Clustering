# Mastering-Image-Compression-with-K-means-Clustering
Introduction:
In the vast realm of machine learning and data science, the K-means algorithm stands as a stalwart technique for clustering data into distinct groups. However, its applications extend beyond merely segmenting data; one fascinating use case is image compression. In this tutorial, we'll delve into the intricacies of K-means clustering, starting from its implementation to its application in compressing images.

1. Implementing K-means

1.1 Finding Closest Centroids

In this section, we'll explore how to find the closest centroids to each data point.

import numpy as np

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distances)
    return idx
1.2 Computing Centroid Means

Next, let's dive into computing centroid means based on the data points assigned to each centroid.

def compute_centroids(X, idx, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        centroids[k, :] = np.mean(X[idx == k, :], axis=0)
    return centroids
2. K-means on a Sample Dataset

Now, let's apply the K-means algorithm to a sample dataset to understand its functioning and its ability to identify clusters.

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()
3. Random Initialization

Before proceeding, let's discuss the critical aspect of random initialization in K-means and its impact on the final clustering.

4. Image Compression with K-means

4.1 Dataset

We'll introduce the concept of image compression and the dataset we'll be working with.

4.2 K-Means on Image Pixels

Applying K-means directly to image pixels for color quantization.

from sklearn.cluster import KMeans
from PIL import Image

# Load image
image = Image.open('example_image.jpg')
image_data = np.array(image) / 255.0  # Normalize pixel values

# Reshape image data
image_reshaped = image_data.reshape(-1, 3)

# Apply K-means
kmeans = KMeans(n_clusters=16, random_state=0).fit(image_reshaped)
compressed_colors = kmeans.cluster_centers_[kmeans.labels_]

# Reshape compressed image data
compressed_image_data = compressed_colors.reshape(image_data.shape)
compressed_image = Image.fromarray((compressed_image_data * 255).astype(np.uint8))
compressed_image.show()
4.3 Compress the Image

Finally, we'll demonstrate how K-means clustering can effectively compress images by reducing the number of colors while preserving the image's visual integrity.

Conclusion:
The journey through K-means clustering for image compression has been both enlightening and practical. By understanding the algorithm's inner workings and its application to image data, we've unlocked a powerful tool for reducing the storage requirements of images without compromising their quality. As you embark on your own exploration, remember the versatility of K-means and its potential to revolutionize various domains beyond traditional data analysis. Happy clustering!

