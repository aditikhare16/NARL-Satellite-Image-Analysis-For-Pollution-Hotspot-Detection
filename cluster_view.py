import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Simulating sample data for demonstration
np.random.seed(42)
data_points = np.vstack([
    np.random.normal(loc=(10, 10), scale=2, size=(100, 2)),  # High pollution cluster
    np.random.normal(loc=(20, 20), scale=3, size=(100, 2)),  # Medium pollution cluster
    np.random.normal(loc=(30, 10), scale=4, size=(100, 2)),  # Low pollution cluster
])

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(data_points)
centroids = kmeans.cluster_centers_

# Visualizing clusters
plt.figure(figsize=(10, 7))
for cluster in np.unique(labels):
    plt.scatter(data_points[labels == cluster, 0], data_points[labels == cluster, 1],
                label=f'Cluster {cluster + 1}', alpha=0.6)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='X')

# Adding titles and labels
plt.title("K-Means Clustering of Pollution Data", fontsize=16)
plt.xlabel("Feature 1 (e.g., Latitude)", fontsize=12)
plt.ylabel("Feature 2 (e.g., Longitude)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save and display the graph
plt.savefig("C:/Users/ASUS/Desktop/NARL Project/processed_data/kmeans_pollution_clustering_demo.png")

plt.show()