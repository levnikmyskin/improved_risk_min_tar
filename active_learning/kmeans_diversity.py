import numpy as np
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist


class DiversityWithKMeans:
    def __init__(self, total_per_clust: int, centroids_per_clust: int, outliers_per_clust: int, rand_per_clust: int,
                 perc_for_outliers: int, batch_size: int, seed=42):
        self.total_per_clust = total_per_clust
        self.centroids_per_clust = centroids_per_clust
        self.outliers_per_clust = outliers_per_clust
        self.rand_per_clust = rand_per_clust
        self.perc_for_outliers = perc_for_outliers
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed=seed)

    def get_data_to_annotate(self, x, mask, test_idxs):
        # see https://medium.datadriveninvestor.com/outlier-detection-with-k-means-clustering-in-python-ee3ac1826fb0
        kmeans = MiniBatchKMeans(math.ceil(self.batch_size / self.total_per_clust))
        clusters = kmeans.fit_predict(x)
        centroids = kmeans.cluster_centers_
        mask[test_idxs] = clusters
        closest = []
        randoms = []
        outliers = []
        for i, center_elem in enumerate(centroids):
            distance = cdist([center_elem], x[clusters == i].toarray(), 'cosine').squeeze()
            point = np.where(mask == i)[0]
            closest.append(point[np.argsort(distance)][:self.centroids_per_clust])
            outliers.append(self.rng.choice(point[np.where(distance > np.percentile(distance, self.perc_for_outliers))],
                                            self.outliers_per_clust, replace=False))
            rand_choice = list(set(point) - set(closest[i]) - set(outliers[i]))
            randoms.append(self.rng.choice(rand_choice, self.rand_per_clust, replace=False))

        return np.concatenate((np.array(closest).flatten(), np.array(outliers).flatten(), np.array(randoms).flatten()))
