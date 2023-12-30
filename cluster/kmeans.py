import faiss
import numpy as np
import os


class Kmeans(object):
    def __init__(self, nmb_clusters, nredo=20, dump_path=''):
        """
        Kmeans clustering.

        Args:
        nmb_clusters (int): The numbers of clusters.
        nredo (int): The number os trials in clustering.
        """
        self.nmb_clusters = nmb_clusters
        self.nredo = nredo
        self.dump_path = dump_path

    def cluster(self, npdata1, npdata2=None, use_pca=False, save_centroids=False):

        # PCA-reducing, whitening and L2-normalization
        npdata1, npdata2 = self.preprocess_features(npdata1, npdata2, use_pca=use_pca)

        # Clustering the data
        labels, loss, centroids = self.run_kmeans(
            npdata1=npdata1,
            npdata2=npdata2,
            nmb_clusters=self.nmb_clusters,
            nredo=self.nredo,
            save_centroids=save_centroids,
            dump_path=self.dump_path,
        )
        if save_centroids:
            self.centroids = faiss.vector_to_array(centroids)

        self.labels = labels
        return loss

    def preprocess_features(self, npdata1, npdata2, pca_dim=128, use_pca=False):
        _, ndim = npdata1.shape
        npdata1 = npdata1.astype("float32")

        # Apply PCA-whitening with Faiss
        if use_pca:
            mat = faiss.PCAMatrix(ndim, pca_dim, eigen_power=-0.5)
            mat.train(npdata1)
            assert mat.is_trained
            npdata1 = mat.apply_py(npdata1)
            if npdata2 is not None:
                npdata2 = mat.apply_py(npdata2)

        # L2 normalization
        row_sums = np.linalg.norm(npdata1, axis=1)
        npdata1 = npdata1 / (row_sums[:, np.newaxis] + 1e-8)
        if npdata2 is not None:
            row2_sums = np.linalg.norm(npdata2, axis=1)
            npdata2 = npdata2 / (row2_sums[:, np.newaxis] + 1e-8)

        return npdata1, npdata2

    def run_kmeans(
            self,
            npdata1,
            npdata2=None,
            nmb_clusters=50,
            nredo=20,
            save_centroids=False,
            dump_path='',
    ):
        d = npdata1.shape[-1]

        # faiss implementation of k-means
        clus = faiss.Clustering(d, nmb_clusters)

        # Change faiss seed at each k-means so that the randomly picked
        # initialization centroids do not correspond to the same feature ids
        # from an epoch to another.
        clus.seed = np.random.randint(1234)

        clus.nredo = nredo
        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
        # perform the training
        clus.train(npdata1, index)

        ##记录聚类id和距离
        D_t, I_t = index.search(npdata1, 50)
        labeled_D = []
        labeled_I = []
        for i in range(len(D_t)):
            arrD = str(D_t[i][0])
            arrI = str(I_t[i][0])
            for j in range(1, 50):
                arrD = arrD + " " + str(D_t[i][j])
                arrI = arrI + " " + str(I_t[i][j])
            labeled_D.append(arrD)
            labeled_I.append(arrI)
        # with open(os.path.join(dump_path,
        #                        "train_D_50.txt"), "a") as f:
        #     f.write("\n".join(labeled_D))
        # with open(os.path.join(dump_path,
        #                        "train_I_50.txt"), "a") as f:
        #     f.write("\n".join(labeled_I))

        labeled_D = []
        labeled_I = []
        for i in range(len(D_t)):
            arrD = str(D_t[i][0])
            arrI = str(I_t[i][0])
            for j in range(1, 3):
                arrD = arrD + " " + str(D_t[i][j])
                arrI = arrI + " " + str(I_t[i][j])
            labeled_D.append(arrD)
            labeled_I.append(arrI)
        # with open(os.path.join(dump_path,
        #                        "train_D_3.txt"), "a") as f:
        #     f.write("\n".join(labeled_D))
        # with open(os.path.join(dump_path,
        #                        "train_I_3.txt"), "a") as f:
        #     f.write("\n".join(labeled_I))


        D_t2, I_t2 = index.search(npdata2, 50)
        labeled_D2 = []
        labeled_I2 = []
        for i in range(len(D_t2)):
            arrD = str(D_t2[i][0])
            arrI = str(I_t2[i][0])
            for j in range(1, 50):
                arrD = arrD + " " + str(D_t2[i][j])
                arrI = arrI + " " + str(I_t2[i][j])
            labeled_D2.append(arrD)
            labeled_I2.append(arrI)
        with open(os.path.join(dump_path,
                               "train_D_50.txt"), "a") as f:
            f.write("\n".join(labeled_D2))
        with open(os.path.join(dump_path,
                               "train_I_50.txt"), "a") as f:
            f.write("\n".join(labeled_I2))
        labeled_D2 = []
        labeled_I2 = []
        for i in range(len(D_t2)):
            arrD = str(D_t2[i][0])
            arrI = str(I_t2[i][0])
            for j in range(1, 3):
                arrD = arrD + " " + str(D_t2[i][j])
                arrI = arrI + " " + str(I_t2[i][j])
            labeled_D2.append(arrD)
            labeled_I2.append(arrI)
        with open(os.path.join(dump_path,
                               "train_D_3.txt"), "a") as f:
            f.write("\n".join(labeled_D2))
        with open(os.path.join(dump_path,
                               "train_I_3.txt"), "a") as f:
            f.write("\n".join(labeled_I2))
        #

        # find the classes
        if npdata2 is not None:
            npdata1 = np.concatenate((npdata1, npdata2), axis=0)
        D, I = index.search(npdata1, 1)

        stats = clus.iteration_stats
        losses = np.array([stats.at(i).obj for i in range(stats.size())])

        if save_centroids:
            centroids = clus.centroids
            return [int(n[0]) for n in I], losses[-1], centroids
        else:
            return [int(n[0]) for n in I], losses[-1], None
