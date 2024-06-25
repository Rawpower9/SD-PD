from pyannote.audio.pipelines.clustering import BaseClustering

import random
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from einops import rearrange
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, Integer, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import oracle_segmentation
from pyannote.audio.utils.permutation import permutate


class RealTimeAgglomerativeClustering(BaseClustering):

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = np.inf,
        constrained_assignment: bool = False,
    ):
        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            constrained_assignment=constrained_assignment,
        )

        self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        self.method = Categorical(
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )
        # minimum cluster size
        self.min_cluster_size = Integer(1, 20)

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
        num_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
        centroids: Optional[np.array] = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension) array
            Centroid vectors of each cluster
        """
        self.train_embeddings, train_chunk_idx, train_speaker_idx = (
            self.filter_embeddings(
                embeddings,
                segmentations=segmentations,
            )
        )

        num_embeddings, _ = self.train_embeddings.shape


        self.num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        if max_clusters < 2:
            clusters = np.zeros((max_clusters,))
            clusters+=len(centroids)
            # do NOT apply clustering when min_clusters = max_clusters = 1
            # num_chunks, num_speakers, _ = embeddings.shape
            # hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            # soft_clusters = np.ones((num_chunks, num_speakers, 1))
            # centroids = np.mean(self.train_embeddings, axis=0, keepdims=True)
            # return hard_clusters, soft_clusters, centroids
        else :
            clusters = self.cluster(
                self.train_embeddings,
                min_clusters,
                max_clusters,
                num_clusters=self.num_clusters,
                large_centroids=centroids
            )

        if len(self.train_embeddings) != 0:
            self.train_clusters = self.assign(
                self.train_embeddings,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                num_clusters=self.num_clusters,
                large_centroids=centroids,
                clusters=clusters,
            )
        else:
            return (
                np.zeros((len(embeddings), 3)),
                np.zeros((len(embeddings), 3)),
                np.zeros((len(embeddings), 3)),
            )


        hard_clusters, soft_clusters, Ccentroids = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            self.train_clusters,
            constrained=self.constrained_assignment,
            important_centroids=centroids
        )

        return hard_clusters, soft_clusters, Ccentroids

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: Optional[int] = None,
        large_centroids: np.array = None,
    ):
        """

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings
        min_clusters : int
            Minimum number of clusters
        max_clusters : int
            Maximum number of clusters
        num_clusters : int, optional
            Actual number of clusters. Default behavior is to estimate it based
            on values provided for `min_clusters`,  `max_clusters`, and `threshold`.

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """
        num_embeddings, _ = embeddings.shape

        # heuristic to reduce self.min_cluster_size when num_embeddings is very small
        # (0.1 value is kind of arbitrary, though)
        # linkage function will complain when there is just one embedding to cluster
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        # centroid, median, and Ward method only support "euclidean" metric
        # therefore we unit-normalize embeddings to somehow make them "euclidean"
        if self.metric == "cosine" and self.method in ["centroid", "median", "ward"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric="euclidean"
            )

        # other methods work just fine with any metric
        else:
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric=self.metric
            )
        self.dendrogram = dendrogram
        # apply the predefined threshold
        clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        clusters += len(large_centroids)
        return clusters

    def assign(self,
            embeddings: np.ndarray,
            min_clusters: int,
            max_clusters: int,
            num_clusters: Optional[int] = None,
            large_centroids: np.array = None,
            clusters: np.array=None,
        ):
        num_embeddings, _ = embeddings.shape
        min_cluster_size = min(self.min_cluster_size, max(1, round(0.1 * num_embeddings)))
        self.clusters = clusters

        cluster_unique, cluster_counts = np.unique(
            clusters,
            return_counts=True,
        )
        self.cluster_unique = cluster_unique
        self.cluster_counts = cluster_counts
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        num_large_clusters = len(large_clusters)
        self.large_clusters = large_clusters
        self.num_large_clusters = num_large_clusters

        self.num_clusters = num_clusters
        if len(embeddings) == 0:
            small_centroids = np.empty((0,256),dtype=np.float32)
        else:
            small_centroids = np.vstack(
                [np.mean(embeddings[clusters == k], axis=0) for k in cluster_unique]
            )

        centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        asmall_centroid_shape = small_centroids.shape
        for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
            m = np.min(centroids_cdist[large_k])
            if m < self.threshold:
                clusters[clusters == cluster_unique[small_k]] = large_k
            # else:
            #     clusters[clusters == cluster_unique[small_k]] = len(large_centroids)

        # re-number clusters from 0 to num_large_clusters
        a = np.arange(0, len(large_centroids), 1)
        c = np.concatenate((clusters, a))
        _, clusters = np.unique(c, return_inverse=True)
        clusters = clusters[: -len(large_centroids)]
        return clusters

    def assign_embeddings(
        self,
        embeddings: np.ndarray,
        train_chunk_idx: np.ndarray,
        train_speaker_idx: np.ndarray,
        train_clusters: np.ndarray,
        constrained: bool = False,
        important_centroids:np.ndarray=None,
    ):
        """Assign embeddings to the closest centroid

        Cluster centroids are computed as the average of the train embeddings
        previously assigned to them.

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension)-shaped array
            Complete set of embeddings.
        train_chunk_idx : (num_embeddings,)-shaped array
        train_speaker_idx : (num_embeddings,)-shaped array
            Indices of subset of embeddings used for "training".
        train_clusters : (num_embedding,)-shaped array
            Clusters of the above subset
        constrained : bool, optional
            Use constrained_argmax, instead of (default) argmax.

        Returns
        -------
        soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
        hard_clusters : (num_chunks, num_speakers)-shaped array
        centroids : (num_clusters, dimension)-shaped array
            Clusters centroids
        """

        # TODO: option to add a new (dummy) cluster in case num_clusters < max(frame_speaker_count)

        num_clusters = np.max(train_clusters) + 1
        num_chunks, num_speakers, dimension = embeddings.shape

        train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]

        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        centroids[:len(important_centroids)] = important_centroids

        # compute distance between embeddings and clusters
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        soft_clusters = 2 - e2k_distance

        # assign each embedding to the cluster with the most similar centroid
        if constrained:
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        # NOTE: train_embeddings might be reassigned to a different cluster
        # in the process. based on experiments, this seems to lead to better
        # results than sticking to the original assignment.

        return hard_clusters, soft_clusters, centroids
