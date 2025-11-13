"""
Stratified Active Learning Sampling Module

This module implements uncertainty quantification and stratified sampling methods
for active learning in multi-label classification tasks.

Key Features:
    - Multiple uncertainty measures (binary entropy, ratio max, clustering)
    - Stratified sampling with optional weighting
    - Hierarchical agglomerative clustering
    - Support for removal and accumulation strategies
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist


class UncertaintyQuantification:
    """
    A class for uncertainty quantification and stratified sampling in active learning.

    This class provides methods for computing uncertainty scores and selecting
    samples using various strategies including stratified sampling across predefined
    strata (e.g., spatial, temporal, or species-based).

    Attributes:
        samples_num (int): Number of samples to select in each sampling round
        clusters (dict): Cached cluster assignments for cluster-based sampling
        x (torch.Tensor): Feature embeddings of shape (N, D)
        y (torch.Tensor): Labels of shape (N, C) for multi-label classification
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, samples_num: int):
        self.samples_num = samples_num
        self.clusters = None
        self.x = x
        self.y = y

    def binaryEntropy(self, outputs: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """
        Compute the maximum binary entropy across all classes for multi-label classification.

        For each sample, computes binary entropy for each class and returns the maximum,
        which represents the most uncertain prediction for that sample.

        Args:
            outputs (torch.Tensor): Model predictions (probabilities) of shape (N, C)
            eps (float): Small constant for numerical stability

        Returns:
            torch.Tensor: Maximum entropy per sample, shape (N,)
        """
        entropy = -(outputs * torch.log2(outputs + eps) + (1 - outputs) * torch.log2(1 - outputs + eps))
        entropy = torch.nan_to_num(entropy) 
        per_sample_entropy = torch.max(entropy, axis=1)[0] 
        return per_sample_entropy
        
    def ratioMax(self, outputs: torch.Tensor):
        """
        Compute ratio-based uncertainty score for multi-label classification.

        This method quantifies uncertainty based on the distance from the decision
        boundary (0.5). Predictions closer to 0.5 have higher uncertainty.

        Args:
            outputs (torch.Tensor): Model predictions (probabilities) of shape (N, C)

        Returns:
            torch.Tensor: Maximum uncertainty per sample, shape (N,)
        """
        uncertainty_scores = (0.5 - torch.abs(outputs - 0.5)) / (0.5 + torch.abs(outputs - 0.5))
        max_uncertainty = torch.max(uncertainty_scores, axis=1)[0]
        return max_uncertainty
    
    def clusterEntropy(self, confidence_scores: torch.Tensor, subsample: float = 0.2) -> list:
        """
        Perform cluster-based sampling using hierarchical agglomerative clustering.

        This method:
        1. Clusters a subset of embeddings using hierarchical clustering
        2. Assigns all samples to nearest cluster centroids
        3. Selects the most uncertain sample from each cluster

        Args:
            confidence_scores (torch.Tensor): Model confidence scores for all samples
            subsample (float): Fraction of samples to use for initial clustering (0.0-1.0)

        Returns:
            list: Indices of selected samples (one per cluster)
        """

        if self.clusters is None:
            print("Generating clusters... ", end="", flush=True)
            subset = self.x[np.random.choice(self.x.shape[0], int(subsample*self.x.shape[0]), replace=False)]
            agglomerativeClustering = AgglomerativeClustering(n_clusters=self.samples_num, distance_threshold=None, linkage='ward')
            clusters_indices = agglomerativeClustering.fit_predict(subset)

            # Compute cluster centroids
            unique_clusters = np.unique(clusters_indices)
            cluster_centroids = np.array([subset[clusters_indices == c].mean(axis=0) for c in unique_clusters])

            # Assign to clusters by comparing distance to cluster centroids
            distances = cdist(self.x, cluster_centroids)
            assigned_clusters = np.argmin(distances, axis=1)

            self.clusters = {}
            for cluster_idx in range(max(clusters_indices)+1):
                indices = np.where(assigned_clusters == cluster_idx)[0]
                self.clusters[cluster_idx] = indices
            print("Done")

        clusters_scores = {}
        for cluster_idx, cluster_indices in self.clusters.items():
            scores = self.binaryEntropy(confidence_scores[cluster_indices])
            clusters_scores[cluster_idx] = scores

        selected = []
        for cluster_idx, scores in clusters_scores.items():
            idx = torch.argmax(scores).item()
            selected.append(self.clusters[cluster_idx][idx])
            clusters_scores[cluster_idx][idx] = 0

            if len(selected) >= self.samples_num:
                break
        return selected
    
    def stratified(self, model, sorted_indices: dict, method="binary", indices=None, weights=None):
        """
        Perform stratified sampling according to the strata given in the input.

        Args:
            model (nn.Module): Model to be used for computing the uncertainty.
            sorted_indices (dict): Dictionary where the keys are the strata and the values are the sorted indices of the embeddings in that strata.
            method (str): Method to be used to compute the uncertainty. Defaults to "binary".
            indices (list): List of indices of the embeddings to be used for sampling. Defaults to None.

        Returns:
            dict: Dictionary with the selected indices for each strata.
        """

        sorted_embeddings = {key: self.x[idx] for key, idx in sorted_indices.items()}
        strata_idx = {}
        # print("sorted_embeddings", sorted_embeddings)

        # Compute per stratum sample count
        n_samples = self.samples_num // len(sorted_embeddings.keys())
        if self.samples_num < len(sorted_embeddings.keys()):
            raise Exception("The number of strata exceeds the number of total samples, increase n_samples.")
        # print("n_samples", n_samples)

        # print(weights)
        if weights is not None:
            weighted_samples = torch.multiply(weights, self.samples_num)
            

        for j, (key, v) in enumerate(sorted_embeddings.items()):
            if weights is not None:
                n_samples = torch.ceil(weighted_samples[j])

            if indices:
                selected = [i for i, val in enumerate(sorted_indices[key]) if val in indices]
            else:
                selected = None
            # print("selected", selected)
        
            idx = self.resample(model, embeddings=v, method=method, indices=selected, override=int(n_samples)) # Provides local indices (per strata)
            mapped = list(np.array(sorted_indices[key])[idx])
            strata_idx[key] = mapped
        return strata_idx

    def resample(self, model, method="random", embeddings=None, indices=None, removal=True, override:int=None):
        """
        Resample the embeddings according to the given method.

        Args:
            model (nn.Module): Model to be used for computing the uncertainty.
            method (str): Method to be used to compute the uncertainty. Defaults to "binary".
            embeddings (tensor): Tensor containing embeddings to be resampled. Defaults to None.
            indices (list): List of indices of the embeddings to be used for sampling. Defaults to None.
            removal (bool): If True, the sampled embeddings will be removed from the original embeddings. Defaults to True.
            override (int): Override the number of samples. Defaults to None.

        Returns:
            list: List of indices of the resampled embeddings.
        """
        
        samples_num = self.samples_num if override == None else override
        x = self.x if embeddings == None else embeddings
        
        model.eval()
        outputs = model(x)
        confidence_scores = F.sigmoid(outputs)

        # Compute uncertainty
        if method == "binary":
            scores = self.binaryEntropy(confidence_scores)
        elif method == "ratio_max":
            scores = self.ratioMax(confidence_scores)
        elif method == "confidence":
            scores = torch.max(confidence_scores, dim=1).values
        elif method == "random":
            idx = np.random.choice(len(x), size=samples_num, replace=False)
            return idx
        elif method == "cluster":
            idx = self.clusterEntropy(confidence_scores)
            return idx
        else:
            raise Exception("Unknown uncertainty quantification method")
        
        # Return top k uncertain samples
        if indices:
            scores[indices] = -float('inf')

        if len(scores) > samples_num:
            _, idx = torch.topk(scores, k=samples_num)
        else: 
            idx = []
            raise Warning("insufficient samples left in strata")
        
        return idx.tolist()
    