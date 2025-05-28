import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering


class UncertaintyQuantification:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, samples_num: int):
        self.samples_num = samples_num
        self.clusters = None
        self.x = x
        self.y = y

    def binaryEntropy(self, outputs: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """
        Compute the maximum binary entropy across all classes
        """
        entropy = -(outputs * torch.log2(outputs + eps) + (1 - outputs) * torch.log2(1 - outputs + eps))
        entropy = torch.nan_to_num(entropy) 
        per_sample_entropy = torch.max(entropy, axis=1)[0] 
        return per_sample_entropy
        
    def ratioMax(self, outputs:torch.tensor):
        uncertainty_scores = (0.5 - torch.abs(outputs - 0.5)) / (0.5 + torch.abs(outputs - 0.5))
        max_uncertainty = torch.max(uncertainty_scores, axis=1)[0]
        return max_uncertainty
    
    def clusterEntropy(self, confidence_scores:torch.tensor) -> list:
        """
        Compute the Hierachical Agglomerative Clustering for the training embeddings.

        Args:
            confidence (tensor): Tensor containing confidence scores of all embeddings.
        Returns:
            list: List of selected indices.
        """

        if self.clusters is None:
            agglomerativeClustering = AgglomerativeClustering(n_clusters=self.samples_num, distance_threshold=None, linkage='ward')
            self.clusters = agglomerativeClustering.fit_predict(self.x)

        clusters_scores = {}
        clusters_indices = {}
        for cluster_idx in range(max(self.clusters)+1):
            indices = np.where(self.clusters == cluster_idx)[0]
            scores = self.binaryEntropy(confidence_scores[indices])
            clusters_scores[cluster_idx] = scores
            clusters_indices[cluster_idx] = list(indices)

        selected = []
        for cluster_idx, scores in clusters_scores.items():
            if len(clusters_indices[cluster_idx]) > 0:
                idx = torch.argmax(scores).item()
                selected.append(clusters_indices[cluster_idx][idx])
                clusters_scores[cluster_idx][idx] = 0

            if len(selected) >= self.samples_num:
                break
        return selected
    
    def stratified(self, model, sorted_indices: dict, method="binary", indices=None):
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
        for key, v in sorted_embeddings.items():

            if indices:
                selected = [i for i, val in enumerate(sorted_indices[key]) if val in indices]
            else:
                selected = None
        
            idx = self.resample(model, embeddings=v, method=method, indices=selected) # Provides local indices (per strata)
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
            scores[indices] = 0

        if len(scores) > samples_num:
            _, idx = torch.topk(scores, k=samples_num)
        else: 
            idx = []
            raise Warning("insufficient samples left in strata")
        
        return idx.tolist()
    