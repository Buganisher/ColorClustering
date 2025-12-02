import numpy as np
import torch

class KmeansPlusPlus_torch():
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, verbose=False, random_state=None, distance_func=None):
        """
        GPU加速的K-means聚类算法，支持自定义距离函数
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = torch.tensor(tol, device=device)
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.distance_func = distance_func or self._euclidean_distance
        
        self.centroids = None
        self.labels = None
    
    def _euclidean_distance(self, X, centroids):
        """欧氏距离计算"""
        return torch.cdist(X, centroids, p=2)
    
    def _initialize_centroids(self, X):
        """使用K-means++方法初始化中心点"""
        n_samples = X.shape[0]
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # 随机选择第一个中心点
        centroids = X[torch.randint(0, n_samples, (1,), device=self.device)]
        
        for i in range(1, self.n_clusters):
            # 计算每个点到最近中心点的距离
            distances = torch.min(torch.stack([self.distance_func(X, c.unsqueeze(0)) for c in centroids]), dim=0)[0].squeeze()
            
            # 避免除以零
            if distances.sum() == 0:
                 probs = torch.ones_like(distances) / len(distances)
            else:
                 probs = distances / distances.sum()
            
            # 选择下一个中心点
            next_idx = torch.multinomial(probs, 1).item()
            centroids = torch.cat([centroids, X[next_idx:next_idx+1]], dim=0)
        
        return centroids
    
    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(dtype=torch.float32, device=self.device)
        
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            distances = torch.cat([self.distance_func(X, c.unsqueeze(0)) for c in self.centroids], dim=1)
            self.labels = torch.argmin(distances, dim=1)
            old_centroids = self.centroids.clone()
            
            # 更新中心点 (向量化操作替代循环可能更快，但这里为了保持逻辑清晰暂且如此)
            for i in range(self.n_clusters):
                mask = (self.labels == i)
                if mask.any():
                    self.centroids[i] = X[mask].mean(dim=0)
            
            centroids_shift = self.distance_func(old_centroids, self.centroids)
            # 对角线是旧点到新点的距离
            if centroids_shift.diag().max() < self.tol:
                break
        
        return self
    
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(dtype=torch.float32, device=self.device)
            
        distances = torch.cat([self.distance_func(X, c.unsqueeze(0)) for c in self.centroids], dim=1)
        return torch.argmin(distances, dim=1)
    
    def fit_predict(self, X):
        """
        训练模型并直接返回聚类标签 (返回 Tensor)
        """
        self.fit(X)
        # 保持在 GPU/Tensor 上，不转 numpy
        return self.labels, self.centroids

def get_cluster_statistics(labels, centroids):
    """
    Get cluster statistics (Pure Tensor Implementation)
    
    Parameters:
    labels: Tensor, shape (n_samples,)
    centroids: Tensor, shape (n_clusters, n_features)
    """
    if labels is None or centroids is None:
        raise ValueError("Labels and centroids cannot be None")
    
    # 使用 bincount 计算每个聚类的样本数
    # minlength 确保即使某些聚类没有点，也能返回 0
    counts = torch.bincount(labels, minlength=len(centroids))
    
    # 计算百分比
    hist = counts.float() / len(labels) * 100
    
    # 排序 (降序)
    sorted_indices = torch.argsort(hist, descending=True)
    
    return hist[sorted_indices], sorted_indices