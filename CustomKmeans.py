import numpy as np
import torch

class KmeansPlusPlus_torch():
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, verbose=False, random_state=None, distance_func=None):
        """
        GPU加速的K-means聚类算法，支持自定义距离函数
        
        参数:
        n_clusters: 聚类数量
        max_iter: 最大迭代次数
        tol: 收敛阈值
        verbose: 是否打印详细信息
        random_state: 随机种子
        device: 'cuda'或'cpu'，指定计算设备
        distance_func: 自定义距离函数，输入为两个张量，返回距离矩阵
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = torch.tensor(tol,device = device)
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
        
        # 设置随机种子
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # 随机选择第一个中心点
        centroids = X[torch.randint(0, n_samples, (1,), device=self.device)]
        
        for i in range(1, self.n_clusters):
            # 计算每个点到最近中心点的距离
            distances = torch.min(torch.stack([self.distance_func(X, c.unsqueeze(0)) for c in centroids]), dim=0)[0].squeeze()
            
            # 根据距离作为权重，计算概率分布
            probs = distances / distances.sum()
            
            # 选择下一个中心点
            next_idx = torch.multinomial(probs, 1).item()
            centroids = torch.cat([centroids, X[next_idx:next_idx+1]], dim=0)
        
        return centroids
    
    def fit(self, X):
        """
        训练K-means模型
        
        参数:
        X: 输入数据，形状为(n_samples, n_features)，可以是numpy数组或torch张量
        """
        # 转换为PyTorch张量并移至指定设备
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # 初始化中心点
        self.centroids = self._initialize_centroids(X)
        
        # 迭代更新中心点
        for iteration in range(self.max_iter):
            # 计算每个点到所有中心点的距离
            distances = torch.cat([self.distance_func(X, c.unsqueeze(0)) for c in self.centroids], dim=1)
            
            # 分配标签：每个点分配给最近的中心点
            self.labels = torch.argmin(distances, dim=1)
            
            # 保存旧中心点
            old_centroids = self.centroids.clone()
            
            # 更新中心点：计算每个簇的新中心点
            for i in range(self.n_clusters):
                # 找到属于当前簇的所有点
                cluster_points = X[self.labels == i]
                
                # 如果簇不为空，则更新中心点
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(dim=0)
            
            # 检查收敛
            centroids_shift = self.distance_func(old_centroids, self.centroids)
            is_converged = centroids_shift.diag() < self.tol
            if is_converged.all():
                break
        
        return self
    
    def predict(self, X):
        """
        预测数据点的聚类标签
        
        参数:
        X: 输入数据，形状为(n_samples, n_features)
        
        返回:
        labels: 聚类标签，形状为(n_samples,)
        """
        if self.centroids is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 转换为PyTorch张量并移至指定设备
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # 计算每个点到所有中心点的距离
        distances = torch.cat([self.distance_func(X, c.unsqueeze(0)) for c in self.centroids], dim=1)
        
        # 分配标签：每个点分配给最近的中心点
        return torch.argmin(distances, dim=1)
    
    def fit_predict(self, X):
        """
        训练模型并直接返回聚类标签
        
        参数:
        X: 输入数据，形状为(n_samples, n_features)
        
        返回:
        labels: 聚类标签，形状为(n_samples,)
        """
        self.fit(X)
        return self.labels.cpu().numpy(), self.centroids.cpu().numpy()

def get_cluster_statistics(labels, centroids):
    """
    Get cluster statistics
    
    Parameters:
    labels: Cluster labels, shape (n_samples,)
    centroids: Cluster centers, shape (n_clusters, n_features)
    
    Returns:
    hist: Percentage of samples in each cluster
    counts: Number of samples in each cluster
    centroids: Cluster center coordinates
    """
    if labels is None or centroids is None:
        raise ValueError("Labels and centroids cannot be None")
        
    n_clusters = len(centroids)
    counts = np.zeros(n_clusters, dtype=int)
    
    for i in range(n_clusters):
        counts[i] = np.sum(labels == i)
    
    # Calculate percentages
    hist = counts / len(labels) * 100

    sorted_indices = np.argsort(hist)[::-1]
        
    return hist[sorted_indices], sorted_indices

