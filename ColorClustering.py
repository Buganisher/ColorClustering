import os
import cv2
import time
import numpy as np
import matplotlib
# 【关键】设置非交互式后端，必须在导入 pyplot 之前设置
# Agg 后端是线程安全的，且不需要显示器支持
matplotlib.use('Agg') 
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, wait
import torch

# 假设这些是你自定义的模块，请确保它们在当前目录下
from CustomKmeans import KmeansPlusPlus_torch, get_cluster_statistics
from ColorDifference import ciede2000_torch

# ===========================
# 独立的 I/O 任务函数 (必须定义在全局)
# ===========================
def save_fig_task(fig, output_path):
    """线程任务：保存 Matplotlib Figure"""
    try:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        # 显式关闭 figure 以释放内存
        fig.clf()
    except Exception as e:
        print(f"Error saving figure {output_path}: {e}")

def save_img_task(path, img):
    """线程任务：保存图片"""
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        print(f"Error saving image {path}: {e}")

# ===========================
# 数据处理辅助函数
# ===========================
def lab2lab_(labs):
    labs = labs.astype(np.float64)
    labs[:, 0] = 100.0 * labs[:, 0] / 255.0
    labs[:, 0] = np.clip(labs[:, 0], 0, 100)
    labs[:, 1:] = labs[:, 1:] - 128.0
    labs[:, 1:] = np.clip(labs[:, 1:], -128, 127)
    return labs

def lab_2lab(labs):
    labs = labs.astype(np.float64)
    labs[:, 0] = 255.0 * labs[:, 0] / 100.0
    labs[:, 1:] = labs[:, 1:] + 128.0
    labs = np.clip(labs, 0, 255)
    return labs.astype(np.uint8)

def get_hex_codes(centroids):
    return [f'#{r:02x}{g:02x}{b:02x}'.upper() for r, g, b in centroids]

def filter_background(img_bgra, threshold=127):
    non_background = (img_bgra[:, :, 3] > threshold)
    rows, cols = np.where(non_background)
    # coords = np.column_stack((rows, cols)) # 如果后面不需要 coords 可以不返回
    non_background_pixels = img_bgra[non_background]
    return non_background_pixels, (rows, cols)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return None, None, None, None
    
    # 处理不同通道数的图片
    if len(img.shape) == 2: # Gray
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3: # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
    pixels_filtered, (rows, cols) = filter_background(img)
    
    if len(pixels_filtered) == 0:
        return None, None, None, None

    alpha = pixels_filtered[:, 3]
    pixels_filtered = pixels_filtered[:, :3]  # 只保留RGB通道
    
    lab_pixels = cv2.cvtColor(pixels_filtered.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab_pixels = lab2lab_(lab_pixels)
    
    return lab_pixels, (rows, cols), img.shape, alpha

# ===========================
# 核心分析类
# ===========================
class ColorClusterAnalyzer:
    def __init__(self, n_clusters=[5]):
        self.n_clusters = n_clusters
        # 预先初始化所有聚类器
        self.clusters = [KmeansPlusPlus_torch(n_clusters=n, distance_func=ciede2000_torch, random_state=0) 
                         for n in n_clusters]
        # print(f"Initialized Analyzer for clusters: {n_clusters}")

    def cluster_image(self, lab_pixels, cluster_index):
        labels, centroids = self.clusters[cluster_index].fit_predict(lab_pixels)
        centroids = lab_2lab(centroids)
        centroids_rgb = cv2.cvtColor(centroids.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
        return labels, centroids_rgb

    def create_hist_figure(self, hist, hex_codes, figsize=(8, 4), dpi=300):
        """【重要】创建独立的 Figure 对象，不依赖 plt 全局状态"""
        fig = Figure(figsize=figsize, dpi=dpi)
        # 必须绑定 Canvas 才能保存
        _ = FigureCanvasAgg(fig) 
        ax = fig.add_subplot(111)
        
        base_fontsize = 15
        base_n_cluster = 5
        n = len(hex_codes)
        
        bars = ax.bar(hex_codes, hist, color=hex_codes, width=0.6)
        ax.set_ylabel('Percentage (%)', fontsize=8)
        ax.set_xlabel('Color', fontsize=8)
        # 旋转 x 轴标签
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', 
                    fontsize=base_fontsize * base_n_cluster / n)
        
        fig.tight_layout()
        return fig

    def analyze_image(self, img_path, output_root_dir, thread_pool):
        """
        分析单张图片
        返回: 提交的所有 future 对象列表
        """
        lab_pixels, (rows, cols), img_shape, alpha = preprocess_image(img_path)
        
        if lab_pixels is None:
            return []

        futures = [] # 收集所有任务凭证

        for i, n in enumerate(self.n_clusters):
            # 1. 准备路径
            output_dir_n = os.path.join(output_root_dir, f'clusters_{n}')
            if not os.path.exists(output_dir_n):
                os.makedirs(output_dir_n, exist_ok=True)
            
            # 2. 执行聚类 (计算密集型，在主线程做)
            labels, centroids_rgb = self.cluster_image(lab_pixels, cluster_index=i)
            hist, sorted_indices = get_cluster_statistics(labels, centroids_rgb)
            hex_codes_sorted = get_hex_codes(centroids_rgb[sorted_indices])
            
            # 3. 绘制直方图 (在主线程创建对象，在子线程保存)
            fig = self.create_hist_figure(hist, hex_codes_sorted)
            hist_path = os.path.join(output_dir_n, 'color_hist.png')
            
            # 提交保存 Figure 的任务
            f_hist = thread_pool.submit(save_fig_task, fig, hist_path)
            futures.append(f_hist)
            
            # 4. 生成分割图 (内存操作在主线程，I/O在子线程)
            # 预先分配好内存
            n_colors = len(centroids_rgb)
            images = np.zeros((n_colors, img_shape[0], img_shape[1], 4), dtype=np.uint8)
            
            # 填充颜色 (向量化操作优化)
            # 这里如果不熟悉向量化，用循环也可以，Numpy 循环比 Python 快
            # 为了保持逻辑清晰，沿用之前的逻辑，但建议优化
            # 这里简单起见，还是用原来的逻辑，但是要注意 rows/cols 的使用
            for k, (r, c) in enumerate(zip(rows, cols)):
                color_idx = labels[k]
                rgb = centroids_rgb[color_idx]
                # BGRA
                images[color_idx, r, c] = [rgb[2], rgb[1], rgb[0], alpha[k]]
                
            # 提交保存分割图的任务
            for rank_idx in range(n_colors):
                color_idx = sorted_indices[rank_idx] # 按占比排序
                file_name = f'{rank_idx + 1}_{hex_codes_sorted[rank_idx]}.png'
                seg_path = os.path.join(output_dir_n, file_name)
                
                f_img = thread_pool.submit(save_img_task, seg_path, images[color_idx])
                futures.append(f_img)

        # 【关键】等待本张图片的所有后台 I/O 任务完成
        # 这样能保证当 work 函数返回时，所有文件已安全写入磁盘
        wait(futures)
        return futures

# ===========================
# 多进程 Worker 逻辑
# ===========================

def init_worker(n_clusters_arg, output_dir_arg):
    """
    子进程初始化函数
    initargs=(n_clusters, output_dir) 会传参给这里
    """
    global g_analyzer, g_root_dir, g_thread_pool
    
    # 1. 限制 PyTorch 在每个进程中只使用 1 个 CPU 核
    # 因为我们已经开了多进程，如果不限制，CPU 会严重超载导致死机或变慢
    torch.set_num_threads(1) 
    
    # 2. 初始化全局变量
    g_root_dir = output_dir_arg
    
    # 每个进程开 2-4 个线程负责写文件即可，太多会造成磁盘 IO 瓶颈
    g_thread_pool = ThreadPoolExecutor(max_workers=4) 
    
    # 初始化分析器（加载模型等）
    g_analyzer = ColorClusterAnalyzer(n_clusters=n_clusters_arg)
    
    print(f"Worker process {os.getpid()} initialized.", flush=True)

def work(img_path):
    """
    实际的工作函数
    """
    try:
        global g_analyzer, g_root_dir, g_thread_pool
        
        filename = os.path.splitext(os.path.basename(img_path))[0]
        current_output_dir = os.path.join(g_root_dir, filename)
        
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir, exist_ok=True)
            
        # 调用分析逻辑，并在内部等待 I/O 完成
        g_analyzer.analyze_image(img_path, current_output_dir, g_thread_pool)
        
        return img_path # 返回成功处理的路径
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}", flush=True)
        return None

# ===========================
# 主程序入口
# ===========================
def cluster_image_multi_process(img_paths, 
                                max_workers=4, # CPU 核心数
                                n=20, 
                                output_dir='results'):
    
    n_clusters = list(range(1, n + 1))
    
    print(f"Starting multiprocessing with {max_workers} workers.")
    print(f"Total images: {len(img_paths)}")
    print(f"Clusters per image: {n_clusters}")
    
    # initargs 必须是元组
    pool_args = (n_clusters, output_dir)
    
    with Pool(processes=max_workers, initializer=init_worker, initargs=pool_args) as pool:
        # 使用 imap_unordered 稍微快一点，因为不保证顺序
        results = list(tqdm(pool.imap_unordered(work, img_paths), 
                           total=len(img_paths),
                           desc="Processing Images"))
                           
    print("All tasks finished.")

if __name__ == "__main__":
    # 测试代码
    # 假设有一个 images 文件夹
    source_folder = "images" 
    output_folder = "output_results"
    
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
        print(f"请在 {source_folder} 下放入图片进行测试")
    else:
        # 收集图片路径
        all_images = []
        for root, _, files in os.walk(source_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(root, file))
        
        if all_images:
            # 运行多进程
            # 建议 max_workers 设置为 CPU 物理核心数的一半到全部
            cluster_image_multi_process(all_images, max_workers=4, n=5, output_dir=output_folder)
        else:
            print("No images found.")