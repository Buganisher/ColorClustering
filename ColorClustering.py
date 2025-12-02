import os, cv2, time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from CustomKmeans import KmeansPlusPlus_torch, get_cluster_statistics
from ColorDifference import ciede2000_torch

def lab2lab_(labs):
    # 确保输入是浮点数类型，避免整数溢出
    labs = labs.astype(np.float64)
    # 缩放L通道：[0,255] -> [0,100]
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

def downsample_image(image, scale_factor=0.1):
    """
    使用OpenCV对图像进行下采样
    
    参数:
    image: 输入图像，numpy数组
    scale_factor: 缩放因子，小于1的值会缩小图像
    
    返回:
    downsampled: 下采样后的图像
    """
    # 确保缩放因子在有效范围内
    if scale_factor <= 0 or scale_factor >= 1:
        raise ValueError("Scale factor must be between 0 and 1")
    
    # 计算新的尺寸
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_size = (new_width, new_height)
    
    # 使用不同的插值方法进行下采样
    # INTER_AREA适用于缩小图像，能提供较好的效果
    downsampled = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    return downsampled

def get_hex_codes(centroids):
    """
    将RGB颜色转换为十六进制代码
    
    Args:
        centroids: RGB颜色中心
        
    Returns:
        hex_codes: 十六进制颜色代码列表
    """
    return [f'#{r:02x}{g:02x}{b:02x}'.upper() for r, g, b in centroids]

def filter_background(img_bgra, threshold = 127):
    """
    过滤透明背景像素
    
    Args:
        img: BGRA格式的图像
        
    Returns:
        过滤后的像素
    """
    non_background = (img_bgra[:, :, 3] > threshold )
    rows, cols = np.where(non_background)
    coords = np.column_stack((rows, cols))
    non_background_pixels = img_bgra[non_background]
    return non_background_pixels, coords

def preprocess_image(img_path):
    # 读取图片
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return None, None, None
    # img = downsample_image(img)
    # 过滤透明背景
    pixels_filtered, non_background_coords= filter_background(img)
    # 提取alpha通道
    alpha = pixels_filtered[:, 3]
    pixels_filtered = pixels_filtered[:, :3]  # 只保留RGB通道
    # 转换到LAB颜色空间
    lab_pixels = cv2.cvtColor(pixels_filtered.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab_pixels = lab2lab_(lab_pixels)
    
    return lab_pixels, non_background_coords, img.shape, alpha

def plot_histogram(hist, hex_codes, output_path,
                   dpi=300, figsize=(8, 4)):
        """
        绘制并保存颜色分布直方图
        
        Args:
            hist: 颜色分布直方图
            centroids: RGB颜色中心
            filename: 输出文件名
        """
        
        base_fontsize = 15
        base_n_cluster = 5
        
        n = len(hex_codes)
        
        plt.figure(figsize=figsize, dpi=dpi) 
        bars = plt.bar(hex_codes, hist, color=hex_codes, width=0.6)
        # plt.title(filename, fontsize=10, pad=10)
        plt.ylabel('Percentage (%)', fontsize=8)
        plt.xlabel('Color', fontsize=8)
        plt.xticks(rotation=45, fontsize=7)
        plt.yticks(fontsize=7)
        
        plt.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.5)
        
        # 添加百分比标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', 
                     fontsize=base_fontsize * base_n_cluster / n)
        
        plt.tight_layout()

        global thread_pool
        thread_pool.submit(save_plt, output_path)

def save_plt(output_path):
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def plot_segmentations(labels, coords_all, centroids_rgb, sorted_indices, 
                       img_shape, alpha, output_dir):
    
    n = len(centroids_rgb)
    hex_codes = get_hex_codes(centroids_rgb)
    images = np.zeros((n, img_shape[0], img_shape[1], 4), dtype=np.uint8) # 透明背景
    
    for i , coords in enumerate(coords_all):
        y, x = coords
        color_num = labels[i]
        (r, g, b) = centroids_rgb[color_num]
        bgra_color = (b, g, r, alpha[i])
        images[color_num, y, x] = bgra_color
    
    for i in range(n):
        color_num = sorted_indices[i]
        output_path = os.path.join(output_dir, f'{i + 1}_{hex_codes[color_num]}.png')
        global thread_pool
        thread_pool.submit(cv2.imwrite, output_path, images[color_num])

class ColorClusterAnalyzer:
    """颜色聚类分析工具，支持将结果输出到Excel文件"""
    def __init__(self, n_clusters=[5]):
        """
        初始化颜色聚类分析器
        
        Args:
            folder_path: 图片文件夹路径
            output_dir: 输出文件夹路径
            excel_path: Excel输出路径
            n_clusters: 聚类数量
            dpi: 图表分辨率
            figsize: 图表大小
        """
        self.n_clusters = n_clusters
        self.clusters = [KmeansPlusPlus_torch(n_clusters=n, distance_func=ciede2000_torch, random_state=0) 
                         for n in n_clusters]

        print(f"Initialized ColorClusterAnalyzer with clusters: {n_clusters}")

    def cluster_image(self, lab_pixels, cluster_index):
        """
        单次聚类
        
        Args:
            img_path: 图片路径
            
        Returns:
            元组(颜色分布直方图, RGB颜色中心, HSB颜色中心)
        """
        # 执行聚类
        labels, centroids = self.clusters[cluster_index].fit_predict(lab_pixels)
        
        centroids = lab_2lab(centroids)
        # 转回RGB
        centroids_rgb = cv2.cvtColor(centroids.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
        
        return labels, centroids_rgb

    def analyze_image(self, img_path, output_dir):
        """
        分析单张图片
        
        Args:
            img_path: 图片路径
            n_cluster: 聚类数量
            
        Returns:
            元组(颜色分布直方图, RGB颜色中心, HSB颜色中心)
        """

        lab_pixels, non_background_coords, img_shape, alpha = preprocess_image(img_path)
        
        for i, n in enumerate(self.n_clusters):
            
            output_dir_n = os.path.join(output_dir, f'clusters_{n}')
            if not os.path.exists(output_dir_n):
                os.makedirs(output_dir_n)
            
            labels, centroids_rgb, = self.cluster_image(lab_pixels, cluster_index=i)
            
            hist, sorted_indices = get_cluster_statistics(labels, centroids_rgb)
            hex_codes_sorted = get_hex_codes(centroids_rgb[sorted_indices])
            output_path = os.path.join(output_dir_n, 'color_hist.png')
            plot_histogram(hist, hex_codes_sorted, output_path)
            plot_segmentations(labels, non_background_coords, centroids_rgb, sorted_indices
                               ,img_shape, alpha ,output_dir_n)
            
        return None
            
    def analyze_folder(self, img_dir, oput_dir):
        """
        分析文件夹中的所有图片
        
        Args:
            img_dir: 图片文件夹路径
        """
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        print(f"开始分析文件夹: {img_dir}")
        
        img_paths = []
        out_dirs = []
        start_time = time.time()
        for root, _, files in os.walk(img_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    
                    img_path = os.path.join(root, file)
                    img_paths.append((img_path, file))
                    filename = os.path.splitext(file)[0]
                    output_dir = os.path.join(oput_dir, filename)
                    out_dirs.append(output_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    self.analyze_image(img_path, output_dir)

        end_time = time.time()
        print(f"文件夹分析完成，耗时 {end_time - start_time:.2f} 秒")
        
# muti_process version --- IGNORE ---

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

def init_worker(n_clusters, output_dir):
    """子进程启动时初始化线程池"""
    print(f"process {os.getpid()} initializing", flush=True)
    
    global root_dir, thread_pool, analyzer
    root_dir = output_dir
    thread_pool = ThreadPoolExecutor(max_workers=2)
    analyzer = ColorClusterAnalyzer(n_clusters = n_clusters)

    print(f"process {os.getpid()} initialized", flush=True)

def work(img_path):
    """在每个子进程中使用全局变量"""
    global analyzer, root_dir
    filename = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join(root_dir, filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"processing {img_path}", flush=True)
    result = analyzer.analyze_image(img_path, output_dir)
    print(f"save results to {output_dir}", flush=True)
    return result

def cluster_image_muti_process(img_paths, 
                               max_workers = 8, 
                               n = 20,
                               output_dir = 'results'):
    
    n_clusters = range(1, n + 1)
                
    print(f"Starting multiprocessing with {max_workers} workers...")
    print(f"Total images to process: {len(img_paths)}")
    with Pool(max_workers, initializer=init_worker, initargs=(n_clusters, output_dir)) as pool:
         
         for res in tqdm(pool.imap_unordered(work, img_paths), 
                         total=len(img_paths), 
                         desc="Processing", 
                         mininterval = 5):
            output = res
            

             
    







