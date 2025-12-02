import os, cv2, torch
import matplotlib
matplotlib.use('Agg') 
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, wait

from .CustomKmeans import KmeansPlusPlus_torch, get_cluster_statistics
from .ColorDifference import ciede2000_torch

# ===========================
# PyTorch 色彩空间转换函数
# ===========================
def rgb_to_lab_torch(rgb):
    """
    PyTorch 实现的 RGB 到 LAB 转换
    input: rgb tensor (N, 3) 范围 [0, 1]
    output: lab tensor (N, 3) L[0,100], a[-128,127], b[-128,127]
    """
    # 1. Inverse Gamma Correction (sRGB -> Linear RGB)
    mask = rgb > 0.04045
    rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # 2. Linear RGB -> XYZ
    # 转换矩阵 (D65)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=rgb.device)
    xyz = torch.matmul(rgb, M.T)
    
    # Normalize for D65 white point
    xyz[:, 0] = xyz[:, 0] / 0.95047
    xyz[:, 1] = xyz[:, 1] / 1.00000
    xyz[:, 2] = xyz[:, 2] / 1.08883
    
    # 3. XYZ -> LAB
    mask = xyz > 0.008856
    xyz_f = torch.where(mask, torch.pow(xyz, 1/3), 7.787 * xyz + 16/116)
    
    L = 116 * xyz_f[:, 1] - 16
    a = 500 * (xyz_f[:, 0] - xyz_f[:, 1])
    b = 200 * (xyz_f[:, 1] - xyz_f[:, 2])
    
    return torch.stack([L, a, b], dim=1)

def lab_to_rgb_torch(lab):
    """
    PyTorch 实现的 LAB 到 RGB 转换
    input: lab tensor (N, 3)
    output: rgb tensor (N, 3) 范围 [0, 1]
    """
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    fx3 = fx ** 3
    fy3 = fy ** 3
    fz3 = fz ** 3
    
    epsilon = 0.008856
    kappa = 903.3
    
    x = torch.where(fx3 > epsilon, fx3, (116 * fx - 16) / kappa)
    y = torch.where(L > (kappa * epsilon), fy3, L / kappa)
    z = torch.where(fz3 > epsilon, fz3, (116 * fz - 16) / kappa)
    
    xyz = torch.stack([x * 0.95047, y * 1.00000, z * 1.08883], dim=1)
    
    # XYZ -> Linear RGB
    M_inv = torch.tensor([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]], device=lab.device)
    
    rgb = torch.matmul(xyz, M_inv.T)
    
    # Linear RGB -> sRGB
    mask = rgb > 0.0031308
    rgb = torch.where(mask, 1.055 * (rgb ** (1/2.4)) - 0.055, 12.92 * rgb)
    
    # Clip to valid range
    rgb = torch.clamp(rgb, 0, 1)
    
    return rgb

# ===========================
# 独立的 I/O 任务函数
# ===========================
def save_fig_task(fig, output_path):
    try:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        fig.clf()
    except Exception as e:
        print(f"Error saving figure {output_path}: {e}")

def save_img_task(path, img_tensor):
    """
    保存图片
    img_tensor: CPU上的 numpy array (H, W, 4)
    """
    try:
        cv2.imwrite(path, img_tensor)
    except Exception as e:
        print(f"Error saving image {path}: {e}")

# ===========================
# 核心分析类
# ===========================

def get_hex_codes(centroids):
    # centroids 是 GPU Tensor, 需要转到 CPU 处理字符串
    centroids_np = centroids.detach().cpu().numpy().astype(int)
    return [f'#{r:02x}{g:02x}{b:02x}'.upper() for r, g, b in centroids_np]

class ColorClusterAnalyzer:
    def __init__(self, n_clusters=[5]):
        self.n_clusters = n_clusters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化聚类器
        self.clusters = [KmeansPlusPlus_torch(n_clusters=n, distance_func=ciede2000_torch, random_state=0) 
                         for n in n_clusters]

    def preprocess_image_tensor(self, img_path):
        """
        读取图片并转换为 GPU Tensor，在 GPU 上进行背景过滤和 LAB 转换
        """
        # I/O 依然使用 cv2，这是不可避免的，但读取后立即转 Tensor
        img_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_np is None:
            return None, None, None, None
        
        # 将数据搬运到 GPU
        img_tensor = torch.from_numpy(img_np).to(self.device)
        
        # 处理通道 (H, W, C)
        if len(img_tensor.shape) == 2: # Gray
            # 扩展为 BGRA
            h, w = img_tensor.shape
            img_bgra = torch.zeros((h, w, 4), dtype=torch.uint8, device=self.device)
            img_bgra[:,:,0] = img_tensor
            img_bgra[:,:,1] = img_tensor
            img_bgra[:,:,2] = img_tensor
            img_bgra[:,:,3] = 255
            img_tensor = img_bgra
        elif img_tensor.shape[2] == 3: # BGR
            # 扩展为 BGRA
            h, w, c = img_tensor.shape
            img_bgra = torch.cat([img_tensor, torch.full((h, w, 1), 255, dtype=torch.uint8, device=self.device)], dim=2)
            img_tensor = img_bgra
        
        # 此时 img_tensor 是 BGRA, uint8
        
        # 过滤背景
        alpha = img_tensor[:, :, 3]
        mask = alpha > 127
        
        # 获取前景像素坐标 (Rows, Cols)
        rows, cols = torch.where(mask)
        
        # 获取前景像素 BGR
        pixels_bgr = img_tensor[mask][:, :3] # (N, 3)
        
        if len(pixels_bgr) == 0:
            return None, None, None, None

        # BGR -> RGB -> Float [0, 1]
        pixels_rgb = torch.flip(pixels_bgr, dims=[1]).float() / 255.0
        
        # RGB -> LAB (Custom Torch Implementation)
        lab_pixels = rgb_to_lab_torch(pixels_rgb)
        
        # 归一化 LAB 到 [0, 100] 和 [-128, 127] 类似的范围供聚类器使用
        # 原始 lab_pixels: L[0,100], a,b 约[-100, 100]
        # 之前的代码有 lab2lab_ 归一化逻辑，这里保持一致性逻辑
        # lab2lab_: L -> L*100/255 (原代码看起来假设输入是 0-255 的 lab？)
        # 通常 cv2.cvtColor(RGB2LAB) 输出 uint8 时 L 是 0-255 (实际L*2.55).
        # 但我们这里是标准 LAB。
        # 为了兼容 ciede2000_torch (它期望标准的 Lab 值)，我们可以直接使用标准 Lab。
        # 但如果之前的聚类器参数是基于缩放过的值调整的，可能需要调整。
        # 假设 ciede2000 是标准实现，则不需要特殊归一化。
        # 原代码 lab2lab_ 似乎是为了把 uint8 的 cv2 LAB 转换回标准 LAB。
        # 既然我们现在直接计算出了标准 LAB，就不需要 lab2lab_ 了。
        
        return lab_pixels, (rows, cols), img_tensor.shape, alpha[mask]

    def cluster_image(self, lab_pixels, cluster_index):
        # 都在 GPU 上运行
        labels, centroids = self.clusters[cluster_index].fit_predict(lab_pixels)
        
        # Centroids 是 LAB 空间，转回 RGB 用于显示
        # LAB -> RGB
        centroids_rgb = lab_to_rgb_torch(centroids)
        # 转回 0-255
        centroids_rgb = (centroids_rgb * 255).clamp(0, 255).byte()
        
        return labels, centroids_rgb

    def create_hist_figure(self, hist, hex_codes, figsize=(8, 4), dpi=300):
        # 绘图部分涉及 matplotlib，必须回传到 CPU
        hist_cpu = hist.cpu().numpy()
        
        fig = Figure(figsize=figsize, dpi=dpi)
        _ = FigureCanvasAgg(fig) 
        ax = fig.add_subplot(111)
        
        base_fontsize = 15
        base_n_cluster = 5
        n = len(hex_codes)
        
        bars = ax.bar(hex_codes, hist_cpu, color=hex_codes, width=0.6)
        ax.set_ylabel('Percentage (%)', fontsize=8)
        ax.set_xlabel('Color', fontsize=8)
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
        # 1. 预处理 (全 Tensor)
        lab_pixels, coords, img_shape, valid_alpha = self.preprocess_image_tensor(img_path)
        
        if lab_pixels is None:
            return []
            
        rows, cols = coords
        futures = []

        for i, n in enumerate(self.n_clusters):
            output_dir_n = os.path.join(output_root_dir, f'clusters_{n}')
            if not os.path.exists(output_dir_n):
                os.makedirs(output_dir_n, exist_ok=True)
            
            # 2. 聚类 (全 Tensor)
            labels, centroids_rgb = self.cluster_image(lab_pixels, cluster_index=i)
            
            # 3. 统计 (全 Tensor)
            hist, sorted_indices = get_cluster_statistics(labels, centroids_rgb)
            
            # 准备绘图数据 (Tensor -> CPU)
            # centroids_rgb 是 (N, 3) RGB 顺序
            # get_hex_codes 需要 numpy
            hex_codes_sorted = get_hex_codes(centroids_rgb[sorted_indices])
            
            # 绘制直方图 (CPU 任务)
            fig = self.create_hist_figure(hist, hex_codes_sorted)
            hist_path = os.path.join(output_dir_n, 'color_hist.png')
            futures.append(thread_pool.submit(save_fig_task, fig, hist_path))
            
            # 4. 生成分割图 (Tensor 向量化操作)
            n_colors = len(centroids_rgb)
            H, W = img_shape[0], img_shape[1]
            
            # 初始化画布 (GPU)
            # 形状: (n_colors, H, W, 4)
            result_images = torch.zeros((n_colors, H, W, 4), dtype=torch.uint8, device=self.device)
            
            # 准备填充数据
            # labels 是每个像素的类别索引 (N_pixels,)
            # rows, cols 是像素坐标 (N_pixels,)
            # centroids_rgb 是颜色表 (n_colors, 3)
            
            # 我们需要一次性将每个像素的颜色填入对应的 result_images 层
            # 这里的逻辑是：对于属于第 k 类颜色的像素，填入第 k 张图
            
            # 获取每个像素对应的颜色值
            pixel_colors = centroids_rgb[labels] # (N_pixels, 3) RGB
            
            # 高级索引填充
            # 目标索引: [labels, rows, cols]
            # result_images[labels, rows, cols, :3] = pixel_colors (需要处理 RGB/BGR)
            
            # 因为 cv2 imwrite 需要 BGR，我们在这里直接存为 BGR
            # pixel_colors 是 RGB, 需要转 BGR
            pixel_colors_bgr = torch.flip(pixel_colors, dims=[1])
            
            result_images[labels, rows, cols, 0] = pixel_colors_bgr[:, 0] # B
            result_images[labels, rows, cols, 1] = pixel_colors_bgr[:, 1] # G
            result_images[labels, rows, cols, 2] = pixel_colors_bgr[:, 2] # R
            result_images[labels, rows, cols, 3] = valid_alpha          # A
            
            # 将 Tensor 移回 CPU 并转为 Numpy 以便保存
            # 注意：如果显存不够，这一步可以分批做，或者在循环里逐个转
            result_images_np = result_images.cpu().numpy()
            
            # 提交保存任务
            for rank_idx in range(n_colors):
                color_idx = sorted_indices[rank_idx].item()
                file_name = f'{rank_idx + 1}_{hex_codes_sorted[rank_idx]}.png'
                seg_path = os.path.join(output_dir_n, file_name)
                
                # 这是一个纯内存拷贝，非常快
                img_to_save = result_images_np[color_idx]
                
                # 如果图片是全空的（除了alpha），可以考虑跳过吗？原逻辑没跳过，这里保留
                futures.append(thread_pool.submit(save_img_task, seg_path, img_to_save))

        wait(futures)
        return futures

# ===========================
# 多进程 Worker 逻辑
# ===========================
def init_worker(n_clusters_arg, output_dir_arg):
    global g_analyzer, g_root_dir, g_thread_pool
    # 多进程下，每个进程独占 GPU 的一部分或全部，
    # 但如果 GPU 显存有限，建议减少 max_workers
    torch.set_num_threads(1) 
    g_root_dir = output_dir_arg
    g_thread_pool = ThreadPoolExecutor(max_workers=4) 
    g_analyzer = ColorClusterAnalyzer(n_clusters=n_clusters_arg)

def work(img_path):
    try:
        global g_analyzer, g_root_dir, g_thread_pool
        filename = os.path.splitext(os.path.basename(img_path))[0]
        current_output_dir = os.path.join(g_root_dir, filename)
        g_analyzer.analyze_image(img_path, current_output_dir, g_thread_pool)
        return img_path
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {img_path}: {e}")
        return None

def cluster_image_multi_process(img_paths, max_workers=2, n=20, output_dir='results'):
    """
    注意：使用 GPU 时，max_workers 不宜过大，否则显存会炸。
    建议设置为 1 或 2，取决于 GPU 显存大小。
    """
    n_clusters = list(range(1, n + 1))
    
    # 强制设置启动方法为 spawn，这在 CUDA 环境下通常更安全
    # 但在已有上下文里可能报错，根据环境调整。Linux下默认fork。
    # 这里保持默认。
    
    print(f"Starting processing with {max_workers} workers")
    
    pool_args = (n_clusters, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    
    with Pool(processes=max_workers, initializer=init_worker, initargs=pool_args) as pool:
        results = list(tqdm(pool.imap_unordered(work, img_paths), 
                           total=len(img_paths),
                           desc="Processing Images"))