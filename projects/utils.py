from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import numpy as np
import cv2, os
from PIL import Image

# 核密度估计并绘图
def kde_estimation(data, output_path='kde_plot.png'):

    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 100)
    plt.figure(figsize=(10, 6))  # 设置图片大小（英寸）
    plt.plot(x, kde(x), 'b-', linewidth=2, label='KDE Curve')
    plt.hist(data, bins=100, density=True, alpha=0.3, color='gray', label='Data Histogram')
    plt.title('Kernel Density Estimation Plot')
    plt.xlabel('value')
    plt.ylabel('density')
    plt.legend()
    plt.grid(alpha=0.3)

    # 保存图片（确保在plt.show()之前调用）
    plt.savefig('kde_plot.png', dpi=300, bbox_inches='tight')

def show_transparent(img_path, output_dir):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    transparent = img[:,:,3]
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, transparent)

def clip_img_according_to_transparent(img_path, output_dir, threshold = 254):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    keep = img[:,:,3] > threshold
    rows, cols = np.where(keep)
    min_row = min(rows)
    min_col = min(cols)
    max_col = max(cols)
    max_row = max(rows)
    img = img[min_row:max_row + 1, min_col:max_col + 1]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, img)

def proccess_img_folder(img_dir, out_dir, proccess_function, **kwargs):
    if not os.path.exists(out_dir):
            os.makedirs(out_dir) 
    for root, _, files in os.walk(img_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    
                    img_path = os.path.join(root, file)
                    proccess_function(img_path, out_dir, **kwargs)

def stat_transparent(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    height, width, channels = img.shape
    img = img[:,:,3]
    transparent = [0]*256
    for y in range(height):
        for x in range(width):
            transparent[img[y, x]] += 1
    for i, num in enumerate(transparent):
        print(i, num)
    return transparent

def merge(imgs_path, output_path):
    imgs = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in imgs_path]
    merged_img = np.concatenate(imgs, axis=1)
    cv2.imwrite(output_path, merged_img)

def show_transparency(image_path, output_path="text.png"):
    """
    （蓝=透明，红=不透明）
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像保存路径，None则自动生成
    """
    try:
        # 打开图像并转换为RGBA模式
        with Image.open(image_path) as img:
            rgba_img = img.convert('RGBA')
            img_array = np.array(rgba_img, dtype=np.float32)  # 转换为数组便于处理
            
            # 提取Alpha通道（0-255）
            alpha = img_array[:, :, 3] / 255.0  # 归一化到0-1范围
            
            # 计算RGB颜色：根据透明度混合红蓝
            # 透明度越高（alpha越小）越蓝；透明度越低（alpha越大）越红
            red = alpha  # 不透明度高→红色强
            blue = 1 - alpha  # 透明度高→蓝色强
            green = np.zeros_like(alpha)  # 绿色通道设为0，突出红蓝对比
            
            # 创建新的RGB通道
            new_rgb = np.stack([red, green, blue], axis=-1) * 255.0  # 转换回0-255
            new_rgb = new_rgb.astype(np.uint8)  # 转为整数类型
            
            # 转换为图像并保存
            result_img = Image.fromarray(new_rgb, mode='RGB')
            
            # 保存结果
            result_img.save(output_path)
            print(f"已在原图上叠加透明度颜色并保存至: {output_path}")
            return result_img
            
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, None

# 边缘平滑
def edge_anti_aliasing(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测边缘
    edges = cv2.Canny(gray, 50, 150)

    # 对边缘区域进行高斯模糊
    kernel = np.ones((3, 3), np.float32) / 9
    edges_smoothed = cv2.filter2D(edges, -1, kernel)

    # 将平滑后的边缘与原图融合
    result = cv2.addWeighted(image, 0.8, cv2.cvtColor(edges_smoothed, cv2.COLOR_GRAY2BGR), 0.2, 0)

    # 保存结果
    cv2.imwrite("edge_smoothed.png", result)

if __name__ == "__main__":
     img_path = 'riders/Unit_FullView_1020001701.png'
     show_transparency(img_path, output_path="text.png")