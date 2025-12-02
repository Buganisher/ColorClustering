from projects.ColorClustering import ColorClusterAnalyzer, cluster_image_multi_process
import os
from projects.utils import proccess_img_folder, clip_img_according_to_transparent

def cluster_image(imgs = [''], img_dir = 'cluster_used', 
                  root_dir = 'color_cluster_results', n = 20, is_folder = False):
    # 配置参数
    n_clusters = range(1, n + 1)
    analyzer = ColorClusterAnalyzer(n_clusters = n_clusters, root_dir = root_dir)
    
    if is_folder:
        analyzer.analyze_folder(img_dir)
    
    else:
        for img in imgs:
            # 处理单图
            img_path = os.path.join(img_dir, img)
            analyzer.analyze_image(img_path, root_dir)

def get_images_path(folder):
    img_paths = []
    # 处理文件夹
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                
                img_path = os.path.join(root, file)
                img_paths.append(img_path)
    return img_paths
    
if __name__ == "__main__":
    
    folder = 'images_clipped'
    out = 'color_cluster_results'

    img_paths = ['temp/Unit_MainView_1015001203.png']
    img_paths = get_images_path(folder)

    # for img_path in img_paths:
    #     clip_img_according_to_transparent(img_path, 
    #                                       out,
    #                                       threshold = 1)
    
    
    
    # img_paths = ['riders_clipped/Unit_FullView_1059005701.png']
    cluster_image_multi_process(img_paths, 
                               max_workers = 8, 
                               n = 20,
                               output_dir=out)