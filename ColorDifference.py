import torch

def ciede2000_torch(Lab1, Lab2):
    """
    使用GPU加速计算CIEDE2000色差
    
    参数:
    Lab1: numpy数组或PyTorch张量，形状为(n, 3)
    Lab2: numpy数组或PyTorch张量，形状为(m, 3)
    device: 'cuda' 或 'cpu'，指定计算设备
    
    返回:
    PyTorch张量，形状为(n, m)，表示所有颜色对之间的色差
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # 转换为PyTorch张量并移至指定设备
    if not isinstance(Lab1, torch.Tensor):
        Lab1 = torch.tensor(Lab1, dtype=dtype, device=device)
    if not isinstance(Lab2, torch.Tensor):
        Lab2 = torch.tensor(Lab2, dtype=dtype, device=device)
    
    # 提取L、a、b通道
    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]
    
    # 扩展维度以实现批量计算
    L1 = L1.unsqueeze(1)  # [n, 1]
    a1 = a1.unsqueeze(1)
    b1 = b1.unsqueeze(1)
    L2 = L2.unsqueeze(0)  # [1, m]
    a2 = a2.unsqueeze(0)
    b2 = b2.unsqueeze(0)
    
    # Step 1: 计算C'和h'
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    
    G = 0.5 * (1 - torch.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    C1_prime = torch.sqrt(a1_prime**2 + b1**2)
    C2_prime = torch.sqrt(a2_prime**2 + b2**2)
    
    # 计算色调角h'
    h1_prime = torch.atan2(b1, a1_prime) * 180 / torch.pi
    h2_prime = torch.atan2(b2, a2_prime) * 180 / torch.pi
    
    # 确保角度在0-360度之间
    h1_prime = h1_prime % 360
    h2_prime = h2_prime % 360
    
    # Step 2: 计算ΔL', ΔC', 和ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    # 计算ΔH'
    delta_h_prime = h2_prime - h1_prime
    
    # 处理角度差超过180度的情况
    mask = torch.abs(delta_h_prime) > 180
    delta_h_prime = torch.where(
        mask,
        torch.where(
            delta_h_prime > 0, 
            delta_h_prime - 360, 
            delta_h_prime + 360
        ),
        delta_h_prime
    )
    
    delta_H_prime = 2 * torch.sqrt(C1_prime * C2_prime) * torch.sin(torch.deg2rad(delta_h_prime / 2))
    
    # Step 3: 计算CIEDE2000色差
    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    
    # 计算平均色调角h_bar_prime
    mask_zero = (C1_prime * C2_prime) == 0
    h_bar_prime = torch.where(
        mask_zero,
        h1_prime + h2_prime,
        torch.where(
            torch.abs(h1_prime - h2_prime) > 180,
            (h1_prime + h2_prime + 360) / 2,
            (h1_prime + h2_prime) / 2
        )
    )
    
    # 确保h_bar_prime在0-360度之间
    h_bar_prime = h_bar_prime % 360
    
    # 计算T
    T = 1 - 0.17 * torch.cos(torch.deg2rad(h_bar_prime - 30)) + \
        0.24 * torch.cos(torch.deg2rad(2 * h_bar_prime)) + \
        0.32 * torch.cos(torch.deg2rad(3 * h_bar_prime + 6)) - \
        0.20 * torch.cos(torch.deg2rad(4 * h_bar_prime - 63))
    
    # 计算delta_theta
    delta_theta = 30 * torch.exp(-(((h_bar_prime - 275) / 25)**2))
    
    # 计算Rc
    Rc = 2 * torch.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    
    # 计算SL, SC, SH
    SL = 1 + (0.015 * ((L_bar_prime - 50)**2)) / torch.sqrt(20 + ((L_bar_prime - 50)**2))
    SC = 1 + 0.045 * C_bar_prime
    SH = 1 + 0.015 * C_bar_prime * T
    
    # 计算RT
    RT = -torch.sin(torch.deg2rad(2 * delta_theta)) * Rc
    
    # 计算最终色差
    delta_E_00 = torch.sqrt(
        (delta_L_prime / SL)**2 + 
        (delta_C_prime / SC)**2 + 
        (delta_H_prime / SH)**2 + 
        RT * (delta_C_prime / SC) * (delta_H_prime / SH)
    )
    
    return delta_E_00

if __name__ == "__main__":
    # 测试代码
    
    a =[
        [17.7900, 7.9800, 11.1100],
        [48.4500, 9.5700, 13.0700],
        [30.1100, 2.1300, -20.6500],
        [23.2400, -10.7800, 17.7700],
        [33.5500, 9.6300, -22.1100],
        [51.5200, -21.9100, -0.7600]
    ]

    
    b =[
        [37.5420, 12.0180, 13.3300],
        [65.2000, 14.8210, 17.5450],
        [50.3660, -1.5730, -21.4310],
        [43.1250, -14.6300, 22.1200],
        [55.3430, 11.4490, -25.2890],
        [71.3600, -32.7180, 1.6360]
    ]
    
    delta_E = ciede2000_torch(a, b)
    print(delta_E.diag())