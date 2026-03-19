import cmath
from math import pi, sin, asin, cos, atan, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 28})
def n_e_SiC(v, epsilon_inf = 6.52, w_LO = 968.0, w_TO = 793.0, GAMMA = 4.76, w_p = 6.0, gamma = 30.0):
    """外延层折射率公式"""
    return (cmath.sqrt(epsilon_inf * (1 + (w_LO**2 - w_TO**2) / (w_TO**2 - v**2 - v * gamma * 1j)) - w_p**2 / (v*(v + GAMMA * 1j)))).real
def n_s_SiC(v, epsilon_inf = 6.52, w_LO = 968.0, w_TO = 793.0, GAMMA = 4.76, w_p = 450.0, gamma = 580.0):
    """衬底折射率公式"""
    return (cmath.sqrt(epsilon_inf * (1 + (w_LO**2 - w_TO**2) / (w_TO**2 - v**2 - v * gamma * 1j)) - w_p**2 / (v*(v + GAMMA * 1j)))).real
def Fresnel(theta, n1, n2):
    """Fresnel公式, theta: 入射角, n1: 入射介质折射率, n2: 折射介质折射率"""
    theta = theta * pi / 180 # 入射角
    phi = asin(n1 * sin(theta) / n2) # 折射角
    rs = (n1 * cos(theta) - n2 * cos(phi)) / (n1 * cos(theta) + n2 * cos(phi)) # 垂直于参考平面的分量反射系数
    rp = (n2 * cos(theta) - n1 * cos(phi)) / (n2 * cos(theta) + n1 * cos(phi)) # 平行参考平面的分量反射系数
    ts = 2 * n1 * cos(theta) / (n1 * cos(theta) + n2 * cos(phi)) # 垂直于参考平面的分量透射系数
    tp = 2 * n1 * cos(theta) / (n2 * cos(theta) + n1 * cos(phi)) # 平行参考平面的分量透射系数
    phi = phi * 180 / pi # 折射角
    return rs, rp, ts, tp, phi
def Delta_phi(v, theta, n_e = n_e_SiC, n_s = n_s_SiC):
    """附加相位公式"""
    n0 = 1
    ne = n_e(v)
    ns = n_s(v)
    rs1, rp1, ts1, tp1, phi = Fresnel(theta, n0, ne)
    rs2, rp2, ts2, tp2, _ = Fresnel(phi, ne, ns)
    rs1_, rp1_, ts1_, tp1_, theta = Fresnel(phi, ne, n0)
    return - atan(rp1/rs1) + atan(tp1_*rp2*tp1/(ts1_*rs2*ts1))
def findpeaks(excel_file = 'data/incident_angle_10.xlsx', filtering = True, window_size = 35, prominence = 0.4):
    """光谱寻峰与可视化"""
    data = pd.read_excel(excel_file, header=None)
    data = data.iloc[1:]  # 跳过第一行（列标题）
    # 提取波数和反射率数据
    wavenumber = np.array(data.iloc[:, 0])  # 波数 (cm-1)
    reflectance = np.array(data.iloc[:, 1])  # 反射率 (%)
    # 创建图表
    plt.figure(figsize=(18, 10))
    # 绘制原始光谱
    if filtering:
        plt.plot(wavenumber, reflectance, 'b-', linewidth=1.5, alpha=0.5, label='原始光谱')
    else:
        plt.plot(wavenumber, reflectance, 'r-', linewidth=2.5, label='原始光谱')
    # 移动平均滤波
    if filtering:
        reflectance = moving_average(reflectance, window_size=window_size)
    # 使用Scipy的find_peaks函数
    peaks, _ = find_peaks(reflectance, height=0, prominence=prominence, width=1)
    valleys, _ = find_peaks(-reflectance, height=-100, prominence=prominence, width=1)
    # 峰值点和谷值点
    peak_points = np.array([wavenumber[peaks], reflectance[peaks]])
    valley_points = np.array([wavenumber[valleys], reflectance[valleys]])
    # 绘制滤波后的光谱
    if filtering:
        plt.plot(wavenumber, reflectance, 'r-', linewidth=2.5, label='移动平均滤波')
    plt.plot(wavenumber[peaks], reflectance[peaks], "v", color='blue', label="波峰", markersize=10) # 绘制峰值
    plt.plot(wavenumber[valleys], reflectance[valleys], "^", color='green', label="波谷", markersize=10)  # 绘制谷值
    # 添加图表标题和坐标轴标签
    plt.xlabel('波数 (cm$^{-1}$)', fontsize=28)
    plt.ylabel('反射率 (%)', fontsize=28)
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # 优化布局
    plt.tight_layout()
    # 显示图表
    plt.show()
    return peak_points, valley_points
def caculate_d(wvnums, theta, round = True):
    """计算厚度d"""
    theta = theta * pi/180
    lamda = [1/wvnums[i] for i in range(len(wvnums))]
    A = [sqrt(n_e_SiC(wvnums[i])**2 - sin(theta)**2) for i in range(len(wvnums))]
    PHI = [Delta_phi(wvnums[i], theta) for i in range(len(wvnums))]
    p = {}
    for i in range(len(wvnums)):
        p[i] = [(0.5*(lamda[j]*A[i] - lamda[i]*A[j])+0.5/pi*(PHI[i]*lamda[i]*A[j]-PHI[j]*lamda[j]*A[i])-0.5*(j-i)*lamda[j]*A[i])/(lamda[j]*A[i] - lamda[i]*A[j]) if i != j else 0 for j in range(len(wvnums))]
    d = np.zeros((len(wvnums), len(wvnums)))
    for i in range(len(wvnums)):
        for j in range(len(wvnums)):
            pij = (int(p[i][j]*2)/2 if not np.isnan(p[i][j]) else np.nan) if round else p[i][j]
            d[i][j] = 10000*(pij - 0.5 + PHI[i]/2/pi)*lamda[i]/(2*A[i]) if i > j else np.nan
    plt.figure(figsize=(12, 10))
    # 使用热图可视化d矩阵
    # 使用seaborn的heatmap函数，它可以提供更好的可视化效果
    sns.heatmap(d,
                annot=True,          # 在单元格中显示数值
                annot_kws={"size": 12},
                fmt=".2f",           # 数值格式
                cmap='viridis',      # 颜色映射
                cbar_kws={'label': '厚度d(μm)'},  # 颜色条标签
                linewidths=0.5,      # 单元格之间的线宽
                linecolor='white')   # 单元格之间的线颜色
    # 设置图表标题和坐标轴标签
    plt.xlabel('辅助极值点j', fontsize=24)
    plt.ylabel('参考极值点i', fontsize=24)
    # 优化布局
    plt.tight_layout()
    # 显示图表
    plt.show()
    # 取d的下三角线性化
    d_list = []
    for i in range(1, len(wvnums)):
        for j in range(i):
            if not np.isnan(d[i][j]):
                d_list.append(d[i][j])
    return d_list