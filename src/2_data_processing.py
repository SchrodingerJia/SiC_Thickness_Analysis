import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 正态分布KS拟合优度检验函数
def normality_ks_test(data, mu=None, sigma=None, alpha=0.05, verbose=True):
    """
    对数据进行指定参数的正态分布KS拟合优度检验
    
    参数:
    data -- 待检验的数据
    mu -- 指定的均值，如果为None则使用样本均值
    sigma -- 指定的标准差，如果为None则使用样本标准差
    alpha -- 显著性水平，默认为0.05
    verbose -- 是否输出详细结果，默认为True
    
    返回:
    dict -- 包含检验结果的字典
    """
    # 计算样本均值和标准差（如果未指定）
    if mu is None:
        mu = np.mean(data)
    if sigma is None:
        sigma = np.std(data, ddof=1)
    
    # 执行KS检验
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu, sigma))
    
    # 计算并分位数比较
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_comparison = {}
    for q in quantiles:
        observed_q = np.quantile(data, q)
        theoretical_q = stats.norm.ppf(q, mu, sigma)
        quantile_comparison[q] = {
            'observed': observed_q,
            'theoretical': theoretical_q,
            'difference': abs(observed_q - theoretical_q)
        }
    
    # 初始化结果字典
    results = {
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'mu': mu,
        'sigma': sigma,
        'result': "不能拒绝正态性假设" if ks_p > alpha else "拒绝正态性假设",
        'quantile_comparison': quantile_comparison
    }
    
    # 输出详细结果
    if verbose:
        print("\n正态分布KS拟合优度检验结果:")
        print(f"指定参数: 均值={mu:.4f}, 标准差={sigma:.4f}")
        print(f"KS统计量: {ks_stat:.4f}")
        print(f"p值: {ks_p:.4f}")
        print(f"结论: {results['result']}")
        
        print("\n分位数比较:")
        print("分位数\t观测值\t理论值\t差异")
        for q in quantiles:
            comp = quantile_comparison[q]
            print(f"{q:.2f}\t{comp['observed']:.4f}\t{comp['theoretical']:.4f}\t{comp['difference']:.4f}")
        
        # 绘制直方图和正态分布拟合
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50, density=True, alpha=0.5, color='g', label='原始数据')
        
        plt.xlim(5,12)
        plt.ylim(0, 4.5)    

        # 绘制理论正态分布
        x = np.linspace(5, 12, 1000)
        y = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, y, 'r-', linewidth=2, label=f'N({mu:.2f}, {sigma:.2f}^2)')
        
        plt.xlabel('厚度值(μm)', fontsize=24)
        plt.ylabel('概率密度', fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return results

def analyze_p_value_bias(data, n_simulations=10000):
    """
    分析使用标准K-S分布在参数估计情况下的p值偏差
    """
    np.random.seed(42)
    n = len(data)

    # 估计原始正态分布
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    p_values_known = []
    p_values_estimated_wrong = []
    ks_stats_estimated = []
    ks_stats_known = []

    # 原始KS统计量
    ks_stat_original, _ = stats.kstest(data, 'norm', args=(mu, sigma))
    
    for i in range(n_simulations):
        if (i+1) % 1000 == 0:
            print(f"p值偏差分析: 已完成 {i+1}/{n_simulations}")
        
        sample = np.random.normal(mu, sigma, n)
        
        # 参数已知情况的KS统计量和p值
        ks_stat_known, p_known = stats.kstest(sample, 'norm', args=(mu, sigma))
        p_values_known.append(p_known)
        ks_stats_known.append(ks_stat_known)
        
        # 参数估计情况
        mu_est = np.mean(sample)
        sigma_est = np.std(sample, ddof=0)
        ks_stat, p_wrong = stats.kstest(sample, 'norm', args=(mu_est, sigma_est))
        
        p_values_estimated_wrong.append(p_wrong)
        ks_stats_estimated.append(ks_stat)
    
    # 计算正确的p值（通过模拟）
    ks_stats_estimated = np.array(ks_stats_estimated)
    critical_values = np.percentile(ks_stats_estimated, [90, 95, 99])
    
    print(f"\n参数估计情况下K-S统计量的经验分位数:")
    print(f"  10%水平临界值: {critical_values[0]:.4f}")
    print(f"  5%水平临界值: {critical_values[1]:.4f}") 
    print(f"  1%水平临界值: {critical_values[2]:.4f}")
    
    # 绘制p值分布比较
    plt.figure(figsize=(12, 5))
    
    # 子图1: p值分布比较
    plt.subplot(1, 2, 1)
    plt.hist(p_values_known, bins=50, alpha=0.7, color='blue', 
             density=True, label='参数已知')
    plt.hist(p_values_estimated_wrong, bins=50, alpha=0.7, color='red', 
             density=True, label='参数估计')
    plt.axvline(0.05, color='green', linestyle='--', label='α=0.05')
    plt.xlabel('p值', fontsize=24)
    plt.ylabel('密度', fontsize=24)
    plt.title('p值分布比较', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # 子图2: KS统计量分布比较
    plt.subplot(1, 2, 2)
    plt.hist(ks_stats_known, bins=50, alpha=0.7, color='blue', 
             density=True, label='参数已知')
    plt.hist(ks_stats_estimated, bins=50, alpha=0.7, color='red', 
             density=True, label='参数估计')
    
    # 添加理论临界值（对于参数已知的情况）
    from scipy.stats import kstwobign
    critical_theory = kstwobign.ppf(0.95) / np.sqrt(n)
    print(f"\n参数已知情况下K-S统计量的理论临界值: {critical_theory:.4f}")
    plt.axvline(critical_theory, color='blue', linestyle='--', 
                label='参数已知临界值')
    
    # 添加经验临界值（对于参数估计的情况）
    print(f"\n参数估计情况下K-S统计量的经验临界值: {critical_values[1]:.4f}")
    plt.axvline(critical_values[1], color='red', linestyle='--', 
                label='参数估计临界值')
    
    # 添加原始KS统计量
    plt.axvline(ks_stat_original, color='green', linestyle='--',
                label='10°原始K-S统计量')
    
    # 添加原始统计量
    plt.axvline(0.1938, color='darkgreen', linestyle='--',
                label='15°原始K-S统计量')
    
    plt.xlabel('K-S统计量', fontsize=24)
    plt.ylabel('密度', fontsize=24)
    plt.title('K-S统计量分布比较', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 计算第一类错误率
    alpha = 0.05
    type1_error_known = np.mean(np.array(p_values_known) < alpha)
    type1_error_wrong = np.mean(np.array(p_values_estimated_wrong) < alpha)
    
    print(f"\n在α={alpha}水平下的第一类错误率:")
    print(f"  参数已知: {type1_error_known:.4f}")
    print(f"  参数估计: {type1_error_wrong:.4f}")
    print(f"  偏差: {type1_error_wrong - type1_error_known:.4f}")