from scipy.stats import ks_2samp, kstest, chi2, norm, probplot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats

# 4. 拟合优度：两样本 KS（原始数据 vs 从 GMM 生成的样本）—— 使用 ks_2samp
# 说明：两样本 KS 不需要解析 CDF，只比较两个样本的 ECDF 差异
def ks_gof_via_sampling(data, gmm, n_samples=None, random_state=0):
    if n_samples is None:
        n_samples = len(data)
    # 生成与原数据等量的 GMM 样本
    gen_samples, _ = gmm.sample(n_samples)
    gen = gen_samples.flatten()
    stat, pvalue = ks_2samp(np.array(data).flatten(), gen)
    return stat, pvalue, gen

# 5. 统计功效分析（Power）
def power_analysis_vs_single_normal_from_data(data, n_list=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200], B=500, random_state=RNG_SEED):
    """
    功效分析：检验 H0 为单一正态（参数由 data 拟合），样本来自实际数据时拒绝 H0 的概率。
    """
    rng = np.random.RandomState(random_state)
    data = np.array(data).flatten()
    
    # H0: 单一正态参数（用实际数据均值和标准差）
    mu0 = np.mean(data)
    sigma0 = np.std(data, ddof=1)
    
    power_res = {}
    for n in n_list:
        rejects = 0
        for b in range(B):
            # 从实际数据中随机采样 n 个点（bootstrap）
            sample = rng.choice(data, size=n, replace=True)
            # KS 检验：H0 为单一正态 N(mu0, sigma0)
            ks_stat, pval = kstest(sample, 'norm', args=(mu0, sigma0))
            if pval < 0.05:
                rejects += 1
        power_res[n] = rejects / B
    return power_res

def analyze_bimodal_p_value_bias(data, n_simulations=10000, alpha=0.05):
    """
    分析双峰高斯混合分布在参数估计情况下的p值偏差
    """
    np.random.seed(42)
    n = len(data)

    # 估计原始双峰高斯混合分布参数
    mu1, sigma1, mu2, sigma2, w1, bic0 = fit_bimodal_gaussian(data)
    
    print(f"拟合的双峰高斯混合分布参数:")
    print(f"  第一个分布: $\mu={mu1:.4f}, σ={sigma1:.4f}, 权重={w1:.4f}")
    print(f"  第二个分布: μ={mu2:.4f}, σ={sigma2:.4f}, 权重={1-w1:.4f}")

    # 数据分布和拟合的双峰分布
    plt.figure(figsize=(12, 8))
    plt.hist(data, bins=50, density=True, alpha=0.8, color='lightblue', label='原始数据')
    
    # 绘制拟合的双峰分布
    x_range = np.linspace(np.min(data), np.max(data), 1000)
    pdf_values = w1 * stats.norm.pdf(x_range, mu1, sigma1) + (1-w1) * stats.norm.pdf(x_range, mu2, sigma2)
    plt.plot(x_range, pdf_values, 'r-', linewidth=3, label='拟合的高斯混合分布')
    
    # 绘制两个单独的高斯成分
    plt.plot(x_range, w1 * stats.norm.pdf(x_range, mu1, sigma1), 'g--', linewidth=2, alpha=0.7, label='高斯成分1')
    plt.plot(x_range, (1-w1) * stats.norm.pdf(x_range, mu2, sigma2), 'm--', linewidth=2, alpha=0.7, label='高斯成分2')
    
    # 绘制正态分布拟合曲线
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    x = np.linspace(min(data), max(data), 1000)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'b-', linewidth=2, label='拟合的正态分布')
    
    plt.xlabel('厚度值(μm)', fontsize=28)
    plt.ylabel('密度', fontsize=28)
    plt.legend(fontsize=24)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    p_values_known = []
    ks_stats_estimated = []
    ks_stats_known = []
    bic_estimated = []
    
    # 自定义CDF函数包装器
    def known_cdf(x):
        return bimodal_gaussian_cdf(x, mu1, sigma1, mu2, sigma2, w1)
    
    # 计算原始统计量
    ks_stat_original, p_original = stats.kstest(data, known_cdf)

    print(f"\n原始统计量:")
    print(f"  K-S统计量: {ks_stat_original:.4f}")
    print(f"  p值: {p_original:.4f}\n")
    
    for i in range(n_simulations):
        if (i+1) % 1000 == 0:
            print(f"p值偏差分析: 已完成 {i+1}/{n_simulations}")
        
        # 复合抽样法从真实分布生成样本
        n1 = np.random.binomial(n, w1)
        n2 = n - n1
        sample1 = np.random.normal(mu1, sigma1, n1)
        sample2 = np.random.normal(mu2, sigma2, n2)
        sample = np.concatenate([sample1, sample2])
        np.random.shuffle(sample)
        
        # 参数已知情况的KS统计量和p值
        ks_stat_known, _ = stats.kstest(sample, known_cdf)
        ks_stats_known.append(ks_stat_known)
        
        # 参数估计情况
        mu1_est, sigma1_est, mu2_est, sigma2_est, w1_est, bic_est = fit_bimodal_gaussian(sample)
        bic_estimated.append(bic_est)

        def estimated_cdf(x):
            return bimodal_gaussian_cdf(x, mu1_est, sigma1_est, mu2_est, sigma2_est, w1_est)
        
        ks_stat, _ = stats.kstest(sample, estimated_cdf)
        ks_stats_estimated.append(ks_stat)

    # 给出原始统计量在模拟中的位置
    print(f"\n原始统计量在模拟中的位置:{np.mean(ks_stats_estimated <= ks_stat_original):.4f}")
    
    # KS统计量分布比较
    plt.figure(figsize=(12, 8))
    plt.hist(ks_stats_known, bins=50, alpha=0.7, color='blue', 
             density=True, label='参数已知')
    plt.hist(ks_stats_estimated, bins=50, alpha=0.7, color='red', 
             density=True, label='参数估计')
    
    # 添加理论临界值（对于参数已知的情况）
    from scipy.stats import kstwobign
    critical_theory = kstwobign.ppf(1-alpha) / np.sqrt(n)
    plt.axvline(critical_theory, color='blue', linestyle='--', 
                label=f'参数已知临界值(α={alpha})')
    
    # 添加经验临界值（对于参数估计的情况）
    plt.axvline(np.percentile(ks_stats_estimated, [100*(1-alpha)])[0], color='red', linestyle='--', 
                label=f'参数估计临界值(α={alpha})')
    
    # 添加原始统计量
    plt.axvline(ks_stat_original, color='green', linestyle='--',
                label='原始K-S统计量')
    
    plt.xlabel('K-S统计量', fontsize=24)
    plt.ylabel('密度', fontsize=24)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return mu1, sigma1, mu2, sigma2, w1

def analyze_bimodal_p_value_bias_chi2(data, n_simulations=10000, alpha=0.05, bins=20):
    """
    分析双峰高斯混合分布在参数估计情况下的p值偏差（使用卡方检验）
    """
    np.random.seed(42)
    n = len(data)

    # 估计原始双峰高斯混合分布参数
    mu1, sigma1, mu2, sigma2, w1, bic0 = fit_bimodal_gaussian(data)
    
    print(f"拟合的双峰高斯混合分布参数:")
    print(f"  第一个分布: μ={mu1:.4f}, σ={sigma1:.4f}, 权重={w1:.4f}")
    print(f"  第二个分布: μ={mu2:.4f}, σ={sigma2:.4f}, 权重={1-w1:.4f}")

    # 数据分布和拟合的双峰分布
    plt.figure(figsize=(12, 8))
    plt.hist(data, bins=50, density=True, alpha=0.8, color='lightblue', label='原始数据')
    
    # 绘制拟合的双峰分布
    x_range = np.linspace(np.min(data), np.max(data), 1000)
    pdf_values = w1 * stats.norm.pdf(x_range, mu1, sigma1) + (1-w1) * stats.norm.pdf(x_range, mu2, sigma2)
    plt.plot(x_range, pdf_values, 'r-', linewidth=3, label='拟合的双峰分布')
    
    # 绘制两个单独的高斯成分
    plt.plot(x_range, w1 * stats.norm.pdf(x_range, mu1, sigma1), 'g--', linewidth=1.5, alpha=0.7, label='高斯成分1')
    plt.plot(x_range, (1-w1) * stats.norm.pdf(x_range, mu2, sigma2), 'm--', linewidth=1.5, alpha=0.7, label='高斯成分2')
    
    plt.xlabel('厚度值(μm)', fontsize=28)
    plt.ylabel('密度', fontsize=28)
    plt.legend(fontsize=24)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # 计算原始数据的卡方检验
    chi2_stat_original, p_original, obs_freq, exp_freq, bin_edges, dof = chi2_test_for_bimodal(
        data, mu1, sigma1, mu2, sigma2, w1, bins)
    
    print(f"\n原始统计量:")
    print(f"  卡方统计量: {chi2_stat_original:.4f}")
    print(f"  自由度: {dof}")
    print(f"  p值: {p_original:.4f}")

    # 绘制观察频率与期望频率的对比图
    plt.figure(figsize=(12, 8))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.8
    
    plt.bar(bin_centers, obs_freq, width=width, alpha=0.7, color='lightblue', label='观察频率')
    plt.bar(bin_centers, exp_freq, width=width*0.6, alpha=0.7, color='red', label='期望频率')
    
    plt.xlabel('厚度值(μm)', fontsize=24)
    plt.ylabel('频率', fontsize=24)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.show()

    chi2_stats_known = []
    chi2_stats_estimated = []
    p_values_estimated = []
    bic_estimated = []
    
    for i in range(n_simulations):
        if (i+1) % 1000 == 0:
            print(f"p值偏差分析: 已完成 {i+1}/{n_simulations}")
        
        # 复合抽样法从真实分布生成样本
        n1 = np.random.binomial(n, w1)
        n2 = n - n1
        sample1 = np.random.normal(mu1, sigma1, n1)
        sample2 = np.random.normal(mu2, sigma2, n2)
        sample = np.concatenate([sample1, sample2])
        np.random.shuffle(sample)
        
        # 参数已知情况的卡方检验
        chi2_stat_known, p_known, _, _, _, _ = chi2_test_for_bimodal(
            sample, mu1, sigma1, mu2, sigma2, w1, bins)
        chi2_stats_known.append(chi2_stat_known)
        
        # 参数估计情况
        mu1_est, sigma1_est, mu2_est, sigma2_est, w1_est, bic_est = fit_bimodal_gaussian(sample)
        bic_estimated.append(bic_est)

        chi2_stat_est, p_est, _, _, _, dof_est = chi2_test_for_bimodal(
            sample, mu1_est, sigma1_est, mu2_est, sigma2_est, w1_est, bins)
        chi2_stats_estimated.append(chi2_stat_est)
        p_values_estimated.append(p_est)

    # 给出原始统计量在模拟中的位置
    print(f"\n原始卡方统计量在模拟中的位置:{np.mean(chi2_stats_estimated <= chi2_stat_original):.4f}")
    
    # 卡方统计量分布比较
    plt.figure(figsize=(12, 8))
    plt.xlim(0,175)
    plt.ylim(0,0.22)
    plt.hist(chi2_stats_known, bins=50, alpha=0.7, color='blue', 
             density=True, label='参数已知')
    plt.hist(chi2_stats_estimated, bins=50, alpha=0.7, color='red', 
             density=True, label='参数估计')
    
    # 添加理论临界值（对于参数已知的情况）
    critical_theory = stats.chi2.ppf(1-alpha, dof)
    plt.axvline(critical_theory, color='blue', linestyle='--', 
                label=f'固定参数临界值(α={alpha})')
    
    # 添加经验临界值（对于参数估计的情况）
    plt.axvline(np.percentile(chi2_stats_estimated, [100*(1-alpha)])[0], color='red', linestyle='--', 
                label=f'估计参数临界值(α={alpha})')
    
    # 添加原始统计量
    plt.axvline(chi2_stat_original, color='green', linestyle='--',
                label='原始卡方统计量')
    
    plt.xlabel('卡方统计量', fontsize=24)
    plt.ylabel('密度', fontsize=24)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return mu1, sigma1, mu2, sigma2, w1, chi2_stat_original, p_original

# 6. 两样本比较（10° vs 15°）用 ks_2samp
def two_sample_ks(d1, d2):
    stat, pvalue = ks_2samp(np.array(d1).flatten(), np.array(d2).flatten())
    return stat, pvalue

def bootstrap_CI_mean_diff(data_10, data_15, gmm_10, gmm_15, B=2000):
    """
    Permutation Bootstrap方法估计均值差异 Δμ = μ1(10°) - μ1(15°)
    """
    # 观测统计量
    idx_10 = np.argmax(gmm_10.weights_)
    idx_15 = np.argmax(gmm_15.weights_)
    mu_10_obs = gmm_10.means_.flatten()[idx_10]
    mu_15_obs = gmm_15.means_.flatten()[idx_15]
    delta_obs = mu_10_obs - mu_15_obs
    
    # 从原始数据中直接重采样
    diffs = []
    n10, n15 = len(data_10), len(data_15)
    
    for _ in range(B):
        # 从原始数据中有放回地重采样
        sample_10 = np.random.choice(data_10, size=n10, replace=True)
        sample_15 = np.random.choice(data_15, size=n15, replace=True)
        
        # 对重采样数据拟合GMM
        gmm10_bs = GaussianMixture(n_components=2, random_state=_, max_iter=200)
        gmm15_bs = GaussianMixture(n_components=2, random_state=_+1000, max_iter=200)
        
        gmm10_bs.fit(sample_10.reshape(-1, 1))
        gmm15_bs.fit(sample_15.reshape(-1, 1))
        
        # 找到主要组分
        idx10_bs = np.argmax(gmm10_bs.weights_)
        idx15_bs = np.argmax(gmm15_bs.weights_)
        
        mu10_bs = gmm10_bs.means_.flatten()[idx10_bs]
        mu15_bs = gmm15_bs.means_.flatten()[idx15_bs]
        
        diffs.append(mu10_bs - mu15_bs)
    
    diffs = np.array(diffs)
    
    # 计算置信区间
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    
    return delta_obs, (ci_low, ci_high), diffs