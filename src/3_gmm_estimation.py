from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
import numpy as np

def fit_gmm(data, n_components=2, random_state=None):
    """
    拟合高斯混合模型
    """
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state
    )
    gmm.fit(data.reshape(-1, 1))
    
    return gmm

def gmm_parameters(gmm):
    """
    提取GMM参数
    """
    params = {
        'weights': gmm.weights_,
        'means': gmm.means_.flatten(),
        'covariances': gmm.covariances_.flatten(),
        'std_devs': np.sqrt(gmm.covariances_.flatten())
    }
    return params

def bootstrap_gmm_estimation(data, n_components=2, n_bootstrap=1000, random_state=48):
    """
    使用Bootstrap方法估计GMM参数的方差和偏差
    """
    
    # 原始数据拟合
    original_gmm = fit_gmm(data, n_components, random_state)
    original_params = gmm_parameters(original_gmm)
    
    print("2组件GMM参数:")
    for i in range(2):
        print(f"高斯分布 {i+1}:")
        print(f"  权重: {original_params['weights'][i]:.4f}")
        print(f"  均值: {original_params['means'][i]:.4f}")
        print(f"  方差: {original_params['covariances'][i]:.4f}")
    
    # 存储Bootstrap样本的参数
    bootstrap_params = {
        'weights': [],
        'means': [],
        'covariances': [],
        'std_devs': []
    }
    
    print(f"进行Bootstrap重抽样，样本量: {n_bootstrap}")
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{n_bootstrap} 次Bootstrap抽样")
        
        # 重抽样
        bootstrap_sample = resample(data, random_state=random_state + i)
        # 拟合GMM
        try:
            bootstrap_gmm = fit_gmm(bootstrap_sample, n_components, random_state + i)
            bootstrap_param = gmm_parameters(bootstrap_gmm)
            
            # 对成分进行排序（按权重升序），避免标签交换问题
            sorted_indices = np.argsort(bootstrap_param['weights'])[::-1]
            
            for key in ['weights', 'means', 'covariances', 'std_devs']:
                bootstrap_params[key].append(bootstrap_param[key][sorted_indices])
                
        except Exception as e:
            print(f"第 {i+1} 次Bootstrap拟合失败: {e}")
            continue
    
    sorted_indices_original = np.argsort(original_params['weights'])[::-1]
    for key in ['weights', 'means', 'covariances', 'std_devs']:
        original_params[key] = original_params[key][sorted_indices_original]

    # 计算统计量
    results = {}
    for key in bootstrap_params.keys():
        bootstrap_array = np.array(bootstrap_params[key])
        
        # 基本统计量
        bootstrap_mean = np.mean(bootstrap_array, axis=0)
        bootstrap_std = np.std(bootstrap_array, axis=0, ddof=1)
        bootstrap_var = np.var(bootstrap_array, axis=0, ddof=1)
        
        # 偏差 = Bootstrap均值 - 原始估计值
        bias = bootstrap_mean - original_params[key]
        
        # 置信区间 (95%)
        ci_lower = np.percentile(bootstrap_array, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_array, 97.5, axis=0)
        
        results[key] = {
            'original': original_params[key],
            'bootstrap_mean': bootstrap_mean,
            'bootstrap_std': bootstrap_std,
            'bootstrap_var': bootstrap_var,
            'bias': bias,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'relative_bias': np.abs(bias / original_params[key]) * 100,  # 相对偏差百分比
            'cv': bootstrap_std / np.abs(bootstrap_mean) * 100  # 变异系数百分比
        }
    
    return original_gmm, original_params, bootstrap_params, results