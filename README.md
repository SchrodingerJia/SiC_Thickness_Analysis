# 基于极值分析的红外干涉法测量碳化硅外延层厚度

## 摘要

本研究围绕红外干涉法测量碳化硅外延层厚度这一核心问题，系统性地开展了从物理模型建立、厚度分布建模到可靠性分析的全流程研究。首先基于双光束干涉原理建立了外延层厚度与干涉极值点波数的数学关系模型，通过极值点分析构建了入射角分别为 10° 和 15° 的厚度样本数据集。针对厚度分布的非正态特性，采用两成分高斯混合模型进行精细建模，通过 EM 算法估计模型参数，并利用 Bootstrap 重抽样方法评估参数的不确定性。进一步通过 K-S 拟合优度检验、统计功效分析和交叉验证等方法系统评估了测量结果的可靠性。

研究结果表明，碳化硅外延层厚度分布呈现显著的非正态特征，两成分高斯混合模型能够有效描述其分布规律，其中主成分（权重 75%-80%）对应理想测量条件下的厚度数据，次要成分反映系统性偏差。厚度估计值在 10° 和 15° 入射角下分别为 7.5043μm 和 7.4770μm，95% 置信区间分别为 [7.4787, 7.5372]μm 和 [7.4252, 7.5229]μm。Permutation Bootstrap 检验（p=0.1550）和 Bootstrap 置信区间分析表明，不同入射角条件下的厚度估计值无显著系统性差异，验证了测量结果的一致性和可靠性。

本文建立了一套完整的碳化硅外延层厚度红外干涉测量统计分析方法体系，为半导体制造工艺中的薄膜厚度表征提供了可靠的技术方案和理论支撑。

**关键词**：碳化硅外延层；红外干涉法；厚度测量；高斯混合模型；EM 算法；Bootstrap

## 项目结构

```
.
├── README.md               # 项目说明文档
├── requirements.txt        # 项目依赖包
├── data/                   # 数据目录
│   ├── incident_angle_10.json  # 10°入射角厚度数据
│   ├── incident_angle_10.xlsx  # 10°入射角原始光谱数据
│   ├── incident_angle_15.json  # 15°入射角厚度数据
│   └── incident_angle_15.xlsx  # 15°入射角原始光谱数据
└── src/                    # 源代码目录
    ├── 1_physical_model.py     # 物理模型建立与厚度计算
    ├── 2_data_processing.py    # 数据处理与正态性检验
    ├── 3_gmm_estimation.py     # 高斯混合模型参数估计
    └── 4_reliability_analysis.py # 可靠性分析与验证
```

## 功能模块说明

### 1. 物理模型建立 (`src/1_physical_model.py`)
- 实现碳化硅外延层和衬底的折射率计算
- 基于Fresnel公式计算反射系数和透射系数
- 实现光谱寻峰算法，提取干涉极值点
- 根据双光束干涉原理计算外延层厚度

### 2. 数据处理 (`src/2_data_processing.py`)
- 实现正态分布K-S拟合优度检验
- 分析参数估计对检验统计量的影响
- 提供数据可视化功能，包括直方图、分位数比较等

### 3. 高斯混合模型估计 (`src/3_gmm_estimation.py`)
- 实现两成分高斯混合模型参数估计
- 采用EM算法进行参数优化
- 利用Bootstrap方法评估参数不确定性

### 4. 可靠性分析 (`src/4_reliability_analysis.py`)
- 实现Permutation Bootstrap检验
- 进行统计功效分析
- 执行交叉验证评估模型稳定性

## 使用方法

### 环境配置
```bash
# 安装依赖包
pip install -r requirements.txt
```

### 运行示例
```python
# 导入物理模型模块
from src.1_physical_model import findpeaks, caculate_d

# 分析10°入射角数据
peak_points, valley_points = findpeaks(excel_file='data/incident_angle_10.xlsx')
thickness_list = caculate_d(wvnums, theta=10)

# 进行正态性检验
from src.2_data_processing import normality_ks_test
results = normality_ks_test(thickness_list)
```

## 数据说明

- `data/incident_angle_10.xlsx`: 10°入射角下的原始红外干涉光谱数据
- `data/incident_angle_15.xlsx`: 15°入射角下的原始红外干涉光谱数据
- `data/incident_angle_10.json`: 从10°数据计算得到的厚度值列表
- `data/incident_angle_15.json`: 从15°数据计算得到的厚度值列表

## 依赖包

主要依赖的Python包如下：
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- seaborn >= 0.11.0

详细依赖见`requirements.txt`文件。

## 许可证

本项目仅供学术研究使用。