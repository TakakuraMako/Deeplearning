import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 读取数据
df = pd.read_excel('./深度学习/实践课/附件2.xlsx')

# 2. 基本信息查看
print("数据维度:", df.shape)
print("原始标签分布:\n", df.iloc[:, -1].value_counts())

# 3. 缺失值检查
print("缺失值统计:\n", df.isnull().sum())
