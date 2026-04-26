import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 配置环境：解决 VS Code 中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False 

# 任务1 目标变量的描述性统计分析
df = pd.read_csv('C:/Users/lenovo/OneDrive/Documents/作业/统计/housing.csv')

print("任务f1:目标变量(median_house_value)描述性统计")
print(df['median_house_value'].describe())

# 任务2 可视化分析及异常值处理
# 箱线图-识别异常值
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='median_house_value', color='skyblue')
plt.title("房价分布箱线图 (用于识别离群点)")
plt.show()

#热力图-分析特征相关性
plt.figure(figsize=(12, 8))
# 仅计算数值列的相关性
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("特征变量相关性热力图")
plt.show()

# 异常值处理：根据箱线图结果，剔除房价等于或超过500001的封顶值数据
df_clean = df[df['median_house_value'] < 500001].copy()
print(f"\n--- 任务 2: 异常值处理完成，剩余样本量: {len(df_clean)} ---")

# 任务3 思考并执行归一化或标准化
# 处理缺失值用中位数填充并准备特征
# 剔除无法直接计算的类别变量，只保留数值特征
X = df_clean.select_dtypes(include=[np.number]).drop('median_house_value', axis=1)
X = X.fillna(X.median())
y = df_clean['median_house_value']

# 由于后续使用PCA（主成分分析），PCA对数据量纲高度敏感，
# 因此必须使用标准化，使特征均值为0，方差为1。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n--- 任务 3: 数据标准化完成 ---")

# 【任务 4】使用 PCA 算法进行降维
# 设定保留 95% 的方差信息
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("--- 任务 4: PCA 降维完成 ---")
print(f"原始特征维度: {X_scaled.shape[1]}")
print(f"降维后主成分数: {X_pca.shape[1]}")

# 任务5 使用回归算法进行预测
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 创建线性回归模型
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# 预测
y_pred = reg_model.predict(X_test)

# 结果评估
print("--- 任务 5: 回归预测评估 ---")
print(f"R2 Score (拟合优度): {r2_score(y_test, y_pred):.4f}")
print(f"RMSE (预测误差): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# 预测结果可视化对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("实际房价")
plt.ylabel("预测房价")
plt.title("PCA回归预测结果: 实际 vs 预测")
plt.show()