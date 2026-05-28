import sys
import os

# 禁用numba的AVX检测，以避免llvmlite兼容性问题
os.environ['NUMBA_ENABLE_AVX'] = '0'

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=== 简化版测试脚本 ===")

# 文件路径
file_path = 'all_tables - 副本 (2).xlsx'

# 1. 加载数据
print("\n1. 加载数据...")
df = pd.read_excel(file_path)
print(f"数据形状: {df.shape}")
print("数据前5行:")
print(df.head())

# 2. 数据预处理
print("\n2. 数据预处理...")

# 处理每一列
for col in df.columns:
    print(f"处理列: {col}")
    # 首先将"—"替换为NaN
    df[col] = df[col].replace('—', np.nan)
    
    # 尝试将列转换为数值类型
    try:
        # 去除可能的空格
        df[col] = df[col].astype(str).str.strip()
        # 替换空字符串为NaN
        df[col] = df[col].replace('', np.nan)
        # 转换为数值类型
        df[col] = pd.to_numeric(df[col], errors='ignore')
    except Exception as e:
        print(f"转换列 {col} 为数值类型时出错: {e}")

# 再次检查列类型并处理
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        # 对于数值列，填充缺失值为均值
        if df[col].isnull().sum() > 0:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
    else:
        # 对于非数值列，填充缺失值为众数
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

# 处理分类变量（非数值列）
categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
if len(categorical_cols) > 0:
    print(f"分类变量: {categorical_cols}")
    from sklearn.preprocessing import LabelEncoder
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 确保所有列都是数值类型
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # 再次填充可能产生的NaN
    if df[col].isnull().sum() > 0:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)

# 3. 准备训练数据
print("\n3. 准备训练数据...")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(f"特征数量: {X.shape[1]}")
print(f"目标变量分布:")
print(y.value_counts())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 4. 训练XGBoost模型
print("\n4. 训练XGBoost模型...")
# 确定任务类型
if len(np.unique(y_train)) <= 2:
    objective = 'binary:logistic'
    eval_metric = 'logloss'
else:
    objective = 'multi:softmax'
    eval_metric = 'mlogloss'

# 创建XGBoost分类器
model = xgb.XGBClassifier(
    objective=objective,
    eval_metric=eval_metric,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)
print("模型训练完成")

# 5. 评估模型
print("\n5. 评估模型...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

print("\n测试完成！")