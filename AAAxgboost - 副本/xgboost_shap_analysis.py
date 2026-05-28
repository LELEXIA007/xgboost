import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===================== 全局设置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
warnings.filterwarnings('ignore')
output_dir = os.path.dirname(os.path.abspath(__file__))


# -------------------- 数据加载 --------------------
def load_data(file_path):
    print(f"\n正在加载数据: {file_path}")
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    print(f"数据形状: {df.shape}")
    print("\n数据前5行:")
    print(df.head())
    print("\n数据列信息:")
    print(df.info())
    return df


# -------------------- 【✅ 只修这里：文本转数字，其他完全不变】 --------------------
def preprocess_data(df):
    print("\n开始数据预处理...")
    df = df.copy()
    df = df.dropna(axis=1, how='all')

    # 把所有特征强制转成数字（模型能看懂的格式）
    for col in df.columns:
        df[col] = df[col].replace(['—', '-', 'None', 'nan', ''], np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print(f"预处理完成 | 最终形状: {df.shape}")
    print("最终列名:", list(df.columns))
    return df


# -------------------- 分层抽样划分（完全不变） --------------------
def prepare_training_data(df):
    print("\n📦 准备训练数据...")
    target_col = df.columns[0]
    print(f"目标列: {target_col}")

    df = df.dropna(subset=[target_col])
    df = df[np.isfinite(df[target_col])]

    X = df.drop([target_col], axis=1)
    y = df[target_col]

    n_bins = 5
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=bins
    )

    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X, target_col


# -------------------- 【✅ 正常模型：能学习、不虚高、不负数】 --------------------
def train_xgboost_model(X_train, y_train):
    print("\n训练【高R²泛化版】XGBoost...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.5,
        min_child_weight=1,
        gamma=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("模型训练完成")
    return model


# -------------------- 模型评估（完全不变） --------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, target_col):
    print("\n评估模型性能...")

    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"===== 训练集性能 =====")
    print(f"R²: {r2_train:.4f}")
    print(f"RMSE: {rmse_train:.4f}")
    print(f"MAE: {mae_train:.4f}")

    print(f"\n===== 测试集性能 =====")
    print(f"R²: {r2_test:.4f}")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"MAE: {mae_test:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel(f"真实{target_col}")
    plt.ylabel(f"预测{target_col}")
    plt.title(f"性能预测 | 测试集R²={r2_test:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction.png'))
    plt.close()
    return y_test_pred


# ===================== SHAP 分析（100% 完全没变！） =====================
def shap_analysis(model, X_train, X_test, X):
    print("\n开始生成全套SHAP图像 + 单独保存数据...")
    X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)
    expected_value = explainer.expected_value

    print("1. 生成内置特征重要性图 + 保存数据")
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(12, 8))
    plt.barh([X.columns[i] for i in sorted_idx], feature_importance[sorted_idx])
    plt.title('特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    feat_df = pd.DataFrame({"特征": X.columns, "重要性": feature_importance})
    feat_df.to_excel(os.path.join(output_dir, "feature_importance.xlsx"), index=False)

    print("2. 生成SHAP全局摘要图 + 保存数据")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_all, feature_names=X.columns, show=False)
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()
    sum_df = pd.DataFrame(shap_values, columns=X.columns)
    sum_df.to_excel(os.path.join(output_dir, "shap_summary.xlsx"), index=False)

    print("3. 生成SHAP重要性条形图 + 保存数据")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_all, feature_names=X.columns, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_dir, 'shap_importance_bar.png'), bbox_inches='tight')
    plt.close()
    bar_df = pd.DataFrame({
        "特征": X.columns,
        "平均SHAP绝对值": np.abs(shap_values).mean(axis=0)
    })
    bar_df.to_excel(os.path.join(output_dir, "shap_importance_bar.xlsx"), index=False)

    print("5. 生成SHAP依赖图 + 保存数据")
    top5_idx = np.argsort(np.abs(shap_values).mean(0))[-5:]
    for i, idx in enumerate(top5_idx):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(idx, shap_values, X_all, feature_names=X.columns, show=False)
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{i + 1}.png'), bbox_inches='tight')
        plt.close()
        dep_df = pd.DataFrame({
            X.columns[idx]: X_all.iloc[:, idx].values,
            "SHAP值": shap_values[:, idx]
        })
        dep_df.to_excel(os.path.join(output_dir, f"shap_dependence_{i+1}.xlsx"), index=False)

    print("6. 生成SHAP交互热图 + 保存数据 【已放大字体+PPT尺寸】")
    feature_corr = np.corrcoef(shap_values.T)
    # ====================== 这里放大 ======================
    plt.figure(figsize=(18, 16))  # PPT大尺寸
    sns.heatmap(feature_corr, annot=True, cmap='coolwarm', fmt='.2f',
                xticklabels=X.columns, yticklabels=X.columns,
                annot_kws={"size": 16, "weight": "bold"},  # 数值字体超大
                cbar_kws={"shrink": 0.8})
    plt.title('特征交互作用热图', fontsize=22, weight='bold', pad=20)
    plt.xticks(fontsize=14, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_interaction.png'))
    plt.close()
    inter_df = pd.DataFrame(feature_corr, index=X.columns, columns=X.columns)
    inter_df.to_excel(os.path.join(output_dir, "shap_interaction.xlsx"), index=True)

    print("7. 生成SHAP样本热力图 + 保存数据")
    plt.figure(figsize=(14, 10))
    shap.plots.heatmap(shap.Explanation(
        values=shap_values[:100], base_values=expected_value,
        data=X_all.iloc[:100], feature_names=X.columns
    ), show=False)
    plt.savefig(os.path.join(output_dir, 'shap_heatmap.png'), bbox_inches='tight')
    plt.close()
    heat_df = pd.DataFrame(shap_values[:100], columns=X.columns)
    heat_df.to_excel(os.path.join(output_dir, "shap_heatmap.xlsx"), index=False)

    print("所有SHAP图像 + 对应Excel数据保存完成！")


# -------------------- 特征相关性矩阵（已放大字体+PPT尺寸） --------------------
def feature_correlation_analysis(X):
    print("\n特征相关性分析... 【已放大字体+PPT尺寸】")
    corr_matrix = X.corr()
    # ====================== 这里放大 ======================
    plt.figure(figsize=(18, 16))  # PPT专用尺寸
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                annot_kws={"size": 16, "weight": "bold"},  # 相关系数字体超大
                cbar_kws={"shrink": 0.8})
    plt.title('特征相关系数矩阵', fontsize=22, weight='bold', pad=20)
    plt.xticks(fontsize=14, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    return corr_matrix


# ===================== 主函数（完全不变） =====================
def main():
    try:
        file_path = 'all_tables.xlsx'
        print("=" * 60)
        print("有机光伏 XGBoost + SHAP 最优R²版")
        print("=" * 60)

        df = load_data(file_path)
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test, X, target_col = prepare_training_data(df)
        model = train_xgboost_model(X_train, y_train)
        evaluate_model(model, X_train, y_train, X_test, y_test, target_col)
        shap_analysis(model, X_train, X_test, X)
        feature_correlation_analysis(X)

        print("\n全部分析完成！所有图像 + SHAP数据已保存！")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()