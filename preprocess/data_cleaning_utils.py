import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import joblib

def check_and_fill_missing(df):
    """缺失值线性插值（时间序列适用）"""
    if df.isnull().sum().sum() > 0:
        df = df.interpolate(method='linear', limit_direction='both')
    return df


"""
def remove_outliers_iqr(df, cols, train_stats=None):
    df_cleaned = df.copy()
    stats = {}
    
    for col in cols:
        if train_stats:  # 测试集模式：复用训练集的统计量
            lower, upper = train_stats[col]['lower'], train_stats[col]['upper']
        else:  # 训练集模式：计算统计量
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            stats[col] = {'lower': lower, 'upper': upper}
        
        # 异常值替换为NaN
        df_cleaned[col] = df[col].mask((df[col] < lower) | (df[col] > upper))
    
    return (df_cleaned, stats) if not train_stats else df_cleaned


def normalize_with_train(df_train, df_test, cols, scaler_path='scalers/standard_scaler.pkl'):
    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    df_train_scaled[cols] = scaler.fit_transform(df_train[cols])
    df_test_scaled[cols] = scaler.transform(df_test[cols])
    
    joblib.dump(scaler, scaler_path)
    print(f"[+] Scaler saved to {scaler_path}")
    return df_train_scaled, df_test_scaled
"""


def remove_outliers_iqr(df, cols, train_stats=None):
    """
    基于IQR处理异常值，并输出每列异常值数量
    :param df: 输入数据
    :param cols: 需要处理的列
    :param train_stats: 训练集的统计量（测试集需传入）
    :return: 处理后的数据，训练集的统计量（仅训练模式返回）
    """
    df_cleaned = df.copy()
    stats = {}
    
    print("\n[异常值统计]")
    
    for col in cols:
        if train_stats:  # 测试集：使用训练集的阈值
            lower, upper = train_stats[col]['lower'], train_stats[col]['upper']
        else:  # 训练集：计算 IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            stats[col] = {'lower': lower, 'upper': upper}
        
        # 标记异常值
        outliers = (df[col] < lower) | (df[col] > upper)
        outlier_count = outliers.sum()
        print(f"{col}: 异常值数量 = {outlier_count}")
        
        # 异常值设为 NaN
        df_cleaned[col] = df[col].mask(outliers)

    return (df_cleaned, stats) if not train_stats else df_cleaned


def normalize_with_train(df_train, df_test, cols, method='standard', scaler_path='scalers/scaler.pkl', 
                         target_col=None, target_scaler_path='scalers/target_scaler.pkl'):
    """支持多种归一化方式 + 可选目标列单独归一化保存"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    elif method == 'maxabs':
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
    else:
        raise ValueError(f"未知归一化方法：{method}")

    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    # 全部列归一化
    df_train_scaled[cols] = scaler.fit_transform(df_train[cols])
    df_test_scaled[cols] = scaler.transform(df_test[cols])
    joblib.dump(scaler, scaler_path)
    print(f"[+] Scaler ({method}) saved to {scaler_path}")
    
    # 如果设置了目标列，则单独对目标列保存 scaler
    if target_col is not None:
        target_scaler = MinMaxScaler()
        df_train_scaled[target_col] = target_scaler.fit_transform(df_train[[target_col]])
        df_test_scaled[target_col] = target_scaler.transform(df_test[[target_col]])
        joblib.dump(target_scaler, target_scaler_path)
        print(f"[+] 目标列 {target_col} 的 MinMaxScaler 已保存到 {target_scaler_path}")
    
    return df_train_scaled, df_test_scaled
