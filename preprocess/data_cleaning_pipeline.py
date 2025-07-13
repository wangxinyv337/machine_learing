import pandas as pd
import os
import numpy as np
from data_cleaning_utils import *

os.makedirs('scalers', exist_ok=True)

# 指定列名
column_names = [
    "DateTime", "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
]

# 定义数值列的数据类型映射
numeric_dtypes = {
    "Global_active_power": 'float64',
    "Global_reactive_power": 'float64',
    "Voltage": 'float64',
    "Global_intensity": 'float64',
    "Sub_metering_1": 'float64',
    "Sub_metering_2": 'float64',
    "Sub_metering_3": 'float64',
    "RR": 'float64',
    "NBJRR1": 'float64',
    "NBJRR5": 'float64',
    "NBJRR10": 'float64',
    "NBJBROU": 'float64'
}

# 预处理函数，将文件中的特殊字符替换为NaN
def preprocess_csv(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace('?', str(np.nan))
    with open(file_path, 'w') as file:
        file.write(content)

# 对训练集文件进行预处理 <<<<<<<<
preprocess_csv('data/quchu_train.csv')
# 对测试集文件进行预处理 <<<<<<<<
preprocess_csv('data/quchu_test.csv')

# 读取训练集数据并指定数值列类型
train_df = pd.read_csv('data/quchu_train.csv', parse_dates=['DateTime'], dtype=numeric_dtypes)

# 读取测试集数据并指定数值列类型
test_df = pd.read_csv('data/quchu_test.csv', names=column_names, header=None, parse_dates=['DateTime'], dtype=numeric_dtypes)

# 数据类型检查函数
def check_data_types(df, df_name):
    print(f"\n{df_name}数据类型检查:")
    print(df.dtypes)
    for col in numeric_dtypes.keys():
        non_numeric = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
        if non_numeric.sum() > 0:
            print(f"警告: {col}列包含{non_numeric.sum()}个无法转换为数值的值")
            print(f"示例: {df.loc[non_numeric, col].head().tolist()}")
    return df

# 执行数据类型检查
train_df = check_data_types(train_df, "训练集")
test_df = check_data_types(test_df, "测试集")

# 时间字段拆解和日期归类
train_df['Date'] = train_df['DateTime'].dt.date
test_df['Date'] = test_df['DateTime'].dt.date

# 构造新变量（添加数据类型强制转换）
for df in [train_df, test_df]:
    # 确保参与计算的列都是数值类型
    for col in ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['gap_wh'] = df['Global_active_power'] * 1000 / 60
    df['sub_metering_total'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    df['Sub_metering_remainder'] = df['gap_wh'] - df['sub_metering_total']

# 日聚合处理
agg_methods = {
    'Global_active_power': 'sum',
    'Global_reactive_power': 'sum',
    'Voltage': 'mean',
    'Global_intensity': 'mean',
    'Sub_metering_1': 'sum',
    'Sub_metering_2': 'sum',
    'Sub_metering_3': 'sum',
    'Sub_metering_remainder': 'sum',
    'RR': 'first',
    'NBJRR1': 'first',
    'NBJRR5': 'first',
    'NBJRR10': 'first',
    'NBJBROU': 'first'
}

train_day = train_df.groupby('Date').agg(agg_methods)
test_day = test_df.groupby('Date').agg(agg_methods)

target_cols = train_day.columns.tolist()
# 定义需要标准化的列，排除目标列（如 Global_active_power）
#target_cols = [col for col in train_day.columns if col != 'Global_active_power']


# ==== 关键修改点1：异常值处理 ====
# 训练集：计算IQR阈值并保存
#train_day, train_stats = remove_outliers_iqr(train_day, target_cols)
# 测试集：复用训练集的阈值
#test_day = remove_outliers_iqr(test_day, target_cols, train_stats)

# 缺失值填充（保持不变）
train_day = check_and_fill_missing(train_day)
test_day = check_and_fill_missing(test_day)

# 标准化（只用训练集拟合）
#train_scaled, test_scaled = normalize_with_train(train_day, test_day, target_cols)


target_col = 'Global_active_power'
all_feature_cols = [col for col in train_day.columns if col != target_col]
train_scaled, test_scaled = normalize_with_train(
    train_day, test_day, 
    cols=all_feature_cols, 
    method='minmax', 
    scaler_path='scalers/full_scaler.pkl',
    target_col='Global_active_power',
    target_scaler_path='scalers/target_scaler.pkl'
)


# 保存处理好的数据

train_scaled.to_csv('data/cleaned6_train.csv')
test_scaled.to_csv('data/cleaned6_test.csv')

# 直接保存原始值，不进行标准化
"""
train_day.to_csv('data/cleanedRevIN_train.csv')
test_day.to_csv('data/cleanedRevIN_test.csv')
"""
print("[+] 清洗后数据已保存至 data/cleaned6_train.csv 与 data/cleaned6_test.csv")