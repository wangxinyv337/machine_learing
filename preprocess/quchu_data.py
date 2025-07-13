import pandas as pd

def check_insufficient_minute_data(df, datetime_col=None, min_minutes=1440, remove_incomplete=True):
    """
    检查并标记不是每分钟都有数据的日期

    参数:
    df: DataFrame，包含时间序列数据
    datetime_col: str，日期时间列的名称，若为None则自动检测
    min_minutes: int，定义完整日的最小分钟数阈值（默认24*60=1440）
    remove_incomplete: bool，是否删除不完整的日期数据

    返回:
    处理后的DataFrame和不完整日期的列表
    """
    # 自动检测日期时间列
    if datetime_col is None:
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
        if not datetime_cols:
            # 如果没有检测到日期时间类型的列，尝试将第一列转换
            datetime_col = df.columns[0]
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        else:
            datetime_col = datetime_cols[0]
        print(f"使用日期时间列: {datetime_col}")

    # 按日期对数据分组，方便后续按天检查分钟级数据完整性
    date_groups = df.groupby(df[datetime_col].dt.date)

    incomplete_dates = []
    for date, group in date_groups:
        # 生成对应日期从0点到23点59分的完整分钟索引
        full_minutes = pd.date_range(start=pd.Timestamp(date), end=pd.Timestamp(date) + pd.Timedelta(days=1), freq='T')[:-1]

        # 获取当前分组（即当天）实际存在的分钟数据，通过将时间列向下取整到分钟级别，并转换为集合去重
        existing_minutes = set(group[datetime_col].dt.floor('T'))

        # 计算当天缺少的分钟数
        missing_minutes = len(full_minutes) - len(existing_minutes)

        if missing_minutes > 0:
            incomplete_dates.append(date)
            print(f"日期 {date} 缺少 {missing_minutes} 分钟的数据")

    # 根据是否存在不完整日期添加标记列，True表示该日期数据完整，False表示不完整
    df['is_complete_day'] = ~df[datetime_col].dt.date.isin(incomplete_dates)

    # 根据参数决定是否删除不完整日期的数据
    if remove_incomplete and incomplete_dates:
        original_len = len(df)
        # 通过筛选标记列为True的数据行，实现删除不完整日期的数据
        df = df[df['is_complete_day']]
        removed_count = original_len - len(df)
        print(f"已删除 {removed_count} 行不完整日期的数据")

    return df, incomplete_dates


# 读取训练集数据（有列名）
train_df = pd.read_csv('data/train.csv')

# 读取测试集数据（无列名，使用训练集的列名）
test_df = pd.read_csv('data/test.csv', header=None, names=train_df.columns)

# 检查并处理数据完整性（默认删除不完整数据）
train_df, incomplete_train_dates = check_insufficient_minute_data(train_df)
test_df, incomplete_test_dates = check_insufficient_minute_data(test_df)

# 保存处理后的数据（测试集不保存表头，训练集保留表头）
train_df.to_csv('data/quchu_train.csv', index=False)
test_df.to_csv('data/quchu_test.csv', index=False, header=False)

# 输出结果
if incomplete_train_dates:
    print(f"训练集共发现 {len(incomplete_train_dates)} 个不完整日期，已删除相关数据")
else:
    print("训练集所有日期数据完整")

if incomplete_test_dates:
    print(f"测试集共发现 {len(incomplete_test_dates)} 个不完整日期，已删除相关数据")
else:
    print("测试集所有日期数据完整")

print(f"处理后的训练集: {len(train_df)} 行，保存至 data/quchu_train.csv")
print(f"处理后的测试集: {len(test_df)} 行，保存至 data/quchu_test.csv")