import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PowerDataset(Dataset):
    def __init__(self, df, input_len=90, output_len=90, target_col='Global_active_power'):
        """
        :param df: 已归一化并清洗的 DataFrame（按日期排序）
        :param input_len: 输入序列长度（例如 90 天）
        :param output_len: 预测长度（例如 90 或 365 天）
        :param target_col: 目标预测变量（默认预测全局有功功率）
        """
        self.df = df.copy()
        self.input_len = input_len
        self.output_len = output_len
        self.target_col = target_col

        self.features = self.df.columns.tolist()
        self.feature_dim = len(self.features)
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        total_len = self.input_len + self.output_len
        for i in range(len(self.df) - total_len + 1):
            input_seq = self.df.iloc[i:i + self.input_len].values
            target_seq = self.df[self.target_col].iloc[i + self.input_len: i + self.input_len + self.output_len].values
            samples.append((input_seq, target_seq))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
