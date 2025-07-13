import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_len=90, dropout_rate=0.3):
        super(LSTMForecast, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_len = output_len
        self.dropout_rate = dropout_rate

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers, 
            batch_first=True,
            dropout=(dropout_rate if num_layers > 1 else 0)  # 多层LSTM间的dropout
        )
        
        # LSTM输出后的dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        # x: [B, input_len, input_dim]
        # 通过LSTM层
        output, _ = self.lstm(x)  # output: [B, input_len, hidden_dim]
        
        # 取最后一个时间步的输出
        last_hidden = output[:, -1, :]  # [B, hidden_dim]
        
        # 应用dropout
        last_hidden = self.dropout(last_hidden)
        
        # 通过全连接层生成预测
        out = self.fc(last_hidden)  # [B, output_len]
        
        return out