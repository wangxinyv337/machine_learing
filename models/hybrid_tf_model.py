import torch
import torch.nn as nn
import torch.fft

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class HybridTimeFrequencyTransformer(nn.Module):
    def __init__(self, 
                 input_dim,
                 input_len=90,  # 添加输入序列长度
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 output_len=90,
                 dropout=0.1,
                 # 频域参数
                 freq_method='amplitude',       # ['amplitude','phase','both']
                 use_freq_pooling=True,         # 是否池化频率分量
                 retain_freq_ratio=0.5,         # 保留的频率分量比例
                 # 时域参数
                 time_pooling='mean',           # ['mean','max','last']
                 pos_encoder_enabled=True,      # 是否启用位置编码
                 # 融合参数
                 fusion_strategy='concat',      # ['concat','add','gate']
                 # 投影层增强
                 hidden_ratio=2                # projection层隐藏单元比例
                ):
        super(HybridTimeFrequencyTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_len = output_len
        self.d_model = d_model
        self.input_len = input_len  # 保存输入序列长度
        
        # ==== 时域分支 ====
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder_enabled = pos_encoder_enabled
        if self.pos_encoder_enabled:
            self.pos_encoder = PositionalEncoding(d_model)
        else:
            self.pos_encoder = nn.Identity()
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dropout=dropout,
                                                   batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                        num_layers=num_layers)
        self.time_pooling = time_pooling
        
        # ==== 频域分支 ====
        self.freq_method = freq_method
        self.use_freq_pooling = use_freq_pooling
        self.retain_freq_ratio = retain_freq_ratio
        
        # 计算保留的频率分量数量
        self.num_retain_freqs = max(1, int(input_len * retain_freq_ratio))
        
        # 确定频域输入维度
        if freq_method == 'both':
            self.freq_in_dim = input_dim * 2
        else:
            self.freq_in_dim = input_dim
        
        # 计算展平后的维度
        if use_freq_pooling:
            proj_input_dim = self.freq_in_dim
        else:
            proj_input_dim = self.num_retain_freqs * self.freq_in_dim
        
        self.freq_proj = nn.Sequential(
            nn.Linear(proj_input_dim, d_model * hidden_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_ratio, d_model)
        )
        
        # ==== 融合分支 ====
        self.fusion_strategy = fusion_strategy
        if fusion_strategy == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion_strategy == 'gate':
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        
        # ==== 输出层 ====
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len)
        )

    def forward(self, x):
        """
        x: [batch_size, input_len, input_dim]
        """
        batch_size, input_len, input_dim = x.shape
        
        # ==== 时域分支 ====
        x_time = self.input_proj(x)                     # [B, T, d_model]
        x_time = self.pos_encoder(x_time)               # 应用位置编码
        x_time = x_time.permute(1, 0, 2)                # [T, B, d_model] for Transformer
        
        # 保存原始长度用于频域分支
        orig_len = x_time.size(0)
        
        x_time = self.transformer_encoder(x_time)       # [T, B, d_model]
        
        # 时域特征池化
        if self.time_pooling == 'mean':
            x_time = torch.mean(x_time, dim=0)          # [B, d_model]
        elif self.time_pooling == 'max':
            x_time = torch.max(x_time, dim=0)[0]        # [B, d_model]
        else:  # 'last'
            x_time = x_time[-1]                         # [B, d_model]
        
        # ==== 频域分支 ====
        # FFT变换 - 沿时间维度
        x_freq = torch.fft.fft(x, dim=1)                # [B, T, D], complex
        
        # 特征提取策略
        if self.freq_method == 'amplitude':
            x_freq = torch.abs(x_freq)                   # 幅度谱
        elif self.freq_method == 'phase':
            x_freq = torch.angle(x_freq)                 # 相位谱
        else:  # 'both'
            # 合并幅度和相位
            amplitude = torch.abs(x_freq)
            phase = torch.angle(x_freq)
            x_freq = torch.cat([amplitude, phase], dim=-1)  # [B, T, 2*D]
        
        # 只保留低频部分
        x_freq = x_freq[:, :self.num_retain_freqs, :]   # [B, num_retain_freqs, D/2*D]
        
        # 池化或展平处理
        if self.use_freq_pooling:
            # 沿频率维度平均池化
            x_freq = torch.mean(x_freq, dim=1)          # [B, D/2*D]
        else:
            # 展平所有频率分量
            x_freq = x_freq.reshape(batch_size, -1)     # [B, num_retain_freqs * D/2*D]
        
        # 投影到模型维度
        x_freq = self.freq_proj(x_freq)                 # [B, d_model]
        
        # ==== 融合分支 ====
        if self.fusion_strategy == 'concat':
            fusion = torch.cat([x_time, x_freq], dim=-1)    # [B, 2*d_model]
            fusion = self.fusion_proj(fusion)               # [B, d_model]
        elif self.fusion_strategy == 'add':
            fusion = x_time + x_freq                        # [B, d_model]
        else:  # 'gate'
            gate = self.fusion_gate(torch.cat([x_time, x_freq], dim=-1))
            fusion = gate * x_time + (1 - gate) * x_freq    # [B, d_model]
        
        # ==== 输出层 ====
        out = self.decoder(fusion)                      # [B, output_len]
        return out



"""
import torch
import torch.nn as nn
import torch.fft

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class HybridTimeFrequencyTransformer(nn.Module):
    def __init__(self, 
                 input_dim,
                 input_len=90,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 output_len=90,
                 dropout=0.1):
        super(HybridTimeFrequencyTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_len = output_len
        self.d_model = d_model
        self.input_len = input_len
        
        # 固定参数设置
        freq_method = 'phase'
        use_freq_pooling = False
        retain_freq_ratio = 0.25
        time_pooling = 'last'
        pos_encoder_enabled = True
        fusion_strategy = 'concat'
        hidden_ratio = 2

        # ==== 时域分支 ====
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)  # 始终启用
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=False,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ==== 频域分支 ====
        # 保留的低频分量数量
        self.num_retain_freqs = max(1, int(input_len * retain_freq_ratio))
        
        # 确定频域输入维度（仅相位谱）
        self.freq_in_dim = input_dim
        
        # 展平后的维度（无池化）
        proj_input_dim = self.num_retain_freqs * self.freq_in_dim
        
        self.freq_proj = nn.Sequential(
            nn.Linear(proj_input_dim, d_model * hidden_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_ratio, d_model)
        )
        
        # ==== 融合分支 ====
        # 仅实现'concat'融合策略
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        
        # ==== 输出层 ====
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len)
        )

    def forward(self, x):
        batch_size, input_len, input_dim = x.shape
        
        # ==== 时域分支 ====
        x_time = self.input_proj(x)                     # [B, T, d_model]
        x_time = self.pos_encoder(x_time)                # 位置编码
        x_time = x_time.permute(1, 0, 2)                 # [T, B, d_model]
        x_time = self.transformer_encoder(x_time)        # [T, B, d_model]
        
        # 时域池化（仅取最后时间步）
        x_time = x_time[-1]                              # [B, d_model]
        
        # ==== 频域分支 ====
        # FFT变换
        x_freq = torch.fft.fft(x, dim=1)                 # [B, T, D], complex
        
        # 相位谱提取
        x_freq = torch.angle(x_freq)                     # 相位谱 [B, T, D]
        
        # 保留低频分量并展平
        x_freq = x_freq[:, :self.num_retain_freqs, :]    # [B, num_retain_freqs, D]
        x_freq = x_freq.reshape(batch_size, -1)          # [B, num_retain_freqs * D]
        
        # 频域投影
        x_freq = self.freq_proj(x_freq)                  # [B, d_model]
        
        # ==== 融合分支 ====
        # 拼接融合
        fusion = torch.cat([x_time, x_freq], dim=-1)     # [B, 2*d_model]
        fusion = self.fusion_proj(fusion)                # [B, d_model]
        
        # ==== 输出层 ====
        out = self.decoder(fusion)                       # [B, output_len]
        return out

"""