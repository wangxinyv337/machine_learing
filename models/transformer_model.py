import torch
import torch.nn as nn

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

class TransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_len=90, dropout=0.1):
        super(TransformerForecast, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_len)
        self.output_len = output_len

    def forward(self, x):
        # x shape: [batch_size, input_len, input_dim]
        x = self.input_proj(x)  # [batch_size, input_len, d_model]
        x = self.pos_encoder(x)  # [batch_size, input_len, d_model]
        x = x.permute(1, 0, 2)  # [input_len, batch_size, d_model]
        output = self.transformer_encoder(x)  # [input_len, batch_size, d_model]
        output = output.mean(dim=0)  # [batch_size, d_model]
        output = self.decoder(output)  # [batch_size, output_len]
        return output