import torch
import torch.nn as nn
import math
from pipeline_utils import SEQUENCE_LENGTH

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransactionSequenceModel(nn.Module):
    def __init__(self, vocab_sizes, num_static_features, hidden_dim=128, num_layers=2):
        super().__init__()
        
        # Sequence input dim = 3 (Monthly Count, Sum, Balance)
        seq_input_dim = 3
        
        # Projection to hidden_dim for transformer
        self.input_proj = nn.Linear(seq_input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=SEQUENCE_LENGTH)
        
        # Transformer Encoder layer to process temporal transaction data using Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            batch_first=True, 
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Static features feed-forward network (Demographics, Financial Aggregates)
        self.static_net = nn.Sequential(
            nn.Linear(num_static_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layer combining sequence context and static context
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, seq_num_feats, static_feats, seq_mask=None):
        """
        seq_num_feats: [batch_size, 35, 3]
        static_feats:  [batch_size, num_static_features]
        seq_mask:      [batch_size, 35] bool mask for observed months
        """
        # Transformer Processing
        seq_x = self.input_proj(seq_num_feats)
        seq_x = self.pos_encoder(seq_x)
        
        trans_out = self.transformer(seq_x)
        
        if seq_mask is None:
            seq_repr = trans_out.mean(dim=1)
        else:
            weights = seq_mask.to(trans_out.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            seq_repr = (trans_out * weights).sum(dim=1) / denom
        
        # Process static tabular features
        static_repr = self.static_net(static_feats)
        
        # Fusion
        combined = torch.cat([seq_repr, static_repr], dim=-1)
        
        # Regression Output
        out = self.fc(combined)
        return out.squeeze(-1)
