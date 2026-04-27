import torch
import torch.nn as nn
import math

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
        
        # Categorical Embeddings for sequence features (0 is padding)
        self.type_emb = nn.Embedding(vocab_sizes.get("TransactionTypeDescription", 20) + 1, 16, padding_idx=0)
        self.batch_emb = nn.Embedding(vocab_sizes.get("TransactionBatchDescription", 10) + 1, 16, padding_idx=0)
        self.dc_emb = nn.Embedding(vocab_sizes.get("IsDebitCredit", 3) + 1, 4, padding_idx=0)
        
        # Sequence input dim = num_feats (2: Amount, Balance) + emb_dims (16+16+4) = 38
        seq_input_dim = 2 + 16 + 16 + 4
        
        # Projection to hidden_dim for transformer
        self.input_proj = nn.Linear(seq_input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
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
        
    def forward(self, seq_num_feats, seq_cat_feats, static_feats):
        """
        seq_num_feats: [batch_size, seq_len, 2]
        seq_cat_feats: [batch_size, seq_len, 3] (type, batch, debit_credit)
        static_feats:  [batch_size, num_static_features]
        """
        # Embed categorical sequence features
        t_emb = self.type_emb(seq_cat_feats[:, :, 0])
        b_emb = self.batch_emb(seq_cat_feats[:, :, 1])
        d_emb = self.dc_emb(seq_cat_feats[:, :, 2])
        
        # Concatenate sequence features together
        seq_x = torch.cat([seq_num_feats, t_emb, b_emb, d_emb], dim=-1)
        
        # Transformer Processing
        seq_x = self.input_proj(seq_x)
        seq_x = self.pos_encoder(seq_x)
        
        trans_out = self.transformer(seq_x)
        
        # Global Average Pooling over the sequence length
        seq_repr = trans_out.mean(dim=1) # Shape: [batch_size, hidden_dim]
        
        # Process static tabular features
        static_repr = self.static_net(static_feats)
        
        # Fusion
        combined = torch.cat([seq_repr, static_repr], dim=-1)
        
        # Regression Output
        out = self.fc(combined)
        return out.squeeze(-1)
