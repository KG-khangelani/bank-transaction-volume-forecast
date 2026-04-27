import torch
import torch.nn as nn

class TransactionSequenceModel(nn.Module):
    def __init__(self, vocab_sizes, num_static_features, hidden_dim=128, num_layers=2):
        super().__init__()
        
        # Categorical Embeddings for sequence features (0 is padding)
        self.type_emb = nn.Embedding(vocab_sizes.get("TransactionTypeDescription", 20) + 1, 16, padding_idx=0)
        self.batch_emb = nn.Embedding(vocab_sizes.get("TransactionBatchDescription", 10) + 1, 16, padding_idx=0)
        self.dc_emb = nn.Embedding(vocab_sizes.get("IsDebitCredit", 3) + 1, 4, padding_idx=0)
        
        # Sequence input dim = num_feats (2: Amount, Balance) + emb_dims (16+16+4) = 38
        seq_input_dim = 2 + 16 + 16 + 4
        
        # LSTM layer to process temporal transaction data
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
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
        
        # Process sequence through LSTM
        # Using output, (h_n, c_n)
        _, (h_n, _) = self.lstm(seq_x)
        
        # Take the hidden state of the last layer
        seq_repr = h_n[-1] # Shape: [batch_size, hidden_dim]
        
        # Process static tabular features
        static_repr = self.static_net(static_feats)
        
        # Fusion
        combined = torch.cat([seq_repr, static_repr], dim=-1)
        
        # Regression Output
        out = self.fc(combined)
        return out.squeeze(-1)
