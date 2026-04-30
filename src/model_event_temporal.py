import torch
import torch.nn as nn


class MaskedAttentionPool(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scorer = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        if mask is None:
            return x.mean(dim=1)

        mask = mask.bool()
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class EventTemporalModel(nn.Module):
    def __init__(
        self,
        event_cont_dim,
        event_cat_cardinalities,
        monthly_dim,
        static_dim,
        hidden_dim=96,
        dropout=0.2,
        num_bands=5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim

        self.event_embeddings = nn.ModuleList()
        emb_total_dim = 0
        for cardinality in event_cat_cardinalities:
            cardinality = int(max(cardinality, 2))
            emb_dim = min(16, max(4, int(round(cardinality ** 0.25 * 4))))
            self.event_embeddings.append(nn.Embedding(cardinality, emb_dim, padding_idx=0))
            emb_total_dim += emb_dim

        self.event_cont_net = nn.Sequential(
            nn.Linear(event_cont_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.event_token_proj = nn.Sequential(
            nn.Linear((hidden_dim // 2) + emb_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.event_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.event_pool = MaskedAttentionPool(hidden_dim * 2)

        self.monthly_proj = nn.Sequential(
            nn.Linear(monthly_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.monthly_gru = nn.GRU(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.monthly_pool = MaskedAttentionPool(hidden_dim)

        if static_dim > 0:
            self.static_net = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            static_out_dim = hidden_dim
        else:
            self.static_net = None
            static_out_dim = 0

        fusion_dim = (hidden_dim * 2) + hidden_dim + static_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.count_head = nn.Linear(hidden_dim, 1)
        self.active_head = nn.Linear(hidden_dim, 1)
        self.band_head = nn.Linear(hidden_dim, num_bands)

    def forward(self, event_cont, event_cat, event_mask, monthly, static):
        cont_repr = self.event_cont_net(event_cont)
        cat_repr = []
        for idx, embedding in enumerate(self.event_embeddings):
            cat_values = event_cat[:, :, idx].long().clamp(min=0, max=embedding.num_embeddings - 1)
            cat_repr.append(embedding(cat_values))
        event_tokens = torch.cat([cont_repr, *cat_repr], dim=-1)
        event_tokens = self.event_token_proj(event_tokens)
        event_encoded, _ = self.event_gru(event_tokens)
        event_repr = self.event_pool(event_encoded, event_mask)

        monthly_mask = monthly.abs().sum(dim=-1) > 0
        monthly_tokens = self.monthly_proj(monthly)
        monthly_encoded, _ = self.monthly_gru(monthly_tokens)
        monthly_repr = self.monthly_pool(monthly_encoded, monthly_mask)

        parts = [event_repr, monthly_repr]
        if self.static_net is not None:
            parts.append(self.static_net(static))
        fused = self.fusion(torch.cat(parts, dim=-1))
        return {
            "count": self.count_head(fused).squeeze(-1),
            "active": self.active_head(fused).squeeze(-1),
            "band": self.band_head(fused),
        }
