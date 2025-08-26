# model_a2c.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    Two-stage masked actor + pooled critic.
    - Encoder: maps per-component feature to embedding h_j
    - Stage1: score per component z_j = w1^T h_j (masked softmax)
    - Stage2: for chosen p, produce scores for all q using MLP on [x_p, x_q, x_p + x_q, |size_p - size_q|]
    - Critic: pool(h_j) -> V(s)
    """
    def __init__(self, feat_dim=6, emb_dim=128, hidden=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.hidden = hidden

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )
        # stage1 head (scalar per comp)
        self.stage1_head = nn.Linear(emb_dim, 1)

        # stage2 pair MLP: input dim = feat_dim * 3 + 1 
        pair_in = feat_dim * 3 + 1
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # critic (pooled)
        self.critic_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def encode(self, feats: torch.Tensor):
        """
        feats: (m, feat_dim)
        returns h: (m, emb_dim)
        """
        return self.encoder(feats)

    def stage1_logits(self, feats: torch.Tensor):
        """
        returns logits shape (m,)
        """
        h = self.encode(feats)
        z = self.stage1_head(h).squeeze(-1)
        return z, h

    def stage2_logits(self, feats: torch.Tensor, p_idx: int):
        """
        feats: (m, feat_dim)
        p_idx: int
        returns: logits (m,) - includes p_idx (should be masked externally)
        """
        m = feats.shape[0]
        x_p = feats[p_idx].unsqueeze(0).repeat(m, 1)  # (m, feat_dim)
        x_q = feats  # (m, feat_dim)
        x_sum = x_p + x_q
        size_diff = torch.abs(feats[:, 0] - feats[p_idx, 0]).unsqueeze(1)  # (m,1)
        pair_feat = torch.cat([x_p, x_q, x_sum, size_diff], dim=1)  # (m, 3*feat_dim + 1)
        scores = self.pair_mlp(pair_feat).squeeze(-1)
        return scores

    def value(self, h: torch.Tensor):
        """
        h: (m, emb_dim)
        returns scalar V(s)
        """
        if h.shape[0] == 0:
            return torch.tensor(0.0, device=h.device)
        g = h.mean(dim=0)  # pooled
        v = self.critic_mlp(g)
        return v.squeeze(0)
