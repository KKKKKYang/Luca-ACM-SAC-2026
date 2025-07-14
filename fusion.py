import torch, torch.nn as nn, torch.nn.functional as F

# class GatedOpFusion(nn.Module):
#     """
#     Fuse the original op‐features (D_op) with text‐features (D_text_small)
#     and output back D_op dims, via a learned gate.
#     """
#     def __init__(self, D_op: int, D_text: int):
#         super().__init__()
#         # project text→D_op
#         self.text_proj = nn.Linear(D_text, D_op)
#         # gate takes [D_op + D_text] → one gate per channel
#         self.gate_fc   = nn.Linear(D_op + D_text, D_op)

#     def forward(self, op_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
#         # op_feat:  [B, D_op, N]
#         # text_feat:[B, D_text, N]
#         B, D_op, N = op_feat.shape
#         # flatten channels into batch
#         o = op_feat.permute(0,2,1).reshape(B*N, D_op)                  # [B*N, D_op]
#         t = text_feat.permute(0,2,1).reshape(B*N, text_feat.size(1))   # [B*N, D_text]
#         t_proj = self.text_proj(t)                                    # [B*N, D_op]
#         g      = torch.sigmoid(self.gate_fc(torch.cat([o,t], dim=1)))# [B*N, D_op]
#         fused  = g*o + (1-g)*t_proj                                   # [B*N, D_op]
#         # reshape back to [B, D_op, N]
#         return fused.view(B, N, D_op).permute(0,2,1)

# fusion.py
import torch, torch.nn as nn, torch.nn.functional as F

class GatedOpFusion(nn.Module):
    """
    Fuse original op‐features (D_op) with text‐features (D_text)
    and output back D_op dims, via a learned gate.
    """
    def __init__(self, D_op: int, D_text: int):
        super().__init__()
        # project text → D_op
        self.text_proj = nn.Linear(D_text, D_op)
        # gate from [D_op + D_op] → D_op
        self.gate_fc   = nn.Linear(2 * D_op, D_op)
        # self.gate_fc   = nn.Linear(D_op + D_text, D_op)

    def forward(self, op_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        # op_feat:  [B, D_op, N]
        # text_feat:[B, D_text, N]
        B, D_op, N = op_feat.shape

        # flatten to [B*N, …]
        o = op_feat.permute(0,2,1).reshape(B*N, D_op)                       # [B*N, D_op]
        t = text_feat.permute(0,2,1).reshape(B*N, text_feat.size(1))        # [B*N, D_text]

        # project text → D_op
        t_proj = self.text_proj(t)                                          # [B*N, D_op]

        # compute a gate over [o, t_proj]
        g = torch.sigmoid(self.gate_fc(torch.cat([o, t_proj], dim=1)))      # [B*N, D_op]

        # fuse
        fused = g * o + (1 - g) * t_proj                                    # [B*N, D_op]

        # reshape back to [B, D_op, N]
        return fused.view(B, N, D_op).permute(0,2,1)


class FilmOpFusion(nn.Module):
    def __init__(self, D_op: int, D_text: int):
        super().__init__()
        # yields [γ,β] per channel
        self.mod_fc = nn.Linear(D_text, 2*D_op)

    def forward(self, op_feat, text_feat):
        B, D_op, N = op_feat.shape
        # flatten text → [B*N, D_text]
        t = text_feat.permute(0,2,1).reshape(B*N, text_feat.size(1))
        γβ = self.mod_fc(t)             # [B*N, 2*D_op]
        γ, β = γβ.chunk(2, dim=-1)      # each [B*N, D_op]
        # reshape back to [B, D_op, N]
        γ = γ.view(B,N,D_op).permute(0,2,1)
        β = β.view(B,N,D_op).permute(0,2,1)
        return γ*op_feat + β