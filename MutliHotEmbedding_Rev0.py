import torch
import torch.nn as nn
import torch.nn.functional as F


class PinPartEmbedder(nn.Module):
    def __init__(
        self,
        num_parts=17,
        num_pins=44,
        emb_dim=32,
        out_dim=32,
        hidden_dim=64,
        use_none=True,
    ):
        super().__init__()

        self.num_parts = num_parts
        self.num_pins = num_pins
        self.none_idx = num_parts if use_none else None

        part_vocab = num_parts + 1 if use_none else num_parts

        self.pin_emb = nn.Embedding(num_pins, emb_dim)
        self.part_emb = nn.Embedding(part_vocab, emb_dim)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pin_to_part):
        """
        pin_to_part: (B, 44)
        value: 0~16 part index, or 17 for no connection
        """

        B, K = pin_to_part.shape
        device = pin_to_part.device

        pin_idx = torch.arange(K, device=device)
        pin_e = self.pin_emb(pin_idx)              # (44, emb_dim)
        pin_e = pin_e.unsqueeze(0).expand(B, -1, -1)

        part_e = self.part_emb(pin_to_part)        # (B, 44, emb_dim)

        token = pin_e + part_e                     # (B, 44, emb_dim)

        # no-connection pin은 pooling에서 제외
        if self.none_idx is not None:
            mask = (pin_to_part != self.none_idx).float()
        else:
            mask = torch.ones(B, K, device=device)

        z = (token * mask.unsqueeze(-1)).sum(dim=1)

        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        z = z / count

        z = self.proj(z)
        z = F.normalize(z, dim=-1)

        return z