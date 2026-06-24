import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Vocabulary
# =========================================================

def build_vocab(equipment_list):
    pin_set = set()
    part_set = set()

    for eq in equipment_list:
        for pin, part in eq.items():
            pin_set.add(str(pin))
            part_set.add(str(part))

    pin2id = {p: i for i, p in enumerate(sorted(pin_set))}
    part2id = {p: i for i, p in enumerate(sorted(part_set))}

    return pin2id, part2id


def save_vocab(pin2id, part2id, path="equipment_vocab.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pin2id": pin2id,
                "part2id": part2id,
            },
            f,
            ensure_ascii=False,
            indent=2
        )


def load_vocab(path="equipment_vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    return obj["pin2id"], obj["part2id"]


# =========================================================
# 2. Dataset
# =========================================================

class EquipmentPairDataset(Dataset):
    def __init__(self, equipment_list, pin2id, part2id):
        self.samples = []

        for eq in equipment_list:
            pairs = []

            for pin, part in eq.items():
                pin = str(pin)
                part = str(part)

                if pin not in pin2id or part not in part2id:
                    continue

                pairs.append([
                    part2id[part],
                    pin2id[pin]
                ])

            if len(pairs) == 0:
                pairs = [[0, 0]]

            self.samples.append(
                torch.tensor(pairs, dtype=torch.long)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pairs(batch):
    """
    batch: list of (num_connections, 2)
    return:
        pair_ids: (B, Lmax, 2)
        mask:     (B, Lmax)
    """
    B = len(batch)
    max_len = max(x.size(0) for x in batch)

    pair_ids = torch.zeros(B, max_len, 2, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.float32)

    for i, pairs in enumerate(batch):
        L = pairs.size(0)
        pair_ids[i, :L] = pairs
        mask[i, :L] = 1.0

    return pair_ids, mask


# =========================================================
# 3. Model
# =========================================================

class EquipmentEmbedder(nn.Module):
    def __init__(
        self,
        num_parts,
        num_pins,
        emb_dim=32,
        pair_hidden_dim=64,
        out_dim=32,
    ):
        super().__init__()

        self.part_emb = nn.Embedding(num_parts, emb_dim)
        self.pin_emb = nn.Embedding(num_pins, emb_dim)

        self.pair_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, pair_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(pair_hidden_dim),
            nn.Linear(pair_hidden_dim, emb_dim),
            nn.ReLU(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, out_dim),
        )

    def forward(self, pair_ids, mask):
        """
        pair_ids: (B, L, 2)
                  pair_ids[..., 0] = part_id
                  pair_ids[..., 1] = pin_id

        mask: (B, L)
        """

        part_ids = pair_ids[..., 0]
        pin_ids = pair_ids[..., 1]

        part_e = self.part_emb(part_ids)
        pin_e = self.pin_emb(pin_ids)

        token = torch.cat([part_e, pin_e], dim=-1)
        token = self.pair_mlp(token)

        token = token * mask.unsqueeze(-1)

        z = token.sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        z = z / count

        z = self.out_proj(z)
        z = F.normalize(z, dim=-1)

        return z


# =========================================================
# 4. Target similarity
# =========================================================

def batch_jaccard_similarity(pair_ids, mask):
    """
    연결 pair set 기준 Jaccard similarity.
    같은 (part, pin) pair가 얼마나 겹치는지 측정.
    """

    B, L, _ = pair_ids.shape
    sim = torch.zeros(B, B, device=pair_ids.device)

    sets = []

    pair_ids_cpu = pair_ids.detach().cpu()
    mask_cpu = mask.detach().cpu()

    for i in range(B):
        valid = mask_cpu[i].bool()
        pairs = pair_ids_cpu[i, valid]
        pair_set = set(
            (int(p[0]), int(p[1]))
            for p in pairs
        )
        sets.append(pair_set)

    for i in range(B):
        for j in range(B):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            sim[i, j] = inter / union if union > 0 else 0.0

    return sim


# =========================================================
# 5. Train
# =========================================================

def train_equipment_embedder(
    equipment_list,
    emb_dim=32,
    out_dim=32,
    batch_size=256,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    save_model_path="equipment_embedder.pt",
    save_vocab_path="equipment_vocab.json",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pin2id, part2id = build_vocab(equipment_list)
    save_vocab(pin2id, part2id, save_vocab_path)

    dataset = EquipmentPairDataset(
        equipment_list,
        pin2id,
        part2id
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        drop_last=False,
    )

    model = EquipmentEmbedder(
        num_parts=len(part2id),
        num_pins=len(pin2id),
        emb_dim=emb_dim,
        out_dim=out_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for pair_ids, mask in loader:
            pair_ids = pair_ids.to(device)
            mask = mask.to(device)

            z = model(pair_ids, mask)
            pred_sim = z @ z.T

            target_sim = batch_jaccard_similarity(
                pair_ids,
                mask
            )

            loss = F.mse_loss(pred_sim, target_sim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = pair_ids.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        avg_loss = total_loss / total_n

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] loss = {avg_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_parts": len(part2id),
            "num_pins": len(pin2id),
            "emb_dim": emb_dim,
            "out_dim": out_dim,
        },
        save_model_path
    )

    print(f"Saved model: {save_model_path}")
    print(f"Saved vocab: {save_vocab_path}")

    return model, pin2id, part2id


# =========================================================
# 6. Load / Transform
# =========================================================

def load_equipment_embedder(
    model_path="equipment_embedder.pt",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(model_path, map_location=device)

    model = EquipmentEmbedder(
        num_parts=ckpt["num_parts"],
        num_pins=ckpt["num_pins"],
        emb_dim=ckpt["emb_dim"],
        out_dim=ckpt["out_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def transform_equipment(
    equipment_list,
    model,
    pin2id,
    part2id,
    batch_size=512,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    dataset = EquipmentPairDataset(
        equipment_list,
        pin2id,
        part2id
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
    )

    embeddings = []

    model.eval()

    for pair_ids, mask in loader:
        pair_ids = pair_ids.to(device)
        mask = mask.to(device)

        z = model(pair_ids, mask)
        embeddings.append(z.cpu())

    return torch.cat(embeddings, dim=0).numpy()


# =========================================================
# 7. Example
# =========================================================

if __name__ == "__main__":

    equipment_list = [
        {
            "PIN01": "PART_A",
            "PIN02": "PART_A",
            "PIN10": "PART_B",
        },
        {
            "PIN01": "PART_A",
            "PIN03": "PART_A",
            "PIN11": "PART_B",
        },
        {
            "PIN20": "PART_C",
            "PIN21": "PART_C",
            "PIN22": "PART_D",
        },
    ]

    model, pin2id, part2id = train_equipment_embedder(
        equipment_list,
        emb_dim=32,
        out_dim=32,
        batch_size=2,
        epochs=100,
        lr=1e-3,
    )

    Z = transform_equipment(
        equipment_list,
        model,
        pin2id,
        part2id
    )

    print(Z.shape)
    print(Z)

    np.save("equipment_embedding.npy", Z)
    
    
def batch_domain_similarity(
    pair_ids,
    mask,
    w_pin=0.7,
    w_pair=0.2,
    w_part=0.1,
):
    """
    pair_ids: (B, L, 2)
              pair_ids[..., 0] = part_id
              pair_ids[..., 1] = pin_id
    mask: (B, L)

    similarity =
        w_pin  * pin overlap
      + w_pair * exact (part,pin) overlap
      + w_part * part overlap
    """

    B, L, _ = pair_ids.shape
    sim = torch.zeros(B, B, device=pair_ids.device)

    pair_ids_cpu = pair_ids.detach().cpu()
    mask_cpu = mask.detach().cpu()

    pin_sets = []
    part_sets = []
    pair_sets = []

    for i in range(B):
        valid = mask_cpu[i].bool()
        pairs = pair_ids_cpu[i, valid]

        part_set = set(int(p[0]) for p in pairs)
        pin_set = set(int(p[1]) for p in pairs)
        pair_set = set((int(p[0]), int(p[1])) for p in pairs)

        pin_sets.append(pin_set)
        part_sets.append(part_set)
        pair_sets.append(pair_set)

    for i in range(B):
        for j in range(B):
            pin_sim = jaccard(pin_sets[i], pin_sets[j])
            part_sim = jaccard(part_sets[i], part_sets[j])
            pair_sim = jaccard(pair_sets[i], pair_sets[j])

            sim[i, j] = (
                w_pin * pin_sim
                + w_pair * pair_sim
                + w_part * part_sim
            )

    return sim


def jaccard(a, b):
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union
    