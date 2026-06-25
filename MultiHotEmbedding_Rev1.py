import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Dataset
# =========================================================

class PinPartAssignDataset(Dataset):
    """
    X shape: (N, 44)
    value:
        -1   = no connection
        0~16 = part id
    """

    def __init__(self, npy_path, mmap=True):
        self.X = np.load(npy_path, mmap_mode="r" if mmap else None)

        assert self.X.ndim == 2, self.X.shape
        assert self.X.shape[1] == 44, self.X.shape

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.int64)
        return torch.from_numpy(x)


# =========================================================
# Encoder
# =========================================================

class PinPartAssignEncoder(nn.Module):
    def __init__(
        self,
        num_parts=17,
        num_pins=44,
        emb_dim=32,
        hidden_dim=64,
        out_dim=32,
    ):
        super().__init__()

        self.num_parts = num_parts
        self.num_pins = num_pins
        self.none_id = num_parts

        self.part_emb = nn.Embedding(num_parts + 1, emb_dim)
        self.pin_emb = nn.Embedding(num_pins, emb_dim)

        self.pair_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, assign):
        """
        assign: (B, 44), -1 or 0~16
        """

        B, num_pins = assign.shape
        device = assign.device

        mask = (assign >= 0).float()

        part_ids = torch.where(
            assign >= 0,
            assign,
            torch.full_like(assign, self.none_id),
        )

        pin_ids = torch.arange(num_pins, device=device)
        pin_ids = pin_ids.unsqueeze(0).expand(B, -1)

        part_e = self.part_emb(part_ids)
        pin_e = self.pin_emb(pin_ids)

        pair_input = torch.cat([part_e, pin_e], dim=-1)
        token = self.pair_mlp(pair_input)

        # 동일 pin 사용성을 강조
        token = token + pin_e

        token = token * mask.unsqueeze(-1)

        z = token.sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        z = z / count

        z = self.out_proj(z)
        z = F.normalize(z, dim=-1)

        return z


# =========================================================
# Similarity
# =========================================================

def batch_domain_similarity_from_assign(
    assign,
    num_parts=17,
    w_pin=0.7,
    w_pair=0.2,
    w_part=0.1,
):
    """
    assign: (B, 44)

    similarity =
        0.7 * same pin usage
      + 0.2 * exact pin-part pair
      + 0.1 * same part usage
    """

    B, num_pins = assign.shape
    device = assign.device

    conn = (assign >= 0).float()

    # 1. pin Jaccard
    pin_inter = conn @ conn.T
    pin_count = conn.sum(dim=1, keepdim=True)
    pin_union = pin_count + pin_count.T - pin_inter
    pin_sim = pin_inter / pin_union.clamp(min=1.0)

    # 2. exact pair Jaccard
    same_part = assign.unsqueeze(1) == assign.unsqueeze(0)
    both_conn = (assign.unsqueeze(1) >= 0) & (assign.unsqueeze(0) >= 0)

    pair_inter = (same_part & both_conn).float().sum(dim=-1)
    pair_count = conn.sum(dim=1, keepdim=True)
    pair_union = pair_count + pair_count.T - pair_inter
    pair_sim = pair_inter / pair_union.clamp(min=1.0)

    # 3. part Jaccard
    part_present = torch.zeros(
        B,
        num_parts,
        dtype=torch.float32,
        device=device,
    )

    valid = assign >= 0
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_pins)

    part_present[
        batch_idx[valid],
        assign[valid].long(),
    ] = 1.0

    part_inter = part_present @ part_present.T
    part_count = part_present.sum(dim=1, keepdim=True)
    part_union = part_count + part_count.T - part_inter
    part_sim = part_inter / part_union.clamp(min=1.0)

    sim = (
        w_pin * pin_sim
        + w_pair * pair_sim
        + w_part * part_sim
    )

    return sim


# =========================================================
# Train
# =========================================================

def train(
    npy_path,
    save_path,
    num_parts=17,
    num_pins=44,
    emb_dim=32,
    hidden_dim=64,
    out_dim=32,
    batch_size=512,
    epochs=3,
    lr=5e-4,
    num_workers=0,
    max_steps_per_epoch=None,
):
    device = torch.device("cpu")

    torch.set_num_threads(max(1, os.cpu_count() // 2))

    dataset = PinPartAssignDataset(npy_path, mmap=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    model = PinPartAssignEncoder(
        num_parts=num_parts,
        num_pins=num_pins,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    print("=================================================")
    print("CPU Pin-Part Embedding Pretrain")
    print("data:", npy_path)
    print("N:", len(dataset))
    print("batch_size:", batch_size)
    print("emb_dim:", emb_dim)
    print("out_dim:", out_dim)
    print("epochs:", epochs)
    print("threads:", torch.get_num_threads())
    print("=================================================")

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_n = 0
        start = time.time()

        for step, assign in enumerate(loader, start=1):
            assign = assign.to(device)

            z = model(assign)
            pred_sim = z @ z.T

            with torch.no_grad():
                target_sim = batch_domain_similarity_from_assign(
                    assign,
                    num_parts=num_parts,
                    w_pin=0.7,
                    w_pair=0.2,
                    w_part=0.1,
                )

            loss = F.mse_loss(pred_sim, target_sim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = assign.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            if step % 100 == 0:
                elapsed = time.time() - start
                print(
                    f"[Epoch {epoch:03d} | Step {step:06d}] "
                    f"loss={total_loss / total_n:.6f} "
                    f"elapsed={elapsed:.1f}s"
                )

            if max_steps_per_epoch is not None:
                if step >= max_steps_per_epoch:
                    break

        avg_loss = total_loss / max(total_n, 1)
        elapsed = time.time() - start

        print(
            f"[Epoch {epoch:03d} Done] "
            f"loss={avg_loss:.6f} "
            f"time={elapsed:.1f}s"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_parts": num_parts,
                "num_pins": num_pins,
                "emb_dim": emb_dim,
                "hidden_dim": hidden_dim,
                "out_dim": out_dim,
                "epoch": epoch,
                "loss": avg_loss,
            },
            save_path,
        )

        print(f"saved: {save_path}")

    return model


# =========================================================
# Transform
# =========================================================

@torch.no_grad()
def transform_to_embedding(
    npy_path,
    ckpt_path,
    out_npy_path,
    batch_size=4096,
):
    device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = PinPartAssignEncoder(
        num_parts=ckpt["num_parts"],
        num_pins=ckpt["num_pins"],
        emb_dim=ckpt["emb_dim"],
        hidden_dim=ckpt["hidden_dim"],
        out_dim=ckpt["out_dim"],
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = PinPartAssignDataset(npy_path, mmap=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    Z = np.memmap(
        out_npy_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(dataset), ckpt["out_dim"]),
    )

    offset = 0

    for step, assign in enumerate(loader, start=1):
        z = model(assign).numpy().astype(np.float32)

        bs = z.shape[0]
        Z[offset:offset + bs] = z
        offset += bs

        if step % 100 == 0:
            print(f"transform step={step}, offset={offset}")

    Z.flush()
    print("saved embedding memmap:", out_npy_path)
    print("shape:", (len(dataset), ckpt["out_dim"]))


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--npy_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="pinpart_encoder_cpu.pt")
    parser.add_argument("--out_npy_path", type=str, default="pinpart_embedding.dat")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)

    args = parser.parse_args()

    if args.mode == "train":
        train(
            npy_path=args.npy_path,
            save_path=args.save_path,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            max_steps_per_epoch=args.max_steps_per_epoch,
        )

    elif args.mode == "transform":
        transform_to_embedding(
            npy_path=args.npy_path,
            ckpt_path=args.save_path,
            out_npy_path=args.out_npy_path,
            batch_size=args.batch_size,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()