import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical


# =========================================================
# Dataset
# =========================================================

class JointSetEmbeddingDataset(Dataset):
    def __init__(
        self,
        meas_npy_path,
        meas_len_npy_path,
        equip_emb_path,
        label_path,
        mmap=True,
    ):
        """
        meas_npy_path:
            padded measurement array
            shape = (N, M_max, meas_dim)

        meas_len_npy_path:
            valid row length
            shape = (N,)

        equip_emb_path:
            pretrained equipment embedding
            shape = (N, equip_dim)

        label_path:
            Pass/Fail label
            shape = (N,)
        """

        self.meas = np.load(meas_npy_path, mmap_mode="r" if mmap else None)
        self.meas_len = np.load(meas_len_npy_path, mmap_mode="r" if mmap else None)
        self.equip_emb = np.load(equip_emb_path, mmap_mode="r" if mmap else None)
        self.labels = np.load(label_path, mmap_mode="r" if mmap else None)

        assert len(self.meas) == len(self.equip_emb) == len(self.labels)
        assert len(self.meas_len) == len(self.meas)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        meas = torch.from_numpy(np.asarray(self.meas[idx])).float()
        meas_len = int(self.meas_len[idx])

        equip = torch.from_numpy(np.asarray(self.equip_emb[idx])).float()
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)

        return meas, meas_len, equip, label


def joint_collate_fn(batch):
    meas, meas_len, equip, label = zip(*batch)

    meas = torch.stack(meas, dim=0)
    equip = torch.stack(equip, dim=0)
    label = torch.stack(label, dim=0)

    B, M_max, _ = meas.shape
    meas_len = torch.tensor(meas_len, dtype=torch.long)

    mask = torch.arange(M_max).unsqueeze(0) < meas_len.unsqueeze(1)
    mask = mask.float()

    return meas, mask, equip, label


# =========================================================
# Row-wise Set Encoder
# =========================================================

class RowWiseSetEncoder(nn.Module):
    def __init__(
        self,
        meas_dim,
        hidden_dim=128,
        out_dim=128,
    ):
        super().__init__()

        self.row_mlp = nn.Sequential(
            nn.Linear(meas_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x, mask):
        """
        x:    (B, M, meas_dim), standard-scaled measurement
        mask: (B, M)
        """

        h = self.row_mlp(x)
        h = h * mask.unsqueeze(-1)

        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        mean_pool = h.sum(dim=1) / count

        h_max = h.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        max_pool = h_max.max(dim=1).values

        row_anomaly = x.abs().mean(dim=-1) * mask
        anomaly_score = row_anomaly.sum(dim=1, keepdim=True) / count

        anomaly_weight = row_anomaly.masked_fill(mask == 0, -1e9)
        anomaly_weight = torch.softmax(anomaly_weight, dim=1)

        anomaly_pool = (h * anomaly_weight.unsqueeze(-1)).sum(dim=1)

        z = torch.cat(
            [
                mean_pool,
                max_pool,
                anomaly_pool,
                anomaly_score,
            ],
            dim=-1,
        )

        z = self.out_proj(z)

        aux = {
            "row_anomaly": row_anomaly,
            "anomaly_weight": anomaly_weight,
            "global_anomaly_score": anomaly_score.squeeze(-1),
        }

        return z, aux


# =========================================================
# Joint Pass/Fail + PPO Model
# =========================================================

class JointPassFailPPOModel(nn.Module):
    def __init__(
        self,
        meas_dim,
        equip_dim=32,
        set_hidden_dim=128,
        set_out_dim=128,
        fusion_dim=128,
        action_dim=4,
    ):
        super().__init__()

        self.set_encoder = RowWiseSetEncoder(
            meas_dim=meas_dim,
            hidden_dim=set_hidden_dim,
            out_dim=set_out_dim,
        )

        self.equip_encoder = nn.Sequential(
            nn.Linear(equip_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(set_out_dim + fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
        )

        self.passfail_head = nn.Linear(fusion_dim, 1)
        self.policy_head = nn.Linear(fusion_dim, action_dim)
        self.value_head = nn.Linear(fusion_dim, 1)

    def forward(self, meas_x, meas_mask, equip_emb):
        z_set, aux = self.set_encoder(meas_x, meas_mask)
        z_equip = self.equip_encoder(equip_emb)

        state = torch.cat([z_set, z_equip], dim=-1)
        state = self.fusion(state)

        passfail_logit = self.passfail_head(state).squeeze(-1)
        action_logits = self.policy_head(state)
        value = self.value_head(state).squeeze(-1)

        return {
            "state": state,
            "passfail_logit": passfail_logit,
            "action_logits": action_logits,
            "value": value,
            "aux": aux,
        }

    @torch.no_grad()
    def act(self, meas_x, meas_mask, equip_emb, epsilon=0.05):
        out = self.forward(meas_x, meas_mask, equip_emb)

        logits = out["action_logits"]
        dist = Categorical(logits=logits)

        if np.random.rand() < epsilon:
            action = torch.randint(
                0,
                logits.size(-1),
                size=(logits.size(0),),
                device=logits.device,
            )
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        value = out["value"]

        return action, log_prob, value, out


# =========================================================
# Train Pass/Fail
# =========================================================

def train_passfail_epoch(
    model,
    loader,
    optimizer,
    device,
    pos_weight=None,
):
    model.train()

    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=device,
        )

    total_loss = 0.0
    total_n = 0

    for meas_x, meas_mask, equip_emb, label in loader:
        meas_x = meas_x.to(device)
        meas_mask = meas_mask.to(device)
        equip_emb = equip_emb.to(device)
        label = label.to(device)

        out = model(meas_x, meas_mask, equip_emb)

        loss = F.binary_cross_entropy_with_logits(
            out["passfail_logit"],
            label,
            pos_weight=pos_weight_tensor,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = label.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_passfail(model, loader, device, threshold=0.5):
    model.eval()

    total = 0
    correct = 0
    tp = fp = tn = fn = 0

    for meas_x, meas_mask, equip_emb, label in loader:
        meas_x = meas_x.to(device)
        meas_mask = meas_mask.to(device)
        equip_emb = equip_emb.to(device)
        label = label.to(device)

        out = model(meas_x, meas_mask, equip_emb)
        prob = torch.sigmoid(out["passfail_logit"])
        pred = (prob >= threshold).float()

        correct += (pred == label).sum().item()
        total += label.numel()

        tp += ((pred == 1) & (label == 1)).sum().item()
        fp += ((pred == 1) & (label == 0)).sum().item()
        tn += ((pred == 0) & (label == 0)).sum().item()
        fn += ((pred == 0) & (label == 1)).sum().item()

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# =========================================================
# PPO Update
# =========================================================

def ppo_update(
    model,
    optimizer,
    batch,
    device,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    passfail_coef=0.2,
):
    meas_x = batch["meas_x"].to(device)
    meas_mask = batch["meas_mask"].to(device)
    equip_emb = batch["equip_emb"].to(device)

    actions = batch["actions"].to(device)
    old_log_probs = batch["old_log_probs"].to(device)
    returns = batch["returns"].to(device)
    advantages = batch["advantages"].to(device)

    advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-6)

    out = model(meas_x, meas_mask, equip_emb)

    dist = Categorical(logits=out["action_logits"])
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(out["value"], returns)

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    passfail_loss = torch.tensor(0.0, device=device)

    if "labels" in batch and batch["labels"] is not None:
        labels = batch["labels"].to(device).float()

        passfail_loss = F.binary_cross_entropy_with_logits(
            out["passfail_logit"],
            labels,
        )

        loss = loss + passfail_coef * passfail_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "passfail_loss": float(passfail_loss.item()),
    }


@torch.no_grad()
def reward_from_passfail(model, meas_x, meas_mask, equip_emb):
    out = model(meas_x, meas_mask, equip_emb)

    p_pass = torch.sigmoid(out["passfail_logit"])

    # [-1, +1]
    reward = 2.0 * p_pass - 1.0

    return reward


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--meas_npy", type=str, required=True)
    parser.add_argument("--meas_len_npy", type=str, required=True)
    parser.add_argument("--equip_emb_npy", type=str, required=True)
    parser.add_argument("--label_npy", type=str, required=True)

    parser.add_argument("--save_path", type=str, default="joint_passfail_ppo.pt")

    parser.add_argument("--meas_dim", type=int, required=True)
    parser.add_argument("--equip_dim", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = JointSetEmbeddingDataset(
        meas_npy_path=args.meas_npy,
        meas_len_npy_path=args.meas_len_npy,
        equip_emb_path=args.equip_emb_npy,
        label_path=args.label_npy,
        mmap=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=joint_collate_fn,
        drop_last=False,
    )

    model = JointPassFailPPOModel(
        meas_dim=args.meas_dim,
        equip_dim=args.equip_dim,
        set_hidden_dim=128,
        set_out_dim=128,
        fusion_dim=128,
        action_dim=args.action_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        loss = train_passfail_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            pos_weight=args.pos_weight,
        )

        metric = evaluate_passfail(
            model=model,
            loader=loader,
            device=device,
            threshold=0.5,
        )

        elapsed = time.time() - start

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={loss:.6f} "
            f"acc={metric['acc']:.4f} "
            f"precision={metric['precision']:.4f} "
            f"recall={metric['recall']:.4f} "
            f"f1={metric['f1']:.4f} "
            f"tp={metric['tp']} fp={metric['fp']} "
            f"tn={metric['tn']} fn={metric['fn']} "
            f"time={elapsed:.1f}s"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "meas_dim": args.meas_dim,
                "equip_dim": args.equip_dim,
                "action_dim": args.action_dim,
                "epoch": epoch,
                "loss": loss,
            },
            args.save_path,
        )

    print("saved:", args.save_path)


if __name__ == "__main__":
    main()
    
    
    python joint_passfail_ppo.py \
  --meas_npy meas_padded.npy \
  --meas_len_npy meas_len.npy \
  --equip_emb_npy pinpart_embedding.npy \
  --label_npy labels.npy \
  --meas_dim 8 \
  --equip_dim 32 \
  --batch_size 256 \
  --epochs 20