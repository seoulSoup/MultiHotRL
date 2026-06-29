import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Frozen Pin/Part Encoder
# =========================================================

class PinPartAssignEncoder(nn.Module):
    def __init__(self, num_parts=17, num_pins=44, emb_dim=32, hidden_dim=64, out_dim=32):
        super().__init__()
        self.num_parts = num_parts
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

    def forward_tokens(self, assign):
        """
        assign: (B, 44), -1 or part_id
        return:
            token: (B, 44, emb_dim)
            mask:  (B, 44)
        """
        B, P = assign.shape
        device = assign.device

        mask = (assign >= 0).float()

        part_ids = torch.where(
            assign >= 0,
            assign,
            torch.full_like(assign, self.none_id),
        )

        pin_ids = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)

        part_e = self.part_emb(part_ids)
        pin_e = self.pin_emb(pin_ids)

        token = self.pair_mlp(torch.cat([part_e, pin_e], dim=-1))
        token = token + pin_e
        token = token * mask.unsqueeze(-1)

        return token, mask

    def encode(self, assign):
        token, mask = self.forward_tokens(assign)
        z = token.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        z = self.out_proj(z)
        return F.normalize(z, dim=-1)


def load_frozen_pinpart_encoder(path, device):
    ckpt = torch.load(path, map_location=device)

    model = PinPartAssignEncoder(
        num_parts=ckpt["num_parts"],
        num_pins=ckpt["num_pins"],
        emb_dim=ckpt["emb_dim"],
        hidden_dim=ckpt["hidden_dim"],
        out_dim=ckpt["out_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


# =========================================================
# 2. Dataset
# =========================================================

class TransitionDataset(Dataset):
    """
    하나의 sample:
        assign:   (44,)
        B_before: (M_max, meas_dim)
        len_before
        action:   int, 0~3
        B_after:  (M_max, meas_dim)
        len_after
    """

    def __init__(
        self,
        assign_npy,
        before_npy,
        before_len_npy,
        action_npy,
        after_npy,
        after_len_npy,
        mmap=True,
    ):
        mode = "r" if mmap else None

        self.assign = np.load(assign_npy, mmap_mode=mode)
        self.before = np.load(before_npy, mmap_mode=mode)
        self.before_len = np.load(before_len_npy, mmap_mode=mode)
        self.action = np.load(action_npy, mmap_mode=mode)
        self.after = np.load(after_npy, mmap_mode=mode)
        self.after_len = np.load(after_len_npy, mmap_mode=mode)

        n = len(self.action)
        assert len(self.assign) == n
        assert len(self.before) == n
        assert len(self.before_len) == n
        assert len(self.after) == n
        assert len(self.after_len) == n

    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        assign = torch.from_numpy(np.asarray(self.assign[idx])).long()
        before = torch.from_numpy(np.asarray(self.before[idx])).float()
        before_len = int(self.before_len[idx])
        action = torch.tensor(int(self.action[idx]), dtype=torch.long)
        after = torch.from_numpy(np.asarray(self.after[idx])).float()
        after_len = int(self.after_len[idx])

        return assign, before, before_len, action, after, after_len


def make_mask(lengths, max_len, device=None):
    if device is None:
        device = lengths.device
    ar = torch.arange(max_len, device=device).unsqueeze(0)
    return (ar < lengths.unsqueeze(1)).float()


def collate_fn(batch):
    assign, before, before_len, action, after, after_len = zip(*batch)

    assign = torch.stack(assign, dim=0)
    before = torch.stack(before, dim=0)
    after = torch.stack(after, dim=0)

    before_len = torch.tensor(before_len, dtype=torch.long)
    after_len = torch.tensor(after_len, dtype=torch.long)
    action = torch.stack(action, dim=0)

    before_mask = make_mask(before_len, before.size(1), device=before.device)
    after_mask = make_mask(after_len, after.size(1), device=after.device)

    return {
        "assign": assign,
        "before": before,
        "before_mask": before_mask,
        "action": action,
        "after": after,
        "after_mask": after_mask,
    }


# =========================================================
# 3. Row-wise Pin-conditioned MIL + Q Model
# =========================================================

class RowWiseMILOfflineQ(nn.Module):
    def __init__(
        self,
        frozen_pinpart_encoder,
        meas_dim,
        token_dim=32,
        row_hidden_dim=128,
        q_hidden_dim=128,
        action_dim=4,
    ):
        super().__init__()

        self.pinpart_encoder = frozen_pinpart_encoder
        self.action_dim = action_dim

        self.row_encoder = nn.Sequential(
            nn.Linear(meas_dim, row_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(row_hidden_dim),
            nn.Linear(row_hidden_dim, token_dim),
            nn.ReLU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=4,
            batch_first=True,
        )

        self.row_risk_head = nn.Sequential(
            nn.Linear(token_dim + 1, row_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(row_hidden_dim),
            nn.Linear(row_hidden_dim, 1),
        )

        self.row_attn_head = nn.Sequential(
            nn.Linear(token_dim + 1, row_hidden_dim),
            nn.Tanh(),
            nn.Linear(row_hidden_dim, 1),
        )

        self.state_proj = nn.Sequential(
            nn.Linear(token_dim + 3, q_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(q_hidden_dim),
            nn.Linear(q_hidden_dim, q_hidden_dim),
            nn.ReLU(),
        )

        self.q_head = nn.Linear(q_hidden_dim, action_dim)

    def forward(self, assign, meas_x, meas_mask):
        """
        assign:    (B, 44)
        meas_x:    (B, M, meas_dim)
        meas_mask: (B, M)

        return:
            row_risk: (B, M)
            q_values: (B, action_dim)
        """

        with torch.no_grad():
            pin_tokens, pin_mask = self.pinpart_encoder.forward_tokens(assign)

        row_tokens = self.row_encoder(meas_x)

        context, cross_weights = self.cross_attn(
            query=row_tokens,
            key=pin_tokens,
            value=pin_tokens,
            key_padding_mask=(pin_mask == 0),
            need_weights=True,
        )

        row_context = row_tokens + context

        # Standard scaled measurement: abs가 row anomaly
        row_anomaly = meas_x.abs().mean(dim=-1) * meas_mask

        row_input = torch.cat(
            [row_context, row_anomaly.unsqueeze(-1)],
            dim=-1,
        )

        row_risk_logit = self.row_risk_head(row_input).squeeze(-1)
        row_risk_logit = row_risk_logit.masked_fill(meas_mask == 0, -1e9)

        row_risk = torch.sigmoid(row_risk_logit) * meas_mask

        row_attn_logit = self.row_attn_head(row_input).squeeze(-1)
        row_attn_logit = row_attn_logit.masked_fill(meas_mask == 0, -1e9)
        row_attn = torch.softmax(row_attn_logit, dim=1)

        pooled_row = (row_context * row_attn.unsqueeze(-1)).sum(dim=1)

        global_anomaly = (
            row_anomaly.sum(dim=1)
            / meas_mask.sum(dim=1).clamp(min=1.0)
        ).unsqueeze(-1)

        max_row_risk = row_risk.max(dim=1).values.unsqueeze(-1)

        weighted_row_risk = (
            row_risk * row_attn
        ).sum(dim=1, keepdim=True)

        state = torch.cat(
            [
                pooled_row,
                global_anomaly,
                max_row_risk,
                weighted_row_risk,
            ],
            dim=-1,
        )

        state = self.state_proj(state)
        q_values = self.q_head(state)

        return {
            "row_risk": row_risk,
            "row_anomaly": row_anomaly,
            "row_attn": row_attn,
            "cross_weights": cross_weights,
            "state": state,
            "q_values": q_values,
            "global_anomaly": global_anomaly.squeeze(-1),
            "max_row_risk": max_row_risk.squeeze(-1),
        }

    @torch.no_grad()
    def select_action(self, assign, meas_x, meas_mask):
        out = self.forward(assign, meas_x, meas_mask)
        action = out["q_values"].argmax(dim=-1)
        return action, out


# =========================================================
# 4. Row-wise Improvement Reward
# =========================================================

@torch.no_grad()
def compute_rowwise_reward(
    model,
    assign,
    before,
    before_mask,
    after,
    after_mask,
    risk_coef=0.5,
    anomaly_coef=0.5,
    new_bad_coef=0.2,
    margin=0.02,
):
    """
    reward는 scalar지만, row-wise improvement로부터 계산.

    risk_improvement:
        기존 위험 row 중심으로 row_risk_before - row_risk_after

    anomaly_improvement:
        기존 위험 row 중심으로 abs(B_before)-abs(B_after)

    new_bad_penalty:
        다른 row가 더 나빠진 것 penalty
    """

    out_b = model(assign, before, before_mask)
    out_a = model(assign, after, after_mask)

    risk_b = out_b["row_risk"].detach()
    risk_a = out_a["row_risk"].detach()

    anom_b = out_b["row_anomaly"].detach()
    anom_a = out_a["row_anomaly"].detach()

    M = min(risk_b.size(1), risk_a.size(1))

    risk_b = risk_b[:, :M]
    risk_a = risk_a[:, :M]
    anom_b = anom_b[:, :M]
    anom_a = anom_a[:, :M]

    mask_b = before_mask[:, :M]
    mask_a = after_mask[:, :M]
    valid = mask_b * mask_a

    fail_weight = risk_b * valid
    denom = fail_weight.sum(dim=1).clamp(min=1e-6)

    row_risk_improve = risk_b - risk_a
    row_anom_improve = anom_b - anom_a

    weighted_risk_improve = (
        fail_weight * row_risk_improve
    ).sum(dim=1) / denom

    weighted_anom_improve = (
        fail_weight * row_anom_improve
    ).sum(dim=1) / denom

    new_bad_penalty = (
        F.relu(anom_a - anom_b) * valid
    ).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    reward = (
        risk_coef * weighted_risk_improve
        + anomaly_coef * weighted_anom_improve
        - new_bad_coef * new_bad_penalty
    )

    improved_rows = (row_risk_improve > margin).float() * valid

    improve_ratio = (
        fail_weight * improved_rows
    ).sum(dim=1) / denom

    dut_fail = (improve_ratio < 0.3).float()

    info = {
        "reward": reward,
        "weighted_risk_improve": weighted_risk_improve,
        "weighted_anom_improve": weighted_anom_improve,
        "new_bad_penalty": new_bad_penalty,
        "improve_ratio": improve_ratio,
        "dut_fail": dut_fail,
        "row_risk_before": risk_b,
        "row_risk_after": risk_a,
        "row_anom_before": anom_b,
        "row_anom_after": anom_a,
    }

    return reward, info


# =========================================================
# 5. Offline Conservative Q Loss
# =========================================================

def conservative_q_loss(
    q_values,
    actions,
    rewards,
    cql_alpha=0.1,
):
    """
    q_values: (B, action_dim)
    actions:  (B,)
    rewards:  (B,)

    observed action만 reward regression.
    CQL penalty로 unseen action 과대평가 방지.
    """

    q_taken = q_values.gather(
        1,
        actions.view(-1, 1),
    ).squeeze(1)

    td_loss = F.mse_loss(q_taken, rewards)

    cql_loss = (
        torch.logsumexp(q_values, dim=1)
        - q_taken
    ).mean()

    loss = td_loss + cql_alpha * cql_loss

    return loss, {
        "td_loss": float(td_loss.item()),
        "cql_loss": float(cql_loss.item()),
        "q_taken_mean": float(q_taken.mean().item()),
        "reward_mean": float(rewards.mean().item()),
    }


# =========================================================
# 6. Train / Eval
# =========================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    cql_alpha=0.1,
):
    model.train()

    total_loss = 0.0
    total_n = 0
    logs = {
        "td_loss": 0.0,
        "cql_loss": 0.0,
        "reward_mean": 0.0,
        "improve_ratio": 0.0,
        "dut_fail_rate": 0.0,
    }

    for batch in loader:
        assign = batch["assign"].to(device)
        before = batch["before"].to(device)
        before_mask = batch["before_mask"].to(device)
        action = batch["action"].to(device)
        after = batch["after"].to(device)
        after_mask = batch["after_mask"].to(device)

        reward, rinfo = compute_rowwise_reward(
            model=model,
            assign=assign,
            before=before,
            before_mask=before_mask,
            after=after,
            after_mask=after_mask,
        )

        out = model(assign, before, before_mask)

        loss, qlog = conservative_q_loss(
            q_values=out["q_values"],
            actions=action,
            rewards=reward.detach(),
            cql_alpha=cql_alpha,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = assign.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        logs["td_loss"] += qlog["td_loss"] * bs
        logs["cql_loss"] += qlog["cql_loss"] * bs
        logs["reward_mean"] += qlog["reward_mean"] * bs
        logs["improve_ratio"] += rinfo["improve_ratio"].mean().item() * bs
        logs["dut_fail_rate"] += rinfo["dut_fail"].mean().item() * bs

    for k in logs:
        logs[k] /= max(total_n, 1)

    logs["loss"] = total_loss / max(total_n, 1)

    return logs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_n = 0
    reward_sum = 0.0
    improve_sum = 0.0
    fail_sum = 0.0

    for batch in loader:
        assign = batch["assign"].to(device)
        before = batch["before"].to(device)
        before_mask = batch["before_mask"].to(device)
        after = batch["after"].to(device)
        after_mask = batch["after_mask"].to(device)

        reward, rinfo = compute_rowwise_reward(
            model=model,
            assign=assign,
            before=before,
            before_mask=before_mask,
            after=after,
            after_mask=after_mask,
        )

        bs = assign.size(0)
        total_n += bs
        reward_sum += reward.mean().item() * bs
        improve_sum += rinfo["improve_ratio"].mean().item() * bs
        fail_sum += rinfo["dut_fail"].mean().item() * bs

    return {
        "reward_mean": reward_sum / max(total_n, 1),
        "improve_ratio": improve_sum / max(total_n, 1),
        "dut_fail_rate": fail_sum / max(total_n, 1),
    }


# =========================================================
# 7. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pinpart_ckpt", type=str, required=True)

    parser.add_argument("--assign_npy", type=str, required=True)
    parser.add_argument("--before_npy", type=str, required=True)
    parser.add_argument("--before_len_npy", type=str, required=True)
    parser.add_argument("--action_npy", type=str, required=True)
    parser.add_argument("--after_npy", type=str, required=True)
    parser.add_argument("--after_len_npy", type=str, required=True)

    parser.add_argument("--meas_dim", type=int, required=True)
    parser.add_argument("--action_dim", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cql_alpha", type=float, default=0.1)

    parser.add_argument("--save_path", type=str, default="rowwise_mil_offline_q.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TransitionDataset(
        assign_npy=args.assign_npy,
        before_npy=args.before_npy,
        before_len_npy=args.before_len_npy,
        action_npy=args.action_npy,
        after_npy=args.after_npy,
        after_len_npy=args.after_len_npy,
        mmap=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    frozen_encoder = load_frozen_pinpart_encoder(
        args.pinpart_ckpt,
        device,
    )

    model = RowWiseMILOfflineQ(
        frozen_pinpart_encoder=frozen_encoder,
        meas_dim=args.meas_dim,
        token_dim=32,
        row_hidden_dim=128,
        q_hidden_dim=128,
        action_dim=args.action_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, args.epochs + 1):
        log = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            cql_alpha=args.cql_alpha,
        )

        eval_log = evaluate(
            model=model,
            loader=loader,
            device=device,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={log['loss']:.6f} "
            f"td={log['td_loss']:.6f} "
            f"cql={log['cql_loss']:.6f} "
            f"reward={log['reward_mean']:.6f} "
            f"improve={log['improve_ratio']:.4f} "
            f"dut_fail={log['dut_fail_rate']:.4f} "
            f"| eval_reward={eval_log['reward_mean']:.6f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "meas_dim": args.meas_dim,
                "action_dim": args.action_dim,
                "epoch": epoch,
            },
            args.save_path,
        )

    print("saved:", args.save_path)


if __name__ == "__main__":
    main()