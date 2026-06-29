import argparse
import pickle
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

class TupleTransitionDataset(Dataset):
    """
    data_list:
        [
            (assign, before, after, action),
            ...
        ]

    assign: (44,), -1 or part_id
    before: (N, M), full rectangular matrix
    after:  (N, M), full rectangular matrix
    action: int, 0~action_dim-1
    """

    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assign, before, after, action = self.data[idx]

        assign = torch.as_tensor(assign, dtype=torch.long)
        before = torch.as_tensor(before, dtype=torch.float32)
        after = torch.as_tensor(after, dtype=torch.float32)
        action = torch.tensor(int(action), dtype=torch.long)

        assert assign.shape == (44,)
        assert before.ndim == 2
        assert after.ndim == 2

        return assign, before, after, action


def collate_matrix_batch(batch):
    assigns, befores, afters, actions = zip(*batch)

    B = len(batch)

    max_rows = max(max(x.shape[0], y.shape[0]) for x, y in zip(befores, afters))
    max_cols = max(max(x.shape[1], y.shape[1]) for x, y in zip(befores, afters))

    assign_batch = torch.stack(assigns, dim=0)

    before_batch = torch.zeros(B, max_rows, max_cols, dtype=torch.float32)
    after_batch = torch.zeros(B, max_rows, max_cols, dtype=torch.float32)

    before_row_mask = torch.zeros(B, max_rows, dtype=torch.float32)
    after_row_mask = torch.zeros(B, max_rows, dtype=torch.float32)

    before_col_mask = torch.zeros(B, max_rows, max_cols, dtype=torch.float32)
    after_col_mask = torch.zeros(B, max_rows, max_cols, dtype=torch.float32)

    for i, (before, after) in enumerate(zip(befores, afters)):
        n0, m0 = before.shape
        n1, m1 = after.shape

        before_batch[i, :n0, :m0] = before
        after_batch[i, :n1, :m1] = after

        before_row_mask[i, :n0] = 1.0
        after_row_mask[i, :n1] = 1.0

        before_col_mask[i, :n0, :m0] = 1.0
        after_col_mask[i, :n1, :m1] = 1.0

    actions = torch.stack(actions, dim=0)

    return {
        "assign": assign_batch,
        "before": before_batch,
        "after": after_batch,
        "before_row_mask": before_row_mask,
        "after_row_mask": after_row_mask,
        "before_col_mask": before_col_mask,
        "after_col_mask": after_col_mask,
        "action": actions,
    }


# =========================================================
# 3. Row Signal Encoder
# =========================================================

class RowSignalEncoder(nn.Module):
    """
    row 하나 = parameter signal, shape (M,)
    M 가변 대응:
        Conv1D + masked adaptive-like pooling
    """

    def __init__(self, token_dim=32, hidden_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.out_norm = nn.LayerNorm(token_dim)

    def forward(self, x, col_mask):
        """
        x:        (B, N, M)
        col_mask: (B, N, M)
        return:
            row_tokens:  (B, N, D)
            row_anomaly: (B, N)
        """

        B, N, M = x.shape

        x_flat = x.reshape(B * N, 1, M)
        h = self.conv(x_flat)              # (B*N, D, M)
        h = h.transpose(1, 2)              # (B*N, M, D)

        mask_flat = col_mask.reshape(B * N, M)
        h = h * mask_flat.unsqueeze(-1)

        denom = mask_flat.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = h.sum(dim=1) / denom      # (B*N, D)

        row_tokens = pooled.reshape(B, N, -1)
        row_tokens = self.out_norm(row_tokens)

        row_anomaly = (
            x.abs() * col_mask
        ).sum(dim=-1) / col_mask.sum(dim=-1).clamp(min=1.0)

        return row_tokens, row_anomaly


# =========================================================
# 4. Row-wise MIL Offline Q Model
# =========================================================

class RowWiseMatrixMILOfflineQ(nn.Module):
    def __init__(
        self,
        frozen_pinpart_encoder,
        token_dim=32,
        row_hidden_dim=128,
        q_hidden_dim=128,
        action_dim=4,
    ):
        super().__init__()

        self.pinpart_encoder = frozen_pinpart_encoder

        self.row_encoder = RowSignalEncoder(
            token_dim=token_dim,
            hidden_dim=64,
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

    def forward(self, assign, matrix, row_mask, col_mask):
        """
        assign:   (B, 44)
        matrix:   (B, N, M)
        row_mask: (B, N)
        col_mask: (B, N, M)

        return:
            row_risk: (B, N)
            q_values: (B, action_dim)
        """

        with torch.no_grad():
            pin_tokens, pin_mask = self.pinpart_encoder.forward_tokens(assign)

        row_tokens, row_anomaly = self.row_encoder(matrix, col_mask)

        context, cross_weights = self.cross_attn(
            query=row_tokens,
            key=pin_tokens,
            value=pin_tokens,
            key_padding_mask=(pin_mask == 0),
            need_weights=True,
        )

        row_context = row_tokens + context

        row_input = torch.cat(
            [row_context, row_anomaly.unsqueeze(-1)],
            dim=-1,
        )

        row_risk_logit = self.row_risk_head(row_input).squeeze(-1)
        row_risk_logit = row_risk_logit.masked_fill(row_mask == 0, -1e9)
        row_risk = torch.sigmoid(row_risk_logit) * row_mask

        row_attn_logit = self.row_attn_head(row_input).squeeze(-1)
        row_attn_logit = row_attn_logit.masked_fill(row_mask == 0, -1e9)
        row_attn = torch.softmax(row_attn_logit, dim=1)

        pooled_row = (row_context * row_attn.unsqueeze(-1)).sum(dim=1)

        global_anomaly = (
            row_anomaly * row_mask
        ).sum(dim=1, keepdim=True) / row_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

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
    def select_action(self, assign, matrix, row_mask, col_mask):
        out = self.forward(assign, matrix, row_mask, col_mask)
        return out["q_values"].argmax(dim=-1), out


# =========================================================
# 5. Row-wise Improvement Reward
# =========================================================

@torch.no_grad()
def compute_rowwise_reward(
    model,
    assign,
    before,
    before_row_mask,
    before_col_mask,
    after,
    after_row_mask,
    after_col_mask,
    risk_coef=0.5,
    anomaly_coef=0.5,
    new_bad_coef=0.2,
    margin=0.02,
    fail_threshold=0.3,
):
    out_b = model(assign, before, before_row_mask, before_col_mask)
    out_a = model(assign, after, after_row_mask, after_col_mask)

    risk_b = out_b["row_risk"].detach()
    risk_a = out_a["row_risk"].detach()

    anom_b = out_b["row_anomaly"].detach()
    anom_a = out_a["row_anomaly"].detach()

    N = min(risk_b.size(1), risk_a.size(1))

    risk_b = risk_b[:, :N]
    risk_a = risk_a[:, :N]
    anom_b = anom_b[:, :N]
    anom_a = anom_a[:, :N]

    mask_b = before_row_mask[:, :N]
    mask_a = after_row_mask[:, :N]
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

    dut_fail = (improve_ratio < fail_threshold).float()

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
# 6. Conservative Q Loss
# =========================================================

def conservative_q_loss(q_values, actions, rewards, cql_alpha=0.1):
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
# 7. Train / Eval
# =========================================================

def train_epoch(model, loader, optimizer, device, cql_alpha=0.1):
    model.train()

    total_loss = 0.0
    total_n = 0

    acc = {
        "td_loss": 0.0,
        "cql_loss": 0.0,
        "reward_mean": 0.0,
        "improve_ratio": 0.0,
        "dut_fail_rate": 0.0,
    }

    for batch in loader:
        assign = batch["assign"].to(device)
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        before_row_mask = batch["before_row_mask"].to(device)
        after_row_mask = batch["after_row_mask"].to(device)
        before_col_mask = batch["before_col_mask"].to(device)
        after_col_mask = batch["after_col_mask"].to(device)
        action = batch["action"].to(device)

        reward, rinfo = compute_rowwise_reward(
            model=model,
            assign=assign,
            before=before,
            before_row_mask=before_row_mask,
            before_col_mask=before_col_mask,
            after=after,
            after_row_mask=after_row_mask,
            after_col_mask=after_col_mask,
        )

        out = model(assign, before, before_row_mask, before_col_mask)

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

        acc["td_loss"] += qlog["td_loss"] * bs
        acc["cql_loss"] += qlog["cql_loss"] * bs
        acc["reward_mean"] += qlog["reward_mean"] * bs
        acc["improve_ratio"] += rinfo["improve_ratio"].mean().item() * bs
        acc["dut_fail_rate"] += rinfo["dut_fail"].mean().item() * bs

    for k in acc:
        acc[k] /= max(total_n, 1)

    acc["loss"] = total_loss / max(total_n, 1)

    return acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total = 0
    reward_sum = 0.0
    improve_sum = 0.0
    fail_sum = 0.0

    for batch in loader:
        assign = batch["assign"].to(device)
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        before_row_mask = batch["before_row_mask"].to(device)
        after_row_mask = batch["after_row_mask"].to(device)
        before_col_mask = batch["before_col_mask"].to(device)
        after_col_mask = batch["after_col_mask"].to(device)

        reward, rinfo = compute_rowwise_reward(
            model=model,
            assign=assign,
            before=before,
            before_row_mask=before_row_mask,
            before_col_mask=before_col_mask,
            after=after,
            after_row_mask=after_row_mask,
            after_col_mask=after_col_mask,
        )

        bs = assign.size(0)
        total += bs
        reward_sum += reward.mean().item() * bs
        improve_sum += rinfo["improve_ratio"].mean().item() * bs
        fail_sum += rinfo["dut_fail"].mean().item() * bs

    return {
        "reward_mean": reward_sum / max(total, 1),
        "improve_ratio": improve_sum / max(total, 1),
        "dut_fail_rate": fail_sum / max(total, 1),
    }


# =========================================================
# 8. Load data
# =========================================================

def load_tuple_data(path):
    """
    pickle file containing:
        list of (assign, before, after, action)
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# =========================================================
# 9. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_pkl", type=str, required=True)
    parser.add_argument("--pinpart_ckpt", type=str, required=True)

    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cql_alpha", type=float, default=0.1)

    parser.add_argument("--save_path", type=str, default="rowwise_matrix_mil_offline_q.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_list = load_tuple_data(args.data_pkl)

    dataset = TupleTransitionDataset(data_list)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_matrix_batch,
        drop_last=False,
    )

    frozen_encoder = load_frozen_pinpart_encoder(
        args.pinpart_ckpt,
        device,
    )

    model = RowWiseMatrixMILOfflineQ(
        frozen_pinpart_encoder=frozen_encoder,
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
                "action_dim": args.action_dim,
                "epoch": epoch,
            },
            args.save_path,
        )

    print("saved:", args.save_path)


if __name__ == "__main__":
    main()