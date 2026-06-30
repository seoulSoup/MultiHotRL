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

        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask[empty, 0] = 1.0
            token[empty, 0, :] = 0.0

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
    data_list item:
        (assign, before, after, action)

    assign: (N, 44), -1 = none, 0~16 = part id
    before: (N, M_before)
    after:  (N, M_after)
    action: int
    """

    def __init__(self, data_list, clip_value=10.0):
        self.data = data_list
        self.clip_value = clip_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assign, before, after, action = self.data[idx]

        assign = np.asarray(assign, dtype=np.int64)
        before = np.atleast_2d(np.asarray(before, dtype=np.float32))
        after = np.atleast_2d(np.asarray(after, dtype=np.float32))

        assert assign.ndim == 2 and assign.shape[1] == 44, assign.shape
        assert before.ndim == 2, before.shape
        assert after.ndim == 2, after.shape
        assert assign.shape[0] == before.shape[0], (assign.shape, before.shape)
        assert assign.shape[0] == after.shape[0], (assign.shape, after.shape)

        before = np.nan_to_num(
            before,
            nan=0.0,
            posinf=self.clip_value,
            neginf=-self.clip_value,
        )
        after = np.nan_to_num(
            after,
            nan=0.0,
            posinf=self.clip_value,
            neginf=-self.clip_value,
        )

        before = np.clip(before, -self.clip_value, self.clip_value)
        after = np.clip(after, -self.clip_value, self.clip_value)

        assign = torch.as_tensor(assign, dtype=torch.long)
        before = torch.as_tensor(before, dtype=torch.float32)
        after = torch.as_tensor(after, dtype=torch.float32)
        action = torch.tensor(int(action), dtype=torch.long)

        return assign, before, after, action


def list_collate_fn(batch):
    return batch


# =========================================================
# 3. Set Transformer Blocks
# =========================================================

class MAB(nn.Module):
    def __init__(self, dim, num_heads=4, ff_hidden=128):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, dim),
        )

    def forward(self, Q, K):
        h, _ = self.attn(Q, K, K, need_weights=False)
        x = self.ln1(Q + h)
        y = self.ln2(x + self.ff(x))
        return y


class SAB(nn.Module):
    def __init__(self, dim, num_heads=4, ff_hidden=128):
        super().__init__()
        self.mab = MAB(dim, num_heads, ff_hidden)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    def __init__(self, dim, num_heads=4, num_seeds=1, ff_hidden=128):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, num_heads, ff_hidden)

    def forward(self, X):
        B = X.size(0)
        S = self.seed.expand(B, -1, -1)
        return self.mab(S, X)


# =========================================================
# 4. Row Set Transformer Encoder
# =========================================================

class RowSetTransformerEncoder(nn.Module):
    """
    matrix: (N, M)
    row = parameter
    column = unordered measurement point set
    """

    def __init__(
        self,
        token_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_sab=2,
    ):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim),
            nn.ReLU(),
        )

        self.sabs = nn.Sequential(
            *[
                SAB(
                    dim=token_dim,
                    num_heads=num_heads,
                    ff_hidden=hidden_dim * 2,
                )
                for _ in range(num_sab)
            ]
        )

        self.pma = PMA(
            dim=token_dim,
            num_heads=num_heads,
            num_seeds=1,
            ff_hidden=hidden_dim * 2,
        )

        self.out_norm = nn.LayerNorm(token_dim)

    def forward(self, matrix):
        """
        matrix: (N, M)
        return:
            row_tokens:  (N, D)
            row_anomaly: (N,)
        """

        x = matrix.unsqueeze(-1)       # (N, M, 1)
        h = self.point_mlp(x)          # (N, M, D)

        h = self.sabs(h)               # (N, M, D)
        z = self.pma(h).squeeze(1)     # (N, D)
        z = self.out_norm(z)

        row_anomaly = matrix.abs().mean(dim=-1)

        return z, row_anomaly


# =========================================================
# 5. Set-MIL Offline Q Model
# =========================================================

class SetMILOfflineQ(nn.Module):
    def __init__(
        self,
        frozen_pinpart_encoder,
        token_dim=32,
        row_hidden_dim=128,
        q_hidden_dim=128,
        action_dim=4,
        num_heads=4,
    ):
        super().__init__()

        self.pinpart_encoder = frozen_pinpart_encoder
        self.action_dim = action_dim

        self.row_encoder = RowSetTransformerEncoder(
            token_dim=token_dim,
            hidden_dim=64,
            num_heads=num_heads,
            num_sab=2,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
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

    def forward_single(self, assign, matrix):
        """
        assign: (N, 44)
        matrix: (N, M)

        return:
            row_risk: (N,)
            q_values: (action_dim,)
        """

        row_tokens, row_anomaly = self.row_encoder(matrix)

        with torch.no_grad():
            pin_tokens, pin_mask = self.pinpart_encoder.forward_tokens(assign)

        query = row_tokens.unsqueeze(1)  # (N, 1, D)

        context, cross_weights = self.cross_attn(
            query=query,
            key=pin_tokens,
            value=pin_tokens,
            key_padding_mask=(pin_mask == 0),
            need_weights=True,
        )

        context = context.squeeze(1)
        cross_weights = cross_weights.squeeze(1)

        row_context = row_tokens + context

        row_input = torch.cat(
            [row_context, row_anomaly.unsqueeze(-1)],
            dim=-1,
        )

        row_risk_logit = self.row_risk_head(row_input).squeeze(-1)
        row_risk = torch.sigmoid(row_risk_logit)

        row_attn_logit = self.row_attn_head(row_input).squeeze(-1)
        row_attn_logit = row_attn_logit.clamp(-30, 30)
        row_attn = torch.softmax(row_attn_logit, dim=0)

        pooled_row = (row_context * row_attn.unsqueeze(-1)).sum(dim=0)

        global_anomaly = row_anomaly.mean().view(1)
        max_row_risk = row_risk.max().view(1)
        weighted_row_risk = (row_risk * row_attn).sum().view(1)

        state = torch.cat(
            [
                pooled_row,
                global_anomaly,
                max_row_risk,
                weighted_row_risk,
            ],
            dim=0,
        )

        state = self.state_proj(state.unsqueeze(0)).squeeze(0)
        q_values = self.q_head(state)

        return {
            "row_risk": row_risk,
            "row_anomaly": row_anomaly,
            "row_attn": row_attn,
            "cross_weights": cross_weights,
            "state": state,
            "q_values": q_values,
            "global_anomaly": global_anomaly.squeeze(0),
            "max_row_risk": max_row_risk.squeeze(0),
        }

    @torch.no_grad()
    def select_action(self, assign, matrix):
        self.eval()
        out = self.forward_single(assign, matrix)
        action = out["q_values"].argmax().item()
        return action, out


# =========================================================
# 6. Reward
# =========================================================

@torch.no_grad()
def compute_rowwise_reward(
    model,
    assign,
    before,
    after,
    margin=0.001,
    new_bad_coef=0.2,
    fail_threshold=0.3,
    reward_scale=1.0,
):
    """
    reward = row-wise anomaly improvement

    before/after의 M이 다를 수 있으므로,
    point-wise 비교가 아니라 row별 anomaly distribution score 차이로 비교.
    """

    out_b = model.forward_single(assign, before)
    out_a = model.forward_single(assign, after)

    anom_b = out_b["row_anomaly"].detach()
    anom_a = out_a["row_anomaly"].detach()

    risk_b = out_b["row_risk"].detach()
    risk_a = out_a["row_risk"].detach()

    N = min(anom_b.size(0), anom_a.size(0))

    anom_b = anom_b[:N]
    anom_a = anom_a[:N]
    risk_b = risk_b[:N]
    risk_a = risk_a[:N]

    anom_mean = anom_b.mean()
    fail_weight = F.relu(anom_b - anom_mean)

    if fail_weight.sum() < 1e-6:
        fail_weight = torch.ones_like(fail_weight)

    denom = fail_weight.sum().clamp(min=1e-6)

    row_anom_improve = anom_b - anom_a
    row_risk_improve = risk_b - risk_a

    weighted_anom_improve = (
        fail_weight * row_anom_improve
    ).sum() / denom

    new_bad_penalty = F.relu(anom_a - anom_b).mean()

    reward = reward_scale * (
        weighted_anom_improve
        - new_bad_coef * new_bad_penalty
    )

    improved_rows = (row_anom_improve > margin).float()

    improve_ratio = (
        fail_weight * improved_rows
    ).sum() / denom

    dut_fail = (improve_ratio < fail_threshold).float()

    return reward, {
        "reward": reward,
        "weighted_anom_improve": weighted_anom_improve,
        "new_bad_penalty": new_bad_penalty,
        "improve_ratio": improve_ratio,
        "dut_fail": dut_fail,
        "row_anom_before": anom_b,
        "row_anom_after": anom_a,
        "row_risk_before": risk_b,
        "row_risk_after": risk_a,
        "row_anom_improve": row_anom_improve,
        "row_risk_improve": row_risk_improve,
    }


# =========================================================
# 7. Conservative Q Loss
# =========================================================

def conservative_q_loss(q_values, action, reward, cql_alpha=0.1):
    q_taken = q_values[action]

    td_loss = F.mse_loss(q_taken, reward)

    cql_loss = torch.logsumexp(q_values, dim=0) - q_taken

    loss = td_loss + cql_alpha * cql_loss

    return loss, {
        "td_loss": td_loss.detach(),
        "cql_loss": cql_loss.detach(),
        "q_taken": q_taken.detach(),
        "reward": reward.detach(),
    }


# =========================================================
# 8. Train / Eval
# =========================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    cql_alpha=0.1,
    grad_clip=1.0,
):
    model.train()

    total_loss = 0.0
    total_td = 0.0
    total_cql = 0.0
    total_reward = 0.0
    total_n = 0

    for batch in loader:
        bs = len(batch)
        optimizer.zero_grad()

        batch_loss_sum = 0.0

        for assign, before, after, action in batch:
            assign = assign.to(device)
            before = before.to(device)
            after = after.to(device)
            action = action.to(device)

            reward, _ = compute_rowwise_reward(
                model=model,
                assign=assign,
                before=before,
                after=after,
            )

            out = model.forward_single(assign, before)

            q_loss, qlog = conservative_q_loss(
                q_values=out["q_values"],
                action=action,
                reward=reward.detach(),
                cql_alpha=cql_alpha,
            )

            loss = q_loss / bs
            loss.backward()

            batch_loss_sum += q_loss.detach().item()

            total_td += qlog["td_loss"].item()
            total_cql += qlog["cql_loss"].item()
            total_reward += reward.detach().item()
            total_n += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )

        optimizer.step()

        total_loss += batch_loss_sum

    return {
        "loss": total_loss / max(total_n, 1),
        "td_loss": total_td / max(total_n, 1),
        "cql_loss": total_cql / max(total_n, 1),
        "reward": total_reward / max(total_n, 1),
    }


@torch.no_grad()
def evaluate_q(model, loader, device):
    model.eval()

    preds = []
    trues = []
    actions = []

    total_improve = 0.0
    total_fail = 0.0
    total_n = 0

    for batch in loader:
        for assign, before, after, action in batch:
            assign = assign.to(device)
            before = before.to(device)
            after = after.to(device)
            action = action.to(device)

            reward, rinfo = compute_rowwise_reward(
                model=model,
                assign=assign,
                before=before,
                after=after,
            )

            out = model.forward_single(assign, before)
            q_taken = out["q_values"][action]

            preds.append(q_taken.detach().cpu())
            trues.append(reward.detach().cpu())
            actions.append(action.detach().cpu())

            total_improve += rinfo["improve_ratio"].detach().item()
            total_fail += rinfo["dut_fail"].detach().item()
            total_n += 1

    preds = torch.stack(preds)
    trues = torch.stack(trues)
    actions = torch.stack(actions)

    eval_td_loss = F.mse_loss(preds, trues).item()

    if preds.std() > 1e-8 and trues.std() > 1e-8:
        corr = torch.corrcoef(torch.stack([preds, trues]))[0, 1].item()
    else:
        corr = 0.0

    action_stats = {}
    for a in torch.unique(actions):
        a_int = int(a.item())
        mask = actions == a

        if mask.sum() > 0:
            action_stats[a_int] = {
                "count": int(mask.sum().item()),
                "q_mean": float(preds[mask].mean().item()),
                "r_mean": float(trues[mask].mean().item()),
                "mse": float(F.mse_loss(preds[mask], trues[mask]).item()),
            }

    return {
        "eval_td_loss": eval_td_loss,
        "q_reward_corr": corr,
        "q_mean": preds.mean().item(),
        "q_std": preds.std().item(),
        "reward_mean": trues.mean().item(),
        "reward_std": trues.std().item(),
        "improve": total_improve / max(total_n, 1),
        "dut_fail": total_fail / max(total_n, 1),
        "action_stats": action_stats,
    }


# =========================================================
# 9. Inference
# =========================================================

@torch.no_grad()
def predict_state(
    model,
    assign,
    matrix,
    device,
    fail_threshold=0.6,
    topk_ratio=0.1,
):
    model.eval()

    assign = torch.as_tensor(assign, dtype=torch.long, device=device)
    matrix = torch.as_tensor(matrix, dtype=torch.float32, device=device)

    out = model.forward_single(assign, matrix)

    row_risk = out["row_risk"]
    q_values = out["q_values"]

    k = max(1, int(row_risk.numel() * topk_ratio))
    fail_score = row_risk.topk(k).values.mean().item()

    best_action = q_values.argmax().item()

    return {
        "is_fail": bool(fail_score >= fail_threshold),
        "fail_score": fail_score,
        "max_row_risk": row_risk.max().item(),
        "best_action": best_action,
        "q_values": q_values.detach().cpu().numpy(),
        "row_risk": row_risk.detach().cpu().numpy(),
        "row_attention": out["row_attn"].detach().cpu().numpy(),
        "pin_attention": out["cross_weights"].detach().cpu().numpy(),
    }


# =========================================================
# 10. Data Load
# =========================================================

def load_tuple_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================
# 11. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_pkl", type=str, required=True)
    parser.add_argument("--pinpart_ckpt", type=str, required=True)

    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cql_alpha", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--save_path", type=str, default="set_mil_offline_q_no_aux.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_list = load_tuple_data(args.data_pkl)

    dataset = TupleTransitionDataset(
        data_list=data_list,
        clip_value=10.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=list_collate_fn,
        drop_last=False,
    )

    frozen_encoder = load_frozen_pinpart_encoder(
        args.pinpart_ckpt,
        device,
    )

    model = SetMILOfflineQ(
        frozen_pinpart_encoder=frozen_encoder,
        token_dim=32,
        row_hidden_dim=128,
        q_hidden_dim=128,
        action_dim=args.action_dim,
        num_heads=4,
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
            grad_clip=args.grad_clip,
        )

        eval_log = evaluate_q(
            model=model,
            loader=loader,
            device=device,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={log['loss']:.6f} "
            f"td={log['td_loss']:.6f} "
            f"cql={log['cql_loss']:.6f} "
            f"reward={log['reward']:.6f} "
            f"| eval_td={eval_log['eval_td_loss']:.6f} "
            f"corr={eval_log['q_reward_corr']:.4f} "
            f"q_mean={eval_log['q_mean']:.4f} "
            f"q_std={eval_log['q_std']:.4f} "
            f"r_std={eval_log['reward_std']:.4f} "
            f"improve={eval_log['improve']:.4f} "
            f"dut_fail={eval_log['dut_fail']:.4f}"
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