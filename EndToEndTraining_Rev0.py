import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical


# =========================================================
# 1. Vocab
# =========================================================

def build_vocab(equipment_dicts):
    pin_set = set()
    part_set = set()

    for eq in equipment_dicts:
        for pin, part in eq.items():
            pin_set.add(str(pin))
            part_set.add(str(part))

    pin2id = {p: i for i, p in enumerate(sorted(pin_set))}
    part2id = {p: i for i, p in enumerate(sorted(part_set))}

    return pin2id, part2id


def save_vocab(pin2id, part2id, path="vocab.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"pin2id": pin2id, "part2id": part2id},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    return obj["pin2id"], obj["part2id"]


# =========================================================
# 2. Dataset
# =========================================================

class AnomalyJointDataset(Dataset):
    def __init__(
        self,
        equipment_dicts,
        measurement_sets,
        labels,
        pin2id,
        part2id,
    ):
        """
        equipment_dicts:
            list of dict
            예: {"PIN01": "PART_A", "PIN02": "PART_B"}

        measurement_sets:
            list of arrays
            each shape: (M_i, meas_dim)
            Standard Scaling된 측정 데이터

        labels:
            0/1 Pass/Fail label
        """

        self.samples = []

        for eq, meas, y in zip(equipment_dicts, measurement_sets, labels):
            pairs = []

            for pin, part in eq.items():
                pin = str(pin)
                part = str(part)

                if pin in pin2id and part in part2id:
                    pairs.append([part2id[part], pin2id[pin]])

            if len(pairs) == 0:
                pairs = [[0, 0]]

            pair_ids = torch.tensor(pairs, dtype=torch.long)

            if isinstance(meas, np.ndarray):
                meas = torch.from_numpy(meas)

            meas = meas.float()
            y = torch.tensor(float(y), dtype=torch.float32)

            self.samples.append((pair_ids, meas, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def anomaly_collate_fn(batch):
    B = len(batch)

    max_pair_len = max(x[0].size(0) for x in batch)
    max_meas_len = max(x[1].size(0) for x in batch)
    meas_dim = batch[0][1].size(-1)

    pair_ids = torch.zeros(B, max_pair_len, 2, dtype=torch.long)
    pair_mask = torch.zeros(B, max_pair_len, dtype=torch.float32)

    meas_x = torch.zeros(B, max_meas_len, meas_dim, dtype=torch.float32)
    meas_mask = torch.zeros(B, max_meas_len, dtype=torch.float32)

    labels = torch.zeros(B, dtype=torch.float32)

    for i, (pairs, meas, y) in enumerate(batch):
        Lp = pairs.size(0)
        Lm = meas.size(0)

        pair_ids[i, :Lp] = pairs
        pair_mask[i, :Lp] = 1.0

        meas_x[i, :Lm] = meas
        meas_mask[i, :Lm] = 1.0

        labels[i] = y

    return pair_ids, pair_mask, meas_x, meas_mask, labels


# =========================================================
# 3. Pin / Part Token Encoder
# =========================================================

class PinPartTokenEncoder(nn.Module):
    def __init__(
        self,
        num_parts,
        num_pins,
        emb_dim=64,
        hidden_dim=128,
    ):
        super().__init__()

        self.part_emb = nn.Embedding(num_parts, emb_dim)
        self.pin_emb = nn.Embedding(num_pins, emb_dim)

        self.pair_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )

    def forward(self, pair_ids):
        """
        pair_ids: (B, L, 2)
            [..., 0] = part_id
            [..., 1] = pin_id

        return:
            pair_tokens: (B, L, emb_dim)
        """

        part_ids = pair_ids[..., 0]
        pin_ids = pair_ids[..., 1]

        part_e = self.part_emb(part_ids)
        pin_e = self.pin_emb(pin_ids)

        token = torch.cat([part_e, pin_e], dim=-1)
        token = self.pair_mlp(token)

        # 동일 pin 사용성을 강조하는 residual
        token = token + pin_e

        return token


# =========================================================
# 4. Measurement Row Encoder
# =========================================================

class MeasurementRowEncoder(nn.Module):
    def __init__(
        self,
        meas_dim,
        token_dim=64,
        hidden_dim=128,
    ):
        super().__init__()

        self.row_mlp = nn.Sequential(
            nn.Linear(meas_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim),
            nn.ReLU(),
        )

    def forward(self, meas_x):
        """
        meas_x: (B, M, meas_dim)
        return:
            row_tokens: (B, M, token_dim)
        """

        return self.row_mlp(meas_x)


# =========================================================
# 5. Anomaly-aware Cross Attention Encoder
# =========================================================

class AnomalyAwareCrossEncoder(nn.Module):
    def __init__(
        self,
        token_dim=64,
        num_heads=4,
        fusion_dim=128,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.row_gate = nn.Sequential(
            nn.Linear(token_dim + 1, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 1),
        )

        self.pair_risk_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 3 + 1, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        pair_tokens,
        pair_mask,
        row_tokens,
        meas_x,
        meas_mask,
    ):
        """
        pair_tokens: (B, L, D)
        pair_mask:   (B, L)

        row_tokens:  (B, M, D)
        meas_x:      (B, M, meas_dim)
        meas_mask:   (B, M)
        """

        # Standard scaling된 값이므로 abs가 anomaly strength
        row_anomaly = meas_x.abs().mean(dim=-1)          # (B, M)
        row_anomaly = row_anomaly * meas_mask

        # measurement row가 pin/part token을 attend
        key_padding_mask = pair_mask == 0

        context_tokens, attn_weights = self.cross_attn(
            query=row_tokens,
            key=pair_tokens,
            value=pair_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        # row token + 관련 pin/part context
        row_context = row_tokens + context_tokens

        # anomaly-aware gated pooling
        gate_input = torch.cat(
            [
                row_context,
                row_anomaly.unsqueeze(-1),
            ],
            dim=-1,
        )

        gate_logit = self.row_gate(gate_input).squeeze(-1)

        gate_logit = gate_logit.masked_fill(meas_mask == 0, -1e9)
        row_weight = torch.softmax(gate_logit, dim=1)    # (B, M)

        pooled_row = torch.sum(
            row_context * row_weight.unsqueeze(-1),
            dim=1,
        )

        # anomaly 자체 기준 pooling도 별도로 추가
        anomaly_logit = row_anomaly.masked_fill(meas_mask == 0, -1e9)
        anomaly_weight = torch.softmax(anomaly_logit, dim=1)

        pooled_anomaly_row = torch.sum(
            row_context * anomaly_weight.unsqueeze(-1),
            dim=1,
        )

        global_anomaly_score = (
            row_anomaly.sum(dim=1)
            / meas_mask.sum(dim=1).clamp(min=1.0)
        ).unsqueeze(-1)

        # pair risk
        pair_risk_logit = self.pair_risk_head(pair_tokens).squeeze(-1)
        pair_risk_logit = pair_risk_logit.masked_fill(pair_mask == 0, -1e9)

        pair_risk_weight = torch.softmax(pair_risk_logit, dim=1)

        pooled_pair_risk = torch.sum(
            pair_tokens * pair_risk_weight.unsqueeze(-1),
            dim=1,
        )

        fused = torch.cat(
            [
                pooled_row,
                pooled_anomaly_row,
                pooled_pair_risk,
                global_anomaly_score,
            ],
            dim=-1,
        )

        state = self.fusion(fused)

        aux = {
            "row_anomaly": row_anomaly,
            "row_weight": row_weight,
            "anomaly_weight": anomaly_weight,
            "pair_risk_weight": pair_risk_weight,
            "attn_weights": attn_weights,
            "global_anomaly_score": global_anomaly_score.squeeze(-1),
        }

        return state, aux


# =========================================================
# 6. Full Model: Pass/Fail + PPO
# =========================================================

class AnomalyPinPartPPOModel(nn.Module):
    def __init__(
        self,
        num_parts,
        num_pins,
        meas_dim,
        token_dim=64,
        fusion_dim=128,
        action_dim=4,
        num_heads=4,
    ):
        super().__init__()

        self.pinpart_encoder = PinPartTokenEncoder(
            num_parts=num_parts,
            num_pins=num_pins,
            emb_dim=token_dim,
            hidden_dim=128,
        )

        self.measurement_encoder = MeasurementRowEncoder(
            meas_dim=meas_dim,
            token_dim=token_dim,
            hidden_dim=128,
        )

        self.cross_encoder = AnomalyAwareCrossEncoder(
            token_dim=token_dim,
            num_heads=num_heads,
            fusion_dim=fusion_dim,
        )

        self.passfail_head = nn.Linear(fusion_dim, 1)
        self.policy_head = nn.Linear(fusion_dim, action_dim)
        self.value_head = nn.Linear(fusion_dim, 1)

    def forward(
        self,
        pair_ids,
        pair_mask,
        meas_x,
        meas_mask,
    ):
        pair_tokens = self.pinpart_encoder(pair_ids)
        row_tokens = self.measurement_encoder(meas_x)

        state, aux = self.cross_encoder(
            pair_tokens=pair_tokens,
            pair_mask=pair_mask,
            row_tokens=row_tokens,
            meas_x=meas_x,
            meas_mask=meas_mask,
        )

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
    def act(
        self,
        pair_ids,
        pair_mask,
        meas_x,
        meas_mask,
        epsilon=0.05,
    ):
        out = self.forward(pair_ids, pair_mask, meas_x, meas_mask)

        logits = out["action_logits"]
        dist = Categorical(logits=logits)

        if random.random() < epsilon:
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
# 7. Loss
# =========================================================

def attention_anomaly_regularization(
    row_weight,
    anomaly_weight,
    labels,
    eps=1e-8,
):
    """
    Fail sample에서는 row attention이 anomaly 큰 row와 비슷해지도록 유도.
    Pass sample에는 강하게 적용하지 않음.
    """

    labels = labels.float()

    p = anomaly_weight.clamp(min=eps)
    q = row_weight.clamp(min=eps)

    kl = (p * (p.log() - q.log())).sum(dim=1)

    loss = (kl * labels).sum() / labels.sum().clamp(min=1.0)

    return loss


def train_passfail_epoch(
    model,
    loader,
    optimizer,
    device,
    pos_weight=None,
    attn_reg_coef=0.05,
):
    model.train()

    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=device,
        )
    else:
        pos_weight_tensor = None

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_n = 0

    for pair_ids, pair_mask, meas_x, meas_mask, labels in loader:
        pair_ids = pair_ids.to(device)
        pair_mask = pair_mask.to(device)
        meas_x = meas_x.to(device)
        meas_mask = meas_mask.to(device)
        labels = labels.to(device)

        out = model(pair_ids, pair_mask, meas_x, meas_mask)

        cls_loss = F.binary_cross_entropy_with_logits(
            out["passfail_logit"],
            labels,
            pos_weight=pos_weight_tensor,
        )

        reg_loss = attention_anomaly_regularization(
            row_weight=out["aux"]["row_weight"],
            anomaly_weight=out["aux"]["anomaly_weight"],
            labels=labels,
        )

        loss = cls_loss + attn_reg_coef * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_reg += reg_loss.item() * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "cls_loss": total_cls / max(total_n, 1),
        "attn_reg": total_reg / max(total_n, 1),
    }


# =========================================================
# 8. PPO Update
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
    attn_reg_coef=0.02,
):
    pair_ids = batch["pair_ids"].to(device)
    pair_mask = batch["pair_mask"].to(device)
    meas_x = batch["meas_x"].to(device)
    meas_mask = batch["meas_mask"].to(device)

    actions = batch["actions"].to(device)
    old_log_probs = batch["old_log_probs"].to(device)
    returns = batch["returns"].to(device)
    advantages = batch["advantages"].to(device)

    advantages = (advantages - advantages.mean()) / (
        advantages.std().clamp(min=1e-6)
    )

    out = model(pair_ids, pair_mask, meas_x, meas_mask)

    dist = Categorical(logits=out["action_logits"])
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(out["value"], returns)

    loss = (
        policy_loss
        + value_coef * value_loss
        - entropy_coef * entropy
    )

    passfail_loss = torch.tensor(0.0, device=device)
    reg_loss = torch.tensor(0.0, device=device)

    if "labels" in batch and batch["labels"] is not None:
        labels = batch["labels"].to(device).float()

        passfail_loss = F.binary_cross_entropy_with_logits(
            out["passfail_logit"],
            labels,
        )

        reg_loss = attention_anomaly_regularization(
            row_weight=out["aux"]["row_weight"],
            anomaly_weight=out["aux"]["anomaly_weight"],
            labels=labels,
        )

        loss = (
            loss
            + passfail_coef * passfail_loss
            + attn_reg_coef * reg_loss
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "passfail_loss": float(passfail_loss.item()),
        "attn_reg": float(reg_loss.item()),
    }


# =========================================================
# 9. Reward from Pass/Fail
# =========================================================

@torch.no_grad()
def reward_from_passfail(
    model,
    pair_ids,
    pair_mask,
    meas_x,
    meas_mask,
):
    out = model(pair_ids, pair_mask, meas_x, meas_mask)
    p_pass = torch.sigmoid(out["passfail_logit"])

    # [-1, +1]
    reward = 2.0 * p_pass - 1.0

    return reward


# =========================================================
# 10. Save / Load
# =========================================================

def save_model(model, path="anomaly_pinpart_ppo.pt"):
    torch.save(model.state_dict(), path)


def load_model(path, model, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# =========================================================
# 11. Example Usage
# =========================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N = 200
    meas_dim = 8

    pins = [f"PIN{i:02d}" for i in range(44)]
    parts = [f"PART{i:02d}" for i in range(17)]

    rng = np.random.default_rng(42)

    equipment_dicts = []
    measurement_sets = []
    labels = []

    for _ in range(N):
        k = rng.integers(2, 8)

        selected_pins = rng.choice(
            pins,
            size=k,
            replace=False,
        )

        eq = {}

        for pin in selected_pins:
            eq[str(pin)] = str(rng.choice(parts))

        M = rng.integers(10, 60)

        # 이미 Standard Scaling되었다고 가정
        meas = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(M, meas_dim),
        ).astype(np.float32)

        # 예시용 fake label
        anomaly_score = np.abs(meas).mean()
        y = 1 if anomaly_score > 0.82 else 0

        equipment_dicts.append(eq)
        measurement_sets.append(meas)
        labels.append(y)

    pin2id, part2id = build_vocab(equipment_dicts)
    save_vocab(pin2id, part2id, "vocab.json")

    dataset = AnomalyJointDataset(
        equipment_dicts=equipment_dicts,
        measurement_sets=measurement_sets,
        labels=labels,
        pin2id=pin2id,
        part2id=part2id,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=anomaly_collate_fn,
    )

    model = AnomalyPinPartPPOModel(
        num_parts=len(part2id),
        num_pins=len(pin2id),
        meas_dim=meas_dim,
        token_dim=64,
        fusion_dim=128,
        action_dim=4,
        num_heads=4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    for epoch in range(1, 21):
        log = train_passfail_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            pos_weight=None,
            attn_reg_coef=0.05,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={log['loss']:.6f} "
            f"cls={log['cls_loss']:.6f} "
            f"attn_reg={log['attn_reg']:.6f}"
        )

    pair_ids, pair_mask, meas_x, meas_mask, labels = next(iter(loader))

    pair_ids = pair_ids.to(device)
    pair_mask = pair_mask.to(device)
    meas_x = meas_x.to(device)
    meas_mask = meas_mask.to(device)

    action, log_prob, value, out = model.act(
        pair_ids,
        pair_mask,
        meas_x,
        meas_mask,
        epsilon=0.05,
    )

    reward = reward_from_passfail(
        model,
        pair_ids,
        pair_mask,
        meas_x,
        meas_mask,
    )

    print("action:", action[:5])
    print("reward:", reward[:5])
    print("pass_prob:", torch.sigmoid(out["passfail_logit"][:5]))

    save_model(model, "anomaly_pinpart_ppo.pt")