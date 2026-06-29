import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# =========================================================
# 1. Frozen Pin/Part Encoder
# =========================================================

class PinPartAssignEncoder(nn.Module):
    def __init__(self, num_parts=17, num_pins=44, emb_dim=32, hidden_dim=64, out_dim=32):
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

    def forward_tokens(self, assign):
        """
        assign: (B, 44), -1 or part_id
        return: pin/part tokens (B, 44, emb_dim)
        """
        B, num_pins = assign.shape
        device = assign.device

        mask = (assign >= 0).float()

        part_ids = torch.where(
            assign >= 0,
            assign,
            torch.full_like(assign, self.none_id),
        )

        pin_ids = torch.arange(num_pins, device=device).unsqueeze(0).expand(B, -1)

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
        z = F.normalize(z, dim=-1)
        return z


def load_frozen_pinpart_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

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
# 2. Pin-conditioned MIL + PPO Model
# =========================================================

class PinConditionedMILPPO(nn.Module):
    def __init__(
        self,
        frozen_pinpart_encoder,
        meas_dim,
        token_dim=32,
        row_hidden_dim=128,
        mil_hidden_dim=128,
        action_dim=4,
        noisy_or_alpha=0.5,
    ):
        super().__init__()

        self.pinpart_encoder = frozen_pinpart_encoder
        self.noisy_or_alpha = noisy_or_alpha

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
            nn.Linear(token_dim + 1, mil_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(mil_hidden_dim),
            nn.Linear(mil_hidden_dim, 1),
        )

        self.attn_head = nn.Sequential(
            nn.Linear(token_dim + 1, mil_hidden_dim),
            nn.Tanh(),
            nn.Linear(mil_hidden_dim, 1),
        )

        self.state_proj = nn.Sequential(
            nn.Linear(token_dim + 3, mil_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(mil_hidden_dim),
            nn.Linear(mil_hidden_dim, mil_hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(mil_hidden_dim, action_dim)
        self.value_head = nn.Linear(mil_hidden_dim, 1)

    def forward(self, assign, meas_x, meas_mask):
        """
        assign:    (B, 44)
        meas_x:    (B, M, meas_dim), standard-scaled
        meas_mask: (B, M)
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

        row_anomaly = meas_x.abs().mean(dim=-1) * meas_mask
        row_input = torch.cat([row_context, row_anomaly.unsqueeze(-1)], dim=-1)

        row_logit = self.row_risk_head(row_input).squeeze(-1)
        row_logit = row_logit.masked_fill(meas_mask == 0, -1e9)
        row_prob = torch.sigmoid(row_logit) * meas_mask

        # Noisy-OR MIL
        safe_prob = (1.0 - row_prob).clamp(min=1e-6, max=1.0)
        noisy_or_prob = 1.0 - torch.prod(
            torch.where(meas_mask > 0, safe_prob, torch.ones_like(safe_prob)),
            dim=1,
        )

        # Attention MIL
        attn_logit = self.attn_head(row_input).squeeze(-1)
        attn_logit = attn_logit.masked_fill(meas_mask == 0, -1e9)
        row_attn = torch.softmax(attn_logit, dim=1)

        attn_prob = (row_attn * row_prob).sum(dim=1)

        bag_prob = (
            self.noisy_or_alpha * noisy_or_prob
            + (1.0 - self.noisy_or_alpha) * attn_prob
        )

        bag_prob = bag_prob.clamp(min=1e-6, max=1.0 - 1e-6)

        pooled_row = (row_context * row_attn.unsqueeze(-1)).sum(dim=1)

        global_anomaly = (
            row_anomaly.sum(dim=1)
            / meas_mask.sum(dim=1).clamp(min=1.0)
        ).unsqueeze(-1)

        max_row_prob = row_prob.max(dim=1).values.unsqueeze(-1)
        bag_prob_feat = bag_prob.unsqueeze(-1)

        state = torch.cat(
            [pooled_row, global_anomaly, max_row_prob, bag_prob_feat],
            dim=-1,
        )

        state = self.state_proj(state)

        action_logits = self.policy_head(state)
        value = self.value_head(state).squeeze(-1)

        return {
            "bag_prob": bag_prob,
            "row_prob": row_prob,
            "row_attn": row_attn,
            "row_anomaly": row_anomaly,
            "cross_weights": cross_weights,
            "state": state,
            "action_logits": action_logits,
            "value": value,
        }

    @torch.no_grad()
    def act(self, assign, meas_x, meas_mask, epsilon=0.05):
        out = self.forward(assign, meas_x, meas_mask)
        dist = Categorical(logits=out["action_logits"])

        if np.random.rand() < epsilon:
            action = torch.randint(
                0,
                out["action_logits"].size(-1),
                size=(assign.size(0),),
                device=assign.device,
            )
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        value = out["value"]

        return action, log_prob, value, out


# =========================================================
# 3. Reward
# =========================================================

@torch.no_grad()
def compute_improvement_reward(
    model,
    assign,
    meas_before,
    mask_before,
    meas_after,
    mask_after,
    action_cost=None,
    row_improve_coef=0.7,
    bag_risk_coef=0.3,
    new_bad_coef=0.2,
):
    """
    reward =
        0.7 * 기존 fail row의 anomaly 감소량
      + 0.3 * 전체 bag risk 감소량
      - 0.2 * 새 anomaly 증가 penalty
      - action_cost
    """

    before = model(assign, meas_before, mask_before)
    after = model(assign, meas_after, mask_after)

    row_risk_before = before["row_prob"].detach()

    anom_before = meas_before.abs().mean(dim=-1) * mask_before
    anom_after = meas_after.abs().mean(dim=-1) * mask_after

    # 길이가 같다는 전제. 다르면 공통 길이만 비교.
    M = min(anom_before.size(1), anom_after.size(1))
    anom_before = anom_before[:, :M]
    anom_after = anom_after[:, :M]
    row_risk_before = row_risk_before[:, :M]

    improvement = anom_before - anom_after

    weighted_improvement = (
        row_risk_before * improvement
    ).sum(dim=1) / row_risk_before.sum(dim=1).clamp(min=1e-6)

    bag_risk_reward = before["bag_prob"] - after["bag_prob"]

    new_bad_penalty = F.relu(anom_after - anom_before).mean(dim=1)

    reward = (
        row_improve_coef * weighted_improvement
        + bag_risk_coef * bag_risk_reward
        - new_bad_coef * new_bad_penalty
    )

    if action_cost is not None:
        reward = reward - action_cost.to(reward.device)

    return reward, {
        "weighted_improvement": weighted_improvement,
        "bag_risk_before": before["bag_prob"],
        "bag_risk_after": after["bag_prob"],
        "bag_risk_reward": bag_risk_reward,
        "new_bad_penalty": new_bad_penalty,
        "row_risk_before": before["row_prob"],
        "row_risk_after": after["row_prob"],
    }


# =========================================================
# 4. PPO Update
# =========================================================

def ppo_update(
    model,
    optimizer,
    batch,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    assign = batch["assign"]
    meas_x = batch["meas_x"]
    meas_mask = batch["meas_mask"]

    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    returns = batch["returns"]
    advantages = batch["advantages"]

    advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-6)

    out = model(assign, meas_x, meas_mask)

    dist = Categorical(logits=out["action_logits"])
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(out["value"], returns)

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }


# =========================================================
# 5. Utility
# =========================================================

def make_mask(lengths, max_len=None, device="cpu"):
    if max_len is None:
        max_len = int(lengths.max())
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    return (arange < lengths.unsqueeze(1)).float()


# =========================================================
# 6. Example
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinpart_ckpt", type=str, required=True)
    parser.add_argument("--meas_dim", type=int, required=True)
    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frozen_encoder = load_frozen_pinpart_encoder(args.pinpart_ckpt, device)

    model = PinConditionedMILPPO(
        frozen_pinpart_encoder=frozen_encoder,
        meas_dim=args.meas_dim,
        token_dim=32,
        row_hidden_dim=128,
        mil_hidden_dim=128,
        action_dim=args.action_dim,
        noisy_or_alpha=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    # -----------------------------
    # Dummy online example
    # 실제 환경에서는 아래 부분을 네 장비 loop로 교체
    # -----------------------------
    B = 8
    M = 40
    meas_dim = args.meas_dim

    assign = torch.randint(-1, 17, (B, 44), device=device)
    assign[assign < 3] = -1

    meas_before = torch.randn(B, M, meas_dim, device=device)
    meas_after = meas_before * 0.8 + 0.1 * torch.randn(B, M, meas_dim, device=device)

    lengths = torch.full((B,), M, device=device)
    mask = make_mask(lengths, max_len=M, device=device)

    action, old_log_prob, value, out_before = model.act(
        assign,
        meas_before,
        mask,
        epsilon=0.05,
    )

    reward, reward_info = compute_improvement_reward(
        model=model,
        assign=assign,
        meas_before=meas_before,
        mask_before=mask,
        meas_after=meas_after,
        mask_after=mask,
        action_cost=None,
    )

    returns = reward
    advantages = reward - value.detach()

    batch = {
        "assign": assign,
        "meas_x": meas_before,
        "meas_mask": mask,
        "actions": action,
        "old_log_probs": old_log_prob.detach(),
        "returns": returns.detach(),
        "advantages": advantages.detach(),
    }

    log = ppo_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
    )

    print("action:", action)
    print("reward:", reward)
    print("bag_risk_before:", reward_info["bag_risk_before"])
    print("bag_risk_after:", reward_info["bag_risk_after"])
    print("ppo_log:", log)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meas_dim": args.meas_dim,
            "action_dim": args.action_dim,
        },
        "pin_conditioned_mil_ppo.pt",
    )


if __name__ == "__main__":
    main()
    
    