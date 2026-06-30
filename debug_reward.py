def debug_raw_reward(before, after, margin=0.001):
    before = torch.as_tensor(before).float()
    after = torch.as_tensor(after).float()

    N = min(before.size(0), after.size(0))
    M = min(before.size(1), after.size(1))

    before = before[:N, :M]
    after = after[:N, :M]

    anom_b = before.abs().mean(dim=1)
    anom_a = after.abs().mean(dim=1)

    anom_mean = anom_b.mean()
    fail_weight = torch.relu(anom_b - anom_mean)

    if fail_weight.sum() < 1e-6:
        fail_weight = torch.ones_like(fail_weight)

    row_improve = anom_b - anom_a

    reward = (fail_weight * row_improve).sum() / fail_weight.sum().clamp(min=1e-6)
    improve_ratio = (fail_weight * (row_improve > margin).float()).sum() / fail_weight.sum().clamp(min=1e-6)

    print("before abs mean:", anom_b.mean().item())
    print("after abs mean :", anom_a.mean().item())
    print("row_improve mean:", row_improve.mean().item())
    print("row_improve min/max:", row_improve.min().item(), row_improve.max().item())
    print("fail_weight sum:", fail_weight.sum().item())
    print("reward:", reward.item())
    print("improve_ratio:", improve_ratio.item())