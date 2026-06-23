import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. Dataset
# =========================

class MultiHotDataset(Dataset):
    def __init__(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        self.X = X.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# =========================
# 2. Model
# =========================

class MultiHotEmbedder(nn.Module):
    def __init__(
        self,
        num_features=616,
        emb_dim=32,
        out_dim=32,
        hidden_dim=64
    ):
        super().__init__()

        self.feature_emb = nn.Embedding(num_features, emb_dim)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        """
        x: (B, 616) multi-hot tensor
        return: (B, out_dim)
        """

        # feature embedding table
        idx = torch.arange(x.size(1), device=x.device)
        E = self.feature_emb(idx)          # (616, emb_dim)

        # active feature embedding sum
        z = x @ E                          # (B, emb_dim)

        # mean pooling
        count = x.sum(dim=1, keepdim=True).clamp(min=1.0)
        z = z / count

        # projection
        z = self.proj(z)

        # cosine similarity용 normalize
        z = F.normalize(z, dim=-1)

        return z


# =========================
# 3. Train Function
# =========================

def train_multihot_embedder(
    X,
    num_features=616,
    emb_dim=32,
    out_dim=32,
    hidden_dim=64,
    batch_size=512,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
    save_path="multihot_embedder.pt"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MultiHotDataset(X)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    model = MultiHotEmbedder(
        num_features=num_features,
        emb_dim=emb_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for x in dataloader:
            x = x.to(device).float()

            # embedding similarity
            z = model(x)                   # (B, out_dim)
            pred_sim = z @ z.T             # (B, B)

            # original multi-hot cosine similarity
            x_norm = F.normalize(x, dim=-1)
            target_sim = x_norm @ x_norm.T # (B, B)

            loss = F.mse_loss(pred_sim, target_sim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        avg_loss = total_loss / total_count

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] loss = {avg_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_features": num_features,
            "emb_dim": emb_dim,
            "out_dim": out_dim,
            "hidden_dim": hidden_dim,
        },
        save_path
    )

    print(f"Saved model to: {save_path}")

    return model


# =========================
# 4. Load Function
# =========================

def load_multihot_embedder(
    path="multihot_embedder.pt",
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(path, map_location=device)

    model = MultiHotEmbedder(
        num_features=ckpt["num_features"],
        emb_dim=ckpt["emb_dim"],
        out_dim=ckpt["out_dim"],
        hidden_dim=ckpt["hidden_dim"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


# =========================
# 5. Transform Function
# =========================

@torch.no_grad()
def transform_multihot(
    model,
    X,
    batch_size=1024,
    device=None
):
    if device is None:
        device = next(model.parameters()).device

    dataset = MultiHotDataset(X)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    embeddings = []

    model.eval()

    for x in dataloader:
        x = x.to(device).float()
        z = model(x)
        embeddings.append(z.cpu())

    return torch.cat(embeddings, dim=0).numpy()


# =========================
# 6. Example Usage
# =========================

if __name__ == "__main__":

    # 예시 데이터
    # 실제로는 네 multi-hot 데이터를 여기에 넣으면 됨
    N = 10000
    X = np.zeros((N, 616), dtype=np.float32)

    rng = np.random.default_rng(42)

    for i in range(N):
        k = rng.integers(2, 8)  # 평균 5개 정도 활성
        active_idx = rng.choice(616, size=k, replace=False)
        X[i, active_idx] = 1.0

    # 학습
    model = train_multihot_embedder(
        X,
        num_features=616,
        emb_dim=32,
        out_dim=32,
        hidden_dim=64,
        batch_size=512,
        epochs=100,
        lr=1e-3,
        save_path="multihot_embedder.pt"
    )

    # embedding 변환
    Z = transform_multihot(
        model,
        X,
        batch_size=1024
    )

    print("Embedding shape:", Z.shape)

    # 저장
    np.save("multihot_embedding.npy", Z)
    print("Saved embedding to: multihot_embedding.npy")