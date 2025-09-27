# score_roca.py — compatible with the updated train_roca.py
import argparse, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Utils
# -----------------------

def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

def standardize_nanaware(x: np.ndarray):
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    s[s == 0] = 1.0
    z = (x - m) / s
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z

def make_windows(x: np.ndarray, window: int, stride: int):
    T, F = x.shape
    if T < window:
        return np.zeros((0, window, F), dtype=np.float32), np.zeros((0,), dtype=int)
    starts = np.arange(0, T - window + 1, stride, dtype=int)
    W = np.stack([x[i:i+window] for i in starts], axis=0).astype(np.float32)
    return W, starts

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)

# -----------------------
# Model — must mirror train_roca.py
# -----------------------

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, d=1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
    def forward(self, x):  # (B,Cin,W)
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, feat_dim: int, hid: int = 64, proj_dim: int = 64):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(feat_dim, 64, k=5, d=1),
            TCNBlock(64, 128, k=5, d=2),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hid, num_layers=2, batch_first=True, bidirectional=False)
        self.proj = nn.Sequential(
            nn.Linear(hid, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim)
        )

    def forward(self, x):  # x: (B, W, F)
        x = x.transpose(1, 2)            # (B,F,W)
        h = self.tcn(x)                  # (B, C, W')
        h = h.transpose(1, 2)            # (B, W', C)
        _, (hn, _) = self.lstm(h)        # hn: (layers, B, hid)
        z = hn[-1]                       # (B, hid)
        q = F.normalize(self.proj(z), dim=-1)  # (B, D)
        return q

# -----------------------
# Inference
# -----------------------

def build_encoder_from_ckpt(ckpt, device):
    arch = ckpt["arch"]
    enc = Encoder(
        feat_dim=arch["feat_dim"],
        hid=arch["hid"],
        proj_dim=arch["proj_dim"],
    ).to(device)
    enc.load_state_dict(ckpt["state_dict"])
    enc.eval()
    return enc

def jitter_scale(t: torch.Tensor, jitter_std: float, scale_std: float) -> torch.Tensor:
    # t: (B, W, F). Apply small noise & per-feature scale like training.
    out = t
    if jitter_std and jitter_std > 0:
        out = out + torch.randn_like(out) * jitter_std
    if scale_std and scale_std > 0:
        # per-feature scale shared over time within a window
        B, W, F = out.shape
        scale = 1.0 + torch.randn((B, 1, F), device=out.device, dtype=out.dtype) * scale_std
        out = out * scale
    return out

def main():
    ap = argparse.ArgumentParser("Score CSV with RoCA (trainer-compatible)")
    ap.add_argument("--csv", required=True, help="Test CSV (rows=time, cols=features)")
    ap.add_argument("--model", required=True, help="Path to trained model .pt")
    ap.add_argument("--cols", default=None, help="Comma-separated subset of columns (must match training order)")
    ap.add_argument("--window", type=int, default=None, help="Override window (defaults to training)")
    ap.add_argument("--stride", type=int, default=None, help="Override stride (defaults to training)")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=None, help="Use fixed threshold (otherwise use ckpt τ; else mean+3σ)")
    ap.add_argument("--z_k", type=float, default=3.0, help="If no τ in ckpt and no --threshold: μ + k·σ")
    ap.add_argument("--no_aug", action="store_true", help="Disable augmentations; score each window once")
    ap.add_argument("--out_csv", default="roca_scores.csv")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    # Rebuild encoder to match training
    encoder = build_encoder_from_ckpt(ckpt, device)
    Ce = ckpt["Ce"].to(device)  # [1, D], already normalized
    tau_ckpt = ckpt.get("tau", None)

    # Pull training hyperparams that affect inference
    W = args.window if args.window is not None else ckpt.get("window", 128)
    S = args.stride if args.stride is not None else ckpt.get("stride", 16)
    jitter_std = 0.0 if args.no_aug else float(ckpt.get("jitter_std", 0.0))
    scale_std  = 0.0 if args.no_aug else float(ckpt.get("scale_std", 0.0))

    # Load and sanitize CSV
    df = pd.read_csv(args.csv)
    df = _coerce_numeric_df(df)
    n_nan_rows = int(df.isna().any(axis=1).sum())
    if n_nan_rows > 0:
        print(f"[WARN] Test CSV has {n_nan_rows} rows with NaN — dropping those rows.")
        df = df.dropna(axis=0)

    if args.cols:
        df = df[args.cols.split(",")]

    X = df.values.astype(np.float64)
    # Check feature dim
    inF = X.shape[1]
    if inF != ckpt["arch"]["feat_dim"]:
        raise ValueError(f"Feature mismatch: CSV has {inF}, model expects {ckpt['arch']['feat_dim']}.")

    # Standardize per file (matches training’s per-capture z-score)
    Xz = standardize_nanaware(X)

    # Windowing (per file, no cross-boundary)
    Wn, starts = make_windows(Xz, W, S)  # (N, W, F)
    if Wn.shape[0] == 0:
        raise ValueError("No windows produced. Reduce --window or ensure CSV has enough rows.")
    N = Wn.shape[0]

    # Torch tensors
    # Encoder expects (B, W, F)
    Xw = torch.from_numpy(Wn).float().to(device)

    scores = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            xb = Xw[i:i+args.batch_size]  # (B,W,F)

            if args.no_aug:
                q1 = encoder(xb)                 # (B,D)
                q2 = q1                          # identical view
            else:
                v1 = jitter_scale(xb, jitter_std, scale_std)
                v2 = jitter_scale(xb, jitter_std, scale_std)
                q1 = encoder(v1)
                q2 = encoder(v2)

            s = 2.0 - cosine_sim(q1, Ce.expand_as(q1)) - cosine_sim(q2, Ce.expand_as(q2))
            scores.append(s.detach().cpu().numpy())

    scores = np.concatenate(scores, axis=0)  # [N]

    # Threshold selection
    if args.threshold is not None:
        thr = float(args.threshold)
        thr_src = "cli"
    elif tau_ckpt is not None:
        thr = float(tau_ckpt)
        thr_src = "checkpoint"
    else:
        mu, sd = float(scores.mean()), float(scores.std() if scores.std() > 1e-12 else 1e-12)
        thr = mu + args.z_k * sd
        thr_src = f"dynamic μ+{args.z_k}σ"

    labels = (scores > thr).astype(np.int32)
    ends = starts + W - 1

    out = pd.DataFrame({
        "window_start_idx": starts,
        "window_end_idx": ends,
        "score": scores,
        "label": labels
    })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")
    print(f"Threshold used ({thr_src}): {thr:.6f}")
    print(f"Windows: {len(scores)}  (window={W}, stride={S})")
    print(f"Augmentations: {'OFF' if args.no_aug else f'jitter_std={jitter_std}, scale_std={scale_std}'}")

if __name__ == "__main__":
    main()
