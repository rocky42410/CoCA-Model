import argparse, math, os, random
import numpy as np
import pandas as pd
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------
# Data utilities
# ---------------------

def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # Force numeric, turn bad tokens to NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def _nan_sanity_report(df: pd.DataFrame, name: str = "CSV"):
    total = len(df)
    bad_rows = df.isna().any(axis=1).sum()
    bad_cols = df.isna().any(axis=0).sum()
    print(f"[{name}] rows={total}  rows_with_NaN={bad_rows}  cols_with_NaN={bad_cols}")
    if bad_rows > 0:
        top_cols = df.isna().sum().sort_values(ascending=False)
        print(f"[{name}] NaN by column (top 5):\n{top_cols.head(5)}")


def standardize_nanaware(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Compute mean/std ignoring NaNs
    mean = np.nanmean(x, axis=0, keepdims=True)
    std  = np.nanstd(x, axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)  # avoid div-by-zero
    xz = (x - mean) / std
    # Any residual NaN (e.g., entire column NaN) → zero-fill
    xz = np.nan_to_num(xz, nan=0.0, posinf=0.0, neginf=0.0)
    return xz.astype(np.float32), mean.squeeze(0), std.squeeze(0)

def make_windows(x: np.ndarray, L: int, delta: int) -> np.ndarray:
    T = x.shape[0]
    starts = np.arange(0, max(1, T - L + 1), delta, dtype=int)
    wins = [x[s:s+L] for s in starts if s+L <= T]
    return np.stack(wins, axis=0) if len(wins) else np.zeros((0, L, x.shape[1]), dtype=x.dtype)

class TSDataset(Dataset):
    def __init__(self, csv_path: str, window: int, stride: int,
                 cols: Optional[list]=None, jitter_std=0.0, scale_std=0.0,
                 nan_policy: str = "drop"):  # "drop" or "zero"
        df = pd.read_csv(csv_path)
        if cols is not None:
            df = df[cols]
        df = _coerce_numeric_df(df)
        _nan_sanity_report(df, name=os.path.basename(csv_path))

        if nan_policy == "drop":
            df = df.dropna(axis=0)
        elif nan_policy == "zero":
            df = df.fillna(0.0)
        else:
            raise ValueError("nan_policy must be 'drop' or 'zero'")

        x = df.values.astype(np.float32)
        x, self.mean, self.std = standardize_nanaware(x)

        self.windows = make_windows(x, window, stride).astype(np.float32)  # [N, L, D]
        if self.windows.shape[0] == 0:
            raise ValueError("No windows produced. Try smaller --window or ensure enough rows after NaN handling.")

        self.jitter_std = jitter_std
        self.scale_std  = scale_std

    def __len__(self): return self.windows.shape[0]

    def __getitem__(self, idx):
        X = self.windows[idx]  # [L, D]
        if self.jitter_std > 0:
            X = X + np.random.normal(0, self.jitter_std, size=X.shape).astype(np.float32)
        if self.scale_std > 0:
            s = np.random.normal(1.0, self.scale_std, size=(1, X.shape[1])).astype(np.float32)
            X = X * s
        return torch.from_numpy(X.T)  # [D, L]

# ---------------------
# Model components
# ---------------------
class TCNEncoder(nn.Module):
    """
    Multi-block 1D CNN encoder: Conv1d -> BN -> ReLU -> MaxPool1d (+ optional Dropout in first block)
    """
    def __init__(self, in_dim, hidden=64, num_blocks=3, dropout_p=0.45, pool= 2, pool_plan=None):
        super().__init__()
        layers = []
        c_in = in_dim
        if pool_plan is None:
            pool_plan = [pool] * num_blocks
        assert len(pool_plan) == num_blocks, 'pool_plan length must equal num_blocks'
        for b in range(num_blocks):
            layers += [
                nn.Conv1d(c_in, hidden, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=pool_plan[b], stride=pool_plan[b]),
            ]
            if b == 0 and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            c_in = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, D, L]
        z = self.net(x)     # [B, C, L']
        return z

class Seq2SeqLSTM(nn.Module):
    """
    LSTM encoder-decoder (Seq2Seq) that summarizes temporal context, then reconstructs features.
    We flatten feature channels before LSTM and reconstruct back to channel-dim by a linear layer.
    """
    def __init__(self, c_feat: int, hidden_lstm: int = 128, num_layers: int = 3):
        super().__init__()
        self.c_feat = c_feat
        self.enc = nn.LSTM(input_size=c_feat, hidden_size=hidden_lstm, num_layers=num_layers, batch_first=True)
        self.dec = nn.LSTM(input_size=hidden_lstm, hidden_size=hidden_lstm, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_lstm, c_feat)

    def forward(self, z_seq):  # [B, C, L'] -> treat as [B, L', C]
        B, C, Lp = z_seq.shape
        x = z_seq.transpose(1, 2)              # [B, L', C]
        _, (h, c) = self.enc(x)                 # summarize to hidden
        # decode L' steps, with zero inputs; (use teacher-forced hidden state)
        dec_in = torch.zeros(B, Lp, h.shape[-1], device=z_seq.device, dtype=z_seq.dtype)
        y, _ = self.dec(dec_in, (h, c))         # [B, L', H]
        z_rec = self.proj(y)                    # [B, L', C]
        return z_rec.transpose(1, 2)            # [B, C, L']

class Projector(nn.Module):
    """MLP projector with one hidden layer, BN and ReLU, then l2-normalize."""
    def __init__(self, in_dim, proj_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, proj_dim, kernel_size=1, bias=True),
        )

    def forward(self, z):  # [B, C, L'] -> produce per-time-step projections, then mean over time
        q_seq = self.net(z)                      # [B, P, L']
        q = q_seq.mean(dim=-1)                  # [B, P]
        q = nn.functional.normalize(q, dim=1, eps=1e-8)
        q = torch.nan_to_num(q)
        return q
class RoCANet(nn.Module):
    def __init__(self, in_dim, num_tcn_blocks=3, tcn_hidden=64, proj_dim=128, proj_hidden=256,
                 lstm_hidden=128, lstm_layers=3, tcn_pool_plan=None):
        super().__init__()
        self.encoder = TCNEncoder(in_dim, hidden=tcn_hidden, num_blocks=num_tcn_blocks, pool_plan=tcn_pool_plan)
        self.seq2seq = Seq2SeqLSTM(c_feat=tcn_hidden, hidden_lstm=lstm_hidden, num_layers=lstm_layers)
        self.projector = Projector(in_dim=tcn_hidden, proj_dim=proj_dim, hidden=proj_hidden)

    def forward(self, x):
        """
        x: [B, D, L]
        returns:
            q    : [B, P] projections of original rep
            q_rec: [B, P] projections of reconstructed rep
            z, z_rec for optional inspection
        """
        z = self.encoder(x)                 # [B, C, L']
        z_rec = self.seq2seq(z)             # [B, C, L']
        q = self.projector(z)               # [B, P]
        q_rec = self.projector(z_rec)       # [B, P]
        return q, q_rec, z, z_rec

# ---------------------
# RoCA losses
# ---------------------
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # both expected L2-normalized already for stability, but normalize anyway
    a = nn.functional.normalize(a, dim=1)
    b = nn.functional.normalize(b, dim=1)
    return (a * b).sum(dim=1)  # [B]

@torch.no_grad()
def one_class_center(q: torch.Tensor, q_rec: torch.Tensor) -> torch.Tensor:
    # Ce = (1/(2N)) sum_i (q_i + q'_i) on the unit hypersphere
    Ce = (q + q_rec).mean(dim=0, keepdim=True)
    Ce = nn.functional.normalize(Ce, dim=1)
    return Ce  # [1, P]

def L_inv(q, q_rec, Ce):
    # LInv = 2 - sim(q,Ce) - sim(q',Ce) (per sample)
    s1 = cosine_sim(q, Ce.expand_as(q))
    s2 = cosine_sim(q_rec, Ce.expand_as(q_rec))
    return 2.0 - s1 - s2  # [B]
    # See Eq. (4) and related invariance/contrastive relationships. :contentReference[oaicite:4]{index=4}

def L_oe(q, q_rec, Ce):
    # L_OE = 2 + sim(q,Ce) + sim(q',Ce) (per sample)
    s1 = cosine_sim(q, Ce.expand_as(q))
    s2 = cosine_sim(q_rec, Ce.expand_as(q_rec))
    return 2.0 + s1 + s2  # [B]
    # Outlier exposure term structure. :contentReference[oaicite:5]{index=5}

def L_var(q, zeta=1.0, eps=1e-4):
    # Hinge on per-feature std to avoid collapse without negatives. (VICReg-style)
    std = q.std(dim=0)  # [P]
    return torch.relu(zeta - torch.sqrt(std**2 + eps)).mean()
    # Variance term and discussion. :contentReference[oaicite:6]{index=6}

# ---------------------
# Training
# ---------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ds = TSDataset(args.csv, window=args.window, stride=args.stride,
                   cols=args.cols.split(',') if args.cols else None,
                   jitter_std=args.jitter_std, scale_std=args.scale_std)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    in_dim = ds.windows.shape[2]  # D
    pool_plan = None
    if getattr(args, 'tcn_pool_plan', None):
        pool_plan = [int(p.strip()) for p in args.tcn_pool_plan.split(',')]
    model = RoCANet(
        in_dim=in_dim,
        num_tcn_blocks=args.tcn_blocks,
        tcn_hidden=args.tcn_hidden,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        tcn_pool_plan=pool_plan,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=5e-4)

    warmup_epochs = max(1, args.warmup_epochs)
    total_steps = 0
    model.train()

    for epoch in range(args.epochs):
        for X in loader:
            X = X.to(device)  # [B, D, L]
            q, q_rec, _, _ = model(X)  # [B, P] each
            with torch.no_grad():
                Ce = one_class_center(q, q_rec)  # [1, P]

            linv = L_inv(q, q_rec, Ce)          # [B]
            if epoch < warmup_epochs:
                # warmup: only invariance to avoid initial bias
                # (paper notes early use of LInv only). :contentReference[oaicite:7]{index=7}
                y_hat = torch.zeros_like(linv, dtype=torch.long)  # placeholder
                l_joint = linv.mean()
            else:
                loe = L_oe(q, q_rec, Ce)        # [B]
                # S_train = LInv - LOE (lower is "more normal"). :contentReference[oaicite:8]{index=8}
                s_train = linv - loe
                # select top nu fraction as anomalies (yi=1) within the minibatch
                B = s_train.shape[0]
                k = max(1, int(math.ceil(args.nu * B)))
                # argsort descending; top-k are anomalies
                idx = torch.argsort(s_train, descending=True)
                yi = torch.zeros(B, device=device)
                yi[idx[:k]] = 1.0
                l_joint = ((args.mu * yi) * loe + (1.0 - yi) * linv).mean()
                # Joint term definition. :contentReference[oaicite:9]{index=9}

            # Variance (uniformity without negatives) on both q and q_rec
            l_var = 0.5 * (L_var(q, zeta=args.zeta, eps=args.epsilon) +
                           L_var(q_rec, zeta=args.zeta, eps=args.epsilon))
            loss = l_joint + args.lambda_var * l_var  # L_RoCA
            # Overall objective. :contentReference[oaicite:10]{index=10}

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_steps += 1

        if (epoch + 1) % max(1, args.log_every) == 0:
            print(f"epoch {epoch+1}/{args.epochs} | loss={loss.item():.4f} | L_joint={l_joint.item():.4f} | L_var={l_var.item():.4f}")

    model.eval()
    all_q = []
    all_qp = []
    with torch.no_grad():
        for X in DataLoader(ds, batch_size=256, shuffle=False, drop_last=False):
            X = X.to(device)
            q, qp, _, _ = model(X)
            all_q.append(q)
            all_qp.append(qp)
    Ce_train = torch.nn.functional.normalize(torch.cat(all_q+all_qp, dim=0).mean(dim=0, keepdim=True), dim=1)
    
    safe_ckpt = {
    "state_dict": model.state_dict(),
    "in_dim": int(in_dim),
    "window": int(args.window),
    "stride": int(args.stride),
    "standardize_mean": torch.from_numpy(ds.mean.astype(np.float32)),
    "standardize_std": torch.from_numpy(ds.std.astype(np.float32)),
    "Ce_train": Ce_train.cpu(),  # <-- add this
    "config": {k: v for k,v in vars(args).items() if isinstance(v,(int,float,str,bool))}
    }
    
    # Calibrate tau on training windows
    S_train = []
    with torch.no_grad():
        for X in DataLoader(ds, batch_size=256, shuffle=False):
            X = X.to(device)
            q, qp, _, _ = model(X)
            s = 2.0 - cosine_sim(q, Ce_train.expand_as(q)) - cosine_sim(qp, Ce_train.expand_as(qp))
            S_train.append(s.cpu().numpy())
    S_train = np.concatenate(S_train)
    tau = float(S_train.mean() + 3.0 * S_train.std())  # choose your k
    safe_ckpt["tau"] = tau

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    torch.save(safe_ckpt, args.model_out)
    """
    # ===== Calibration over training windows (Ce_train and tau) =====
    model.eval()
    from torch.utils.data import DataLoader
    all_q, all_qp, S_train_chunks = [], [], []
    with torch.no_grad():
        for _X in DataLoader(ds, batch_size=256, shuffle=False, drop_last=False):
            _X = _X.to(device)
            _q, _qp, _, _ = model(_X)
            all_q.append(_q); all_qp.append(_qp)
    Q_all = torch.cat(all_q + all_qp, dim=0)  # [M, P]
    Ce_train = nn.functional.normalize(Q_all.mean(dim=0, keepdim=True), dim=1)

    with torch.no_grad():
        for _X in DataLoader(ds, batch_size=256, shuffle=False, drop_last=False):
            _X = _X.to(device)
            _q, _qp, _, _ = model(_X)
            _s = 2.0 - cosine_sim(_q, Ce_train.expand_as(_q)) - cosine_sim(_qp, Ce_train.expand_as(_qp))
            S_train_chunks.append(_s.detach().cpu().numpy())
    S_train = np.concatenate(S_train_chunks)
    mu, sd = float(S_train.mean()), float(S_train.std())
    tau = float(mu + 3.0 * sd)  # alternatively: np.percentile(S_train, 99.7)

    # ===== Assemble "safe" checkpoint =====
    save_mean = torch.from_numpy(ds.mean.astype(np.float32))
    save_std  = torch.from_numpy(ds.std.astype(np.float32))
    safe_ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": int(in_dim),
        "window": int(args.window),
        "stride": int(args.stride),
        "standardize_mean": save_mean,
        "standardize_std": save_std,
        "Ce_train": Ce_train.cpu(),
        "tau": tau,
        "config": {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool)) or v is None},
    }
    torch.save(safe_ckpt, args.model_out)
    print(f"[train_roca] Saved model to: {args.model_out}")
    print(f"[train_roca] Ce_norm={Ce_train.norm().item():.6f} | S_train min/mean/std/max="
          f"{S_train.min():.6f}/{mu:.6f}/{sd:.6f}/{S_train.max():.6f} | tau={tau:.6f}")
 """
    print(f"Saved model to: {args.model_out}")

# ---------------------
# CLI
# ---------------------
def parse_args():
    ap = argparse.ArgumentParser(description="RoCA trainer for CSV time series")
    ap.add_argument("--csv", required=True, help="Path to CSV (rows=time, columns=features)")
    ap.add_argument("--cols", default=None, help="Comma-separated subset of columns to use (default: all)")
    ap.add_argument("--window", type=int, default=128, help="Sliding window length L")
    ap.add_argument("--stride", type=int, default=16, help="Sliding step delta")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs using only LInv")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--tcn_blocks", type=int, default=3)
    ap.add_argument("--tcn_hidden", type=int, default=64)
    ap.add_argument("--tcn_pool_plan", default=None, help="Comma list of pool strides per block, e.g. '2,2,1'.")
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lstm_layers", type=int, default=3)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--proj_hidden", type=int, default=256)
    ap.add_argument("--nu", type=float, default=0.01, help="Assumed contamination fraction in each batch")
    ap.add_argument("--mu", type=float, default=7.0, help="Weight of OE term for anomalies")
    ap.add_argument("--lambda_var", type=float, default=1.0, help="Weight on variance term")
    ap.add_argument("--zeta", type=float, default=1.0, help="Target stddev in variance term")
    ap.add_argument("--epsilon", type=float, default=1e-4)
    ap.add_argument("--jitter_std", type=float, default=0.01, help="Augmentation: Gaussian noise std")
    ap.add_argument("--scale_std", type=float, default=0.01, help="Augmentation: magnitude scaling std")
    ap.add_argument("--model_out", default="roca_model.pt", help="Path to save trained model")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--log_every", type=int, default=1)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    import random, numpy as _np
    seed = 1337
    random.seed(seed); _np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train(args)