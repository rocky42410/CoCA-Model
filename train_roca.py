# train_roca.py
import argparse, math, os, glob, random, itertools
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# -----------------------
# Data utilities
# -----------------------

def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # Force numeric; bad tokens -> NaN
    return df.apply(pd.to_numeric, errors="coerce")

def _nan_sanity_report(df: pd.DataFrame, name: str):
    total = len(df)
    bad_rows = df.isna().any(axis=1).sum()
    bad_cols = df.isna().any(axis=0).sum()
    print(f"[{name}] rows={total} rows_with_NaN={bad_rows} cols_with_NaN={bad_cols}")

def standardize_nanaware(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-feature z-score ignoring NaNs. NaN-only columns become zeros.
    Returns (x_std, mean, std).
    """
    mean = np.nanmean(x, axis=0)
    std  = np.nanstd(x, axis=0)
    std[std == 0] = 1.0
    x_std = (x - mean) / std
    x_std = np.nan_to_num(x_std, nan=0.0, posinf=0.0, neginf=0.0)
    return x_std, mean, std

def make_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Sliding windows within a single capture (T,F) -> (N, window, F).
    No cross-boundary mixing—caller applies per-file.
    """
    T, F = x.shape
    if T < window: 
        return np.zeros((0, window, F), dtype=np.float32)
    idxs = range(0, T - window + 1, stride)
    out = np.stack([x[i:i+window] for i in idxs], axis=0)
    return out.astype(np.float32)

# -----------------------
# Dataset + Sampler
# -----------------------

class MultiCSVTSDataset(Dataset):
    """
    Loads multiple CSV captures. Each capture:
      - coerce numeric
      - per-file z-score
      - per-file windowing
    Returns (aug_view1, aug_view2, capture_id).
    """
    def __init__(
        self,
        csv_paths: List[str],
        window: int,
        stride: int,
        jitter_std: float = 0.005,
        scale_std: float = 0.01,
        dropna_rows: bool = True,
    ):
        self.window = window
        self.stride = stride
        self.jitter_std = jitter_std
        self.scale_std = scale_std

        self.windows: List[torch.Tensor] = []
        self.capture_ids: List[int] = []
        self.capture_start: List[int] = []  # index offsets per capture (for diagnostics)

        all_windows = []
        for cap_id, p in enumerate(csv_paths):
            name = os.path.basename(p)
            df = pd.read_csv(p)
            df = _coerce_numeric_df(df)
            if dropna_rows:
                # Drop rows with any NaN (safer for time series)
                df = df.dropna(how="any")
            _nan_sanity_report(df, name)
            x = df.to_numpy(dtype=np.float64)

            xz, _, _ = standardize_nanaware(x)
            w = make_windows(xz, window, stride)  # (N, W, F)
            if w.shape[0] == 0:
                print(f"[WARN] {name}: not enough rows for window={window}")
                continue

            start_idx = sum([arr.shape[0] for arr in all_windows])
            self.capture_start.append(start_idx)
            all_windows.append(torch.from_numpy(w))  # (N,W,F)
            self.capture_ids.extend([cap_id] * w.shape[0])

        if len(all_windows) == 0:
            raise ValueError("No valid windows from provided CSVs. Check window/stride and data quality.")

        X = torch.cat(all_windows, dim=0)  # (M, W, F)
        self.windows = X
        self.capture_ids = torch.tensor(self.capture_ids, dtype=torch.long)
        self.num_captures = len(csv_paths)
        self.feature_dim = X.shape[-1]
        print(f"[DATA] total_windows={len(self)} captures={self.num_captures} feat={self.feature_dim}")

    def __len__(self):
        return self.windows.shape[0]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (W,F)
        if self.jitter_std > 0:
            x = x + torch.randn_like(x) * self.jitter_std
        if self.scale_std > 0:
            scale = 1.0 + torch.randn((1, x.shape[1]), device=x.device) * self.scale_std
            x = x * scale
        return x

    def __getitem__(self, idx):
        x = self.windows[idx]  # (W,F)
        v1 = self._augment(x.clone())
        v2 = self._augment(x.clone())
        return v1.float(), v2.float(), self.capture_ids[idx]

class BalancedBatchSampler(Sampler[List[int]]):
    """
    Ensures each batch mixes captures (domain balance).
    If some captures have fewer windows, we round-robin with cycling.
    """
    def __init__(self, capture_ids: torch.Tensor, batch_size: int, drop_last: bool = True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.capture_ids = capture_ids.cpu().numpy()

        self.by_cap: Dict[int, List[int]] = {}
        for i, cid in enumerate(self.capture_ids):
            self.by_cap.setdefault(int(cid), []).append(i)

        for k in self.by_cap:
            random.shuffle(self.by_cap[k])

        self.cap_iters = {
            k: itertools.cycle(self.by_cap[k]) for k in self.by_cap
        }
        self.cap_list = list(self.by_cap.keys())
        self.num_batches = len(capture_ids) // batch_size

    def __iter__(self):
        # round-robin: pick one index per capture until batch_size filled
        cap_idx = 0
        for _ in range(self.num_batches):
            batch = []
            while len(batch) < self.batch_size:
                cid = self.cap_list[cap_idx % len(self.cap_list)]
                batch.append(next(self.cap_iters[cid]))
                cap_idx += 1
            yield batch

    def __len__(self):
        return self.num_batches if self.drop_last else math.ceil(len(self.capture_ids) / self.batch_size)

# -----------------------
# Model
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
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, feat_dim: int, hid: int = 64, proj_dim: int = 64):
        super().__init__()
        # TCN over features-as-channels: (B,F,W) expected
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
        out, (hn, cn) = self.lstm(h)     # hn: (num_layers, B, hid)
        z = hn[-1]                       # (B, hid)
        q = F.normalize(self.proj(z), dim=-1)  # (B, D)
        return q

# -----------------------
# RoCA losses
# -----------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)

def invariance_loss(q, qp, Ce):
    # L_inv = 2 - sim(q,Ce) - sim(q',Ce)
    return (2.0 - cosine_sim(q, Ce) - cosine_sim(qp, Ce)).mean()

def oe_loss(q, qp, Ce, top_mask):
    # L_oe = 2 + sim(q,Ce) + sim(q',Ce)   applied to top-ν "suspects"
    sim_sum = (2.0 + cosine_sim(q, Ce) + cosine_sim(qp, Ce))
    if top_mask is None:
        return sim_sum.mean()
    return sim_sum[top_mask].mean() if top_mask.any() else sim_sum.mean()

def variance_hinge(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    # simple VICReg-like per-dim hinge: encourage std >= gamma
    std = torch.sqrt(z.var(dim=0) + 1e-6)
    return torch.relu(gamma - std).mean()

# -----------------------
# Train / Calibrate
# -----------------------

@torch.no_grad()
def batch_center(q: torch.Tensor, qp: torch.Tensor) -> torch.Tensor:
    Ce = (q + qp).mean(dim=0, keepdim=True)
    return F.normalize(Ce, dim=-1)

def select_top_nu(q, qp, Ce, nu: float):
    """
    Return boolean mask for top-ν samples by (high sim_sum) as "suspects".
    """
    score = 2.0 + cosine_sim(q, Ce) + cosine_sim(qp, Ce)  # larger => more suspect
    k = max(1, int(math.ceil(nu * q.shape[0])))
    _, idx = torch.topk(score, k=k, largest=True)
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask[idx] = True
    return mask

@torch.no_grad()
def compute_center_over_loader(encoder: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    all_q = []
    for v1, v2, _cid in loader:
        v1 = v1.to(device); v2 = v2.to(device)
        q1 = encoder(v1); q2 = encoder(v2)
        all_q.append(F.normalize((q1+q2)/2.0, dim=-1))
    Q = torch.cat(all_q, dim=0)
    Ce = F.normalize(Q.mean(dim=0, keepdim=True), dim=-1)
    return Ce

@torch.no_grad()
def calibrate_thresholds(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    Ce: torch.Tensor,
    per_capture: bool = False
) -> Tuple[float, Optional[Dict[int, float]]]:
    """
    Score = L_inv(q, q', Ce). Global τ = mean + 3*std.
    Optionally compute τ per capture ID for diagnostics.
    """
    scores = []
    caps = []
    for v1, v2, cid in loader:
        v1 = v1.to(device); v2 = v2.to(device)
        q1 = encoder(v1); q2 = encoder(v2)
        s = (2.0 - cosine_sim(q1, Ce) - cosine_sim(q2, Ce)).detach().cpu().numpy()
        scores.append(s)
        caps.append(cid.numpy())
    S = np.concatenate(scores, axis=0)
    mu, sd = S.mean(), S.std() if S.std() > 1e-9 else 1e-9
    tau = float(mu + 3.0 * sd)

    cap_tau = None
    if per_capture:
        cap_tau = {}
        C = np.concatenate(caps, axis=0)
        for c in np.unique(C):
            s = S[C == c]
            m, d = s.mean(), s.std() if s.std() > 1e-9 else 1e-9
            cap_tau[int(c)] = float(m + 3.0 * d)
    return tau, cap_tau

def train(args):
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # Resolve CSVs (comma list or glob)
    csvs: List[str] = []
    for token in args.csvs.split(","):
        token = token.strip()
        if any(ch in token for ch in ["*", "?", "[", "]"]):
            csvs.extend(sorted(glob.glob(token)))
        else:
            csvs.append(token)
    csvs = [c for c in csvs if os.path.isfile(c)]
    if not csvs:
        raise ValueError("No CSVs found. Provide --csvs 'a.csv,b.csv' or a glob like 'data/*.csv'.")

    ds = MultiCSVTSDataset(
        csv_paths=csvs,
        window=args.window,
        stride=args.stride,
        jitter_std=args.jitter_std,
        scale_std=args.scale_std,
        dropna_rows=True,
    )

    # Balanced batches (prevents one capture from dominating ν-selection)
    if args.balance_batches:
        sampler = BalancedBatchSampler(ds.capture_ids, batch_size=args.batch_size, drop_last=True)
        loader = DataLoader(ds, batch_sampler=sampler)
        eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    encoder = Encoder(feat_dim=ds.feature_dim, hid=args.hid, proj_dim=args.proj_dim).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warm-up: invariance only; estimate a stable center and optionally freeze it
    Ce_running = None
    for epoch in range(args.epochs):
        encoder.train()
        total = 0.0

        for v1, v2, _cid in loader:
            v1 = v1.to(device); v2 = v2.to(device)
            q1 = encoder(v1); q2 = encoder(v2)

            Ce = batch_center(q1, q2)
            if epoch < args.warmup_epochs:
                # Warm-up: invariance + small variance penalty
                L = invariance_loss(q1, q2, Ce) + args.var_w * (variance_hinge(q1) + variance_hinge(q2)) * 0.5
            else:
                # Optionally freeze center after warm-up using running estimate
                if args.freeze_center_after_warmup:
                    if Ce_running is None:
                        encoder.eval()
                        Ce_running = compute_center_over_loader(encoder, eval_loader, device)
                        encoder.train()
                    Ce_use = Ce_running
                else:
                    Ce_use = Ce
                top_mask = select_top_nu(q1, q2, Ce_use, args.nu)
                L = (
                    invariance_loss(q1, q2, Ce_use)
                    + args.mu * oe_loss(q1, q2, Ce_use, top_mask)
                    + args.var_w * (variance_hinge(q1) + variance_hinge(q2)) * 0.5
                )

            opt.zero_grad(set_to_none=True)
            L.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            opt.step()
            total += float(L.item())

        if (epoch + 1) % args.log_every == 0:
            print(f"[epoch {epoch+1}/{args.epochs}] loss={total:.4f}")

        # Refresh running center at the end of warm-up
        if args.freeze_center_after_warmup and epoch + 1 == args.warmup_epochs:
            encoder.eval()
            Ce_running = compute_center_over_loader(encoder, eval_loader, device)
            encoder.train()

    # Final center over *all* windows, then threshold calibration
    encoder.eval()
    Ce_final = compute_center_over_loader(encoder, eval_loader, device)
    tau_global, tau_per_cap = calibrate_thresholds(
        encoder, eval_loader, device, Ce_final, per_capture=args.per_capture_thresholds
    )

    save_obj = {
        "state_dict": encoder.state_dict(),
        "arch": {"feat_dim": ds.feature_dim, "hid": args.hid, "proj_dim": args.proj_dim},
        "Ce": Ce_final.detach().cpu(),
        "tau": tau_global,
        "tau_per_capture": tau_per_cap,
        "window": args.window,
        "stride": args.stride,
        "nu": args.nu,
        "mu": args.mu,
        "var_w": args.var_w,
        "jitter_std": args.jitter_std,
        "scale_std": args.scale_std,
    }
    torch.save(save_obj, args.model_out)
    print(f"[SAVE] {args.model_out}")
    print(f"[CALIBRATION] tau(global)={tau_global:.6f}")
    if tau_per_cap:
        for k, v in tau_per_cap.items():
            print(f"[CALIBRATION] tau(capture={k})={v:.6f}")

# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser("RoCA trainer (multi-CSV, per-file windows, balanced batches)")
    ap.add_argument("--csvs", type=str, required=True,
                    help="Comma-separated CSVs or glob (e.g., 'data/*.csv' or 'a.csv,b.csv').")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--nu", type=float, default=0.01, help="Top-ν fraction per batch for OE.")
    ap.add_argument("--mu", type=float, default=7.0, help="Weight for OE term.")
    ap.add_argument("--var_w", type=float, default=1.0, help="Weight for variance hinge.")
    ap.add_argument("--jitter_std", type=float, default=0.005)
    ap.add_argument("--scale_std", type=float, default=0.01)
    ap.add_argument("--balance_batches", action="store_true", help="Enable capture-balanced batches.")
    ap.add_argument("--freeze_center_after_warmup", action="store_true",
                    help="Use running center from warm-up for rest of training.")
    ap.add_argument("--per_capture_thresholds", action="store_true",
                    help="Compute per-capture τ for diagnostics (saved alongside global τ).")
    ap.add_argument("--model_out", type=str, default="roca_model.pt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed = 1337
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train(args)
