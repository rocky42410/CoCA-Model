import argparse, os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------- must match train_roca.py ----------
class TCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, num_blocks=3, dropout_p=0.45, pool=2):
        super().__init__()
        layers, c_in = [], in_dim
        for b in range(num_blocks):
            layers += [
                nn.Conv1d(c_in, hidden, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=pool, stride=pool),
            ]
            if b == 0 and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            c_in = hidden
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Seq2SeqLSTM(nn.Module):
    def __init__(self, c_feat: int, hidden_lstm: int = 128, num_layers: int = 3):
        super().__init__()
        self.c_feat = c_feat
        self.enc = nn.LSTM(input_size=c_feat, hidden_size=hidden_lstm, num_layers=num_layers, batch_first=True)
        self.dec = nn.LSTM(input_size=hidden_lstm, hidden_size=hidden_lstm, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_lstm, c_feat)
    def forward(self, z_seq):
        B, C, Lp = z_seq.shape
        x = z_seq.transpose(1, 2)               # [B, L', C]
        _, (h, c) = self.enc(x)
        dec_in = torch.zeros(B, Lp, h.shape[-1], device=z_seq.device, dtype=z_seq.dtype)
        y, _ = self.dec(dec_in, (h, c))
        z_rec = self.proj(y)
        return z_rec.transpose(1, 2)            # [B, C, L']

class Projector(nn.Module):
    def __init__(self, in_dim, proj_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, proj_dim, kernel_size=1, bias=True),
        )
    def forward(self, z):
        q_seq = self.net(z)           # [B, P, L']
        q = q_seq.mean(dim=-1)        # [B, P]
        return nn.functional.normalize(q, dim=1)

class RoCANet(nn.Module):
    def __init__(self, in_dim, num_tcn_blocks=3, tcn_hidden=64, proj_dim=128, proj_hidden=256,
                 lstm_hidden=128, lstm_layers=3):
        super().__init__()
        self.encoder = TCNEncoder(in_dim, hidden=tcn_hidden, num_blocks=num_tcn_blocks)
        self.seq2seq = Seq2SeqLSTM(c_feat=tcn_hidden, hidden_lstm=lstm_hidden, num_layers=lstm_layers)
        self.projector = Projector(in_dim=tcn_hidden, proj_dim=proj_dim, hidden=proj_hidden)
    def forward(self, x):
        z = self.encoder(x)           # [B, C, L']
        z_rec = self.seq2seq(z)       # [B, C, L']
        q = self.projector(z)         # [B, P]
        q_rec = self.projector(z_rec) # [B, P]
        return q, q_rec, z, z_rec

# ---------- helpers ----------
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(a, dim=1)
    b = nn.functional.normalize(b, dim=1)
    return (a * b).sum(dim=1)  # [B]

@torch.no_grad()
def batch_center(q: torch.Tensor, qp: torch.Tensor) -> torch.Tensor:
    c = (q + qp).mean(dim=0, keepdim=True)
    return nn.functional.normalize(c, dim=1)    # [1, P]

def make_windows(x: np.ndarray, L: int, delta: int):
    T = x.shape[0]
    starts = np.arange(0, max(1, T - L + 1), delta, dtype=int)
    xs = [x[s:s+L] for s in starts if s+L <= T]
    return np.stack(xs, axis=0), starts  # [N, L, D], [N]

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    in_dim = ckpt["in_dim"]
    net = RoCANet(
        in_dim=in_dim,
        num_tcn_blocks=cfg["tcn_blocks"],
        tcn_hidden=cfg["tcn_hidden"],
        proj_dim=cfg["proj_dim"],
        proj_hidden=cfg["proj_hidden"],
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
    ).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    stats = {"mean": np.array(ckpt["standardize_mean"]), "std": np.array(ckpt["standardize_std"])}
    meta = {"window": ckpt["window"], "stride": ckpt["stride"], "in_dim": in_dim}
    return net, stats, meta

def standardize(x, mean, std):
    return (x - mean) / (std + 1e-8)

def main():
    ap = argparse.ArgumentParser(description="Score CSV with trained RoCA model")
    ap.add_argument("--csv", required=True, help="Path to test CSV (rows=time, columns=features)")
    ap.add_argument("--model", required=True, help="Path to trained model .pt")
    ap.add_argument("--cols", default=None, help="Comma-separated subset of columns to use (must match training order)")
    ap.add_argument("--window", type=int, default=None, help="Override window length (defaults to training)")
    ap.add_argument("--stride", type=int, default=None, help="Override stride (defaults to training)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=None, help="Fixed threshold; if set, skips z-score rule")
    ap.add_argument("--z_k", type=float, default=3.0, help="Label 1 if score > mean + k*std (used when --threshold is not set)")
    ap.add_argument("--out_csv", default="roca_scores.csv")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, stats, meta = load_model(args.model, device)

    df = pd.read_csv(args.csv)
    if args.cols:
        df = df[args.cols.split(",")]
    X = df.values.astype(np.float32)

    # Check feature dimensionality
    if X.shape[1] != meta["in_dim"]:
        raise ValueError(f"Feature dimension mismatch: test has D={X.shape[1]}, model expects D={meta['in_dim']}.")

    # Standardize using training stats
    X = standardize(X, stats["mean"], stats["std"])

    # Windowing
    W = args.window if args.window is not None else meta["window"]
    S = args.stride if args.stride is not None else meta["stride"]
    win, starts = make_windows(X, W, S)         # [N, L, D], [N]
    if win.size == 0:
        raise ValueError("No windows produced; try smaller --window or ensure CSV has enough rows.")

    # To torch: [N, D, L]
    win_t = torch.from_numpy(win.transpose(0, 2, 1)).to(device)

    scores = []
    with torch.no_grad():
        # mini-batch to avoid OOM
        for i in range(0, win_t.shape[0], args.batch_size):
            xb = win_t[i:i+args.batch_size]              # [B, D, L]
            q, qp, _, _ = model(xb)                      # [B, P]
            Ce = batch_center(q, qp)                     # [1, P] (batch estimate)
            s = 2.0 - cosine_sim(q, Ce.expand_as(q)) - cosine_sim(qp, Ce.expand_as(qp))
            scores.append(s.detach().cpu().numpy())
    scores = np.concatenate(scores, axis=0)              # [N]

    # Thresholding
    if args.threshold is not None:
        thr = float(args.threshold)
    else:
        mu, sd = float(scores.mean()), float(scores.std() + 1e-12)
        thr = mu + args.z_k * sd

    labels = (scores > thr).astype(np.int32)

    # Align each window's score to its END index (end = start + W - 1)
    ends = starts + W - 1
    out = pd.DataFrame({
        "window_start_idx": starts,
        "window_end_idx": ends,
        "score": scores,
        "label": labels
    })

    # Optional: if you want a per-row table the same length as input,
    # you could propagate each window score to its end index.
    # Here we only emit one row per window for clarity and no double-counting.

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}\nThreshold used: {thr:.6f}\nWindows: {len(scores)}  (window={W}, stride={S})")

if __name__ == "__main__":
    main()
