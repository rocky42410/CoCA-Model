#!/usr/bin/env python3
"""
patch_train_roca.py — one-shot patcher for train_roca.py

What it does (idempotent):
  1) Makes cosine_sim() numerically safe (adds eps) and normalizes with eps.
  2) Makes Projector.forward() normalize with eps and nan_to_num().
  3) Adds optional TCN pooling plan: --tcn_pool_plan and threads it through TCNEncoder/RoCANet.
  4) Ensures training-time calibration: computes/saves Ce_train and tau, logs calibration stats.
  5) Saves standardize mean/std as torch tensors; keeps config values safely (incl. strings/None).
  6) Adds deterministic seeding in __main__.
"""

import sys, re, json, pathlib

def load_text(p):
    p = pathlib.Path(p)
    return p, p.read_text(encoding="utf-8")

def save_text(p, s):
    pathlib.Path(p).write_text(s, encoding="utf-8")

def replace_block(src, name, pattern, repl):
    """Replace the first regex 'pattern' with 'repl'. Return (new_src, changed:bool)."""
    new, n = re.subn(pattern, repl, src, count=1, flags=re.DOTALL)
    return new, bool(n)

def ensure_imports(src):
    changed = False
    if "import numpy as np" not in src:
        src = "import numpy as np\n" + src
        changed = True
    if "import torch.nn.functional as F" not in src and re.search(r"\bF\.", src):
        src = "import torch.nn.functional as F\n" + src
        changed = True
    return src, changed

def patch_cosine_sim(src):
    # Replace any def cosine_sim(a,b) style with eps-safe version
    pat = r"def\s+cosine_sim\s*\([^)]*\):\s*.*?return\s*\(a\s*\*\s*b\)\.sum\(dim=1\)\s*#?\s*\[B\]\s*"
    repl = (
        "def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:\n"
        "    a = nn.functional.normalize(a, dim=1, eps=eps)\n"
        "    b = nn.functional.normalize(b, dim=1, eps=eps)\n"
        "    return (a * b).sum(dim=1)  # [B]\n"
    )
    return replace_block(src, "cosine_sim", pat, repl)

def patch_projector_forward(src):
    # Replace Projector.forward body to include eps + nan_to_num
    pat = (
        r"(class\s+Projector\s*\([\s\S]*?\n\s*def\s+forward\s*\(self,\s*z\)\s*:[\s\S]*?q_seq\s*=\s*self\.net\(z\)[\s\S]*?q\s*=\s*q_seq\.mean\(dim=-1\)\s*#?\s*\[B,\s*P\]\s*\n)"
        r"(\s*q\s*=\s*nn\.functional\.normalize\(q,\s*dim=1\)\s*#?.*?\n)"
        r"(\s*return\s+q\s*)"
    )
    repl = (
        r"\1"
        r"    q = nn.functional.normalize(q, dim=1, eps=1e-8)\n"
        r"    q = torch.nan_to_num(q)\n"
        r"    return q\n"
    )
    new, changed = re.subn(pat, repl, src, count=1, flags=re.DOTALL)
    if not changed:
        # Fallback: try a simpler normalize line replace
        new2, changed2 = re.subn(r"nn\.functional\.normalize\(q,\s*dim=1\)", "nn.functional.normalize(q, dim=1, eps=1e-8)", src, count=1)
        if changed2:
            # Add nan_to_num right before return q
            new2, changed3 = re.subn(r"(\n\s*return\s+q\s*\n)", "\n    q = torch.nan_to_num(q)\n\\1", new2, count=1)
            return new2, True
    return new, changed

def patch_tcn_pool_plan(src):
    changed_any = False

    # 1) TCNEncoder signature: add pool_plan=None
    pat1 = r"(class\s+TCNEncoder\s*\([\s\S]*?def\s+__init__\s*\(self,\s*in_dim,\s*hidden=\d+,\s*num_blocks=\d+,\s*dropout_p=[0-9.]+,\s*pool=\d+\)\s*:)"
    repl1 = r"\1\n"
    if re.search(pat1, src):
        src = re.sub(pat1, lambda m: m.group(0).replace("pool=", "pool= ").rstrip(")") + ", pool_plan=None):", src, count=1, flags=re.DOTALL)
        changed_any = True

    # 2) Insert pool_plan logic inside TCNEncoder.__init__
    if "pool_plan is None" not in src:
        pat2 = r"(for\s+b\s+in\s+range\(num_blocks\)\s*:\s*\n\s*layers\s*\+=\s*\[\s*\n\s*nn\.Conv1d[\s\S]*?nn\.BatchNorm1d[\s\S]*?nn\.ReLU[\s\S]*?)(nn\.MaxPool1d\(kernel_size=\s*pool,\s*stride=\s*pool\)\s*,?\s*\]\s*,?\s*\n)"
        repl2 = (
            r"if pool_plan is None:\n"
            r"            pool_plan = [pool] * num_blocks\n"
            r"        assert len(pool_plan) == num_blocks, 'pool_plan length must equal num_blocks'\n"
            r"        for b in range(num_blocks):\n"
            r"            layers += [\n"
            r"                nn.Conv1d(c_in, hidden, kernel_size=5, padding=2, bias=False),\n"
            r"                nn.BatchNorm1d(hidden),\n"
            r"                nn.ReLU(inplace=True),\n"
            r"                nn.MaxPool1d(kernel_size=pool_plan[b], stride=pool_plan[b]),\n"
            r"            ]\n"
        )
        new, changed = replace_block(src, "TCNEncoder_pool_plan", pat2, repl2)
        if changed: src, changed_any = new, True

    # 3) RoCANet __init__ add tcn_pool_plan=None and pass to TCNEncoder
    if "tcn_pool_plan=None" not in src:
        pat3 = r"(class\s+RoCANet\s*\([\s\S]*?def\s+__init__\s*\(self,\s*in_dim,\s*num_tcn_blocks=\d+,\s*tcn_hidden=\d+,\s*proj_dim=\d+,\s*proj_hidden=\d+,\s*lstm_hidden=\d+,\s*lstm_layers=\d+\)\s*:)"
        repl3 = r"class RoCANet(nn.Module):\n    def __init__(self, in_dim, num_tcn_blocks=3, tcn_hidden=64, proj_dim=128, proj_hidden=256,\n                 lstm_hidden=128, lstm_layers=3, tcn_pool_plan=None):"
        src, ch = replace_block(src, "RoCANet_sig", pat3, repl3)
        changed_any = changed_any or ch

    if "pool_plan=tcn_pool_plan" not in src:
        pat4 = r"(self\.encoder\s*=\s*TCNEncoder\s*\(\s*in_dim,\s*hidden=\s*tcn_hidden,\s*num_blocks=\s*num_tcn_blocks\s*\))"
        repl4 = r"self.encoder = TCNEncoder(in_dim, hidden=tcn_hidden, num_blocks=num_tcn_blocks, pool_plan=tcn_pool_plan)"
        src, ch = replace_block(src, "RoCANet_encoder_pass_pool_plan", pat4, repl4)
        changed_any = changed_any or ch

    # 4) CLI: add --tcn_pool_plan and parse to list, pass to RoCANet(...)
    if "--tcn_pool_plan" not in src:
        pat5 = r"(ap\.add_argument\(\"--tcn_hidden\"[\s\S]*?\)\s*\n)"
        repl5 = r"\1    ap.add_argument(\"--tcn_pool_plan\", default=None, help=\"Comma list of pool strides per block, e.g. '2,2,1'.\")\n"
        src, ch = replace_block(src, "parse_args_add_pool_plan", pat5, repl5)
        changed_any = changed_any or ch

    if "pool_plan =" not in src or "tcn_pool_plan=pool_plan" not in src:
        pat6 = r"(model\s*=\s*RoCANet\s*\(\s*in_dim=in_dim,\s*num_tcn_blocks=args\.tcn_blocks,[\s\S]*?lstm_layers=args\.lstm_layers,?\s*\)\.to\(device\))"
        repl6 = (
            "pool_plan = None\n"
            "    if getattr(args, 'tcn_pool_plan', None):\n"
            "        pool_plan = [int(p.strip()) for p in args.tcn_pool_plan.split(',')]\n"
            "    model = RoCANet(\n"
            "        in_dim=in_dim,\n"
            "        num_tcn_blocks=args.tcn_blocks,\n"
            "        tcn_hidden=args.tcn_hidden,\n"
            "        proj_dim=args.proj_dim,\n"
            "        proj_hidden=args.proj_hidden,\n"
            "        lstm_hidden=args.lstm_hidden,\n"
            "        lstm_layers=args.lstm_layers,\n"
            "        tcn_pool_plan=pool_plan,\n"
            "    ).to(device)"
        )
        src, ch = replace_block(src, "model_construction_pool_plan", pat6, repl6)
        changed_any = changed_any or ch

    return src, changed_any

def patch_save_checkpoint_and_calibration(src):
    """
    Ensure the training script:
      - Computes Ce_train over ALL training windows
      - Computes S_train and tau (mean+3σ) (and logs)
      - Saves standardize_mean/std as torch tensors
      - Includes Ce_train and tau in the checkpoint
    We search for torch.save(...) and inject a block above it if needed.
    """
    if "Ce_train" in src and re.search(r'\"tau\"\s*:', src):
        # Already doing it.
        return src, False

    # Try to find a save block
    m = re.search(r"torch\.save\(\s*\{[\s\S]*?\}\s*,\s*args\.model_out\s*\)", src)
    if not m:
        # Could not locate save; append a robust block at the end of train(args)
        insert_anchor = re.search(r"def\s+train\s*\(\s*args\s*\)\s*:\s*[\s\S]*?^\s*for\s+epoch", src, flags=re.M)
        if not insert_anchor:
            return src, False

    # Build calibration code snippet (uses ds, model, device, cosine_sim, DataLoader, np)
    calib = r"""
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

    if m:
        # Replace existing torch.save(...) blob with the safe block
        start, end = m.span()
        new_src = src[:start] + calib + src[end:]
        return new_src, True
    else:
        # Append at end of train() before it returns; try to find "print(f\"Saved model"
        pat = r"(print\(.{0,120}Saved model.*?\)\s*)"
        new_src, changed = re.subn(pat, calib, src, count=1)
        if changed:
            return new_src, True
        # Worst case: append calib at the end of train()
        new_src, changed2 = re.subn(r"(def\s+train\s*\(\s*args\s*\)\s*:\s*[\s\S]*?)\Z", r"\1\n" + calib + "\n", src, count=1)
        return new_src, changed2

def patch_seed_block(src):
    # Make __main__ deterministic. If block exists, tweak it; else insert.
    if "torch.backends.cudnn.deterministic = True" in src:
        return src, False
    pat = r"(if\s+__name__\s*==\s*[\"']__main__[\"']\s*:\s*\n\s*args\s*=\s*parse_args\(\)\s*\n[\s\S]*?train\(args\)\s*)"
    repl = (
        "if __name__ == \"__main__\":\n"
        "    args = parse_args()\n"
        "    import random, numpy as _np\n"
        "    seed = 1337\n"
        "    random.seed(seed); _np.random.seed(seed)\n"
        "    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)\n"
        "    torch.backends.cudnn.benchmark = False\n"
        "    torch.backends.cudnn.deterministic = True\n"
        "    train(args)\n"
    )
    new, changed = re.subn(pat, repl, src, count=1, flags=re.DOTALL)
    return new, changed

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 patch_train_roca.py /path/to/train_roca.py")
        sys.exit(1)

    path, src = load_text(sys.argv[1])
    original = src
    report = []

    # Ensure imports present (np, F if used)
    src, ch = ensure_imports(src); report.append(("ensure_imports", ch))

    # 1) cosine_sim eps-safe
    src, ch = patch_cosine_sim(src); report.append(("cosine_sim_eps", ch))

    # 2) projector forward normalize + nan_to_num
    src, ch = patch_projector_forward(src); report.append(("projector_forward_eps_nan", ch))

    # 3) add tcn_pool_plan across encoder, net, CLI, construction
    src, ch = patch_tcn_pool_plan(src); report.append(("tcn_pool_plan", ch))

    # 4) safe save + calibration (Ce_train, tau)
    src, ch = patch_save_checkpoint_and_calibration(src); report.append(("save_calibration", ch))

    # 5) deterministic seeding
    src, ch = patch_seed_block(src); report.append(("deterministic_seed_main", ch))

    changed = any(ch for _, ch in report)
    if changed:
        save_text(path, src)
        print(f"[OK] Patched {path}")
    else:
        print(f"[INFO] No changes made; patches already present in {path}")

    print("\nChange report:")
    for name, ch in report:
        print(f"  - {name:28s}: {'APPLIED' if ch else 'already ok'}")

    # Sanity hint
    print("\nNext run (1-feature defaults):")
    print("  python3 train_roca.py --csv <idle.csv> --cols joint_val "
          "--window 128 --stride 16 --epochs 50 --batch_size 128 "
          "--nu 0.005 --mu 7.0 --warmup_epochs 5 "
          "--jitter_std 0.005 --scale_std 0.005 "
          "--tcn_pool_plan 2,2,1 --nan_policy drop "
          "--model_out models/roca_model.pt")

if __name__ == "__main__":
    main()
