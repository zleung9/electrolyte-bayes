#!/usr/bin/env python3
"""
Bayesian Optimization for Electrolyte Formulation Screening
===========================================================
Optimizes electrolyte composition to maximize CIP or AGG fraction,
using Gaussian Process regression with simplex constraints.

Usage:
    python bayesian_opt.py --data electrolyte_data.csv --target cip
    python bayesian_opt.py --data electrolyte_data.csv --target agg --n-candidates 8
    python bayesian_opt.py --data electrolyte_data.csv --target cip --rounds 0

Input CSV format:
    round,id,LiTFSI,FEC,MDFA,TFB,ssip,cip,agg
    Round 0,1,0.0989,0.0,0.9011,0.0,0.2939,0.5425,0.1636
    ...

    - Rows with measured values (ssip, cip, agg all present) are used for GP training.
    - Rows with only composition (ssip/cip/agg empty) are treated as previous suggestions
      pending measurement (included for reference, excluded from training).
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Composition-space transform
# ============================================================

def alr_transform(X):
    """Additive Log-Ratio: (n,4) simplex -> (n,3) unconstrained, ref = TFB."""
    eps = 1e-10
    return np.log((X[:, :3] + eps) / (X[:, 3:] + eps))


def alr_inverse(Y):
    """Inverse ALR: (n,3) unconstrained -> (n,4) simplex."""
    ey = np.exp(Y)
    denom = 1.0 + ey.sum(axis=1, keepdims=True)
    return np.hstack([ey / denom, 1.0 / denom])


# ============================================================
# GP model
# ============================================================

def build_gp(random_state=42):
    """Create a Gaussian Process regressor with Matérn 5/2 kernel."""
    kernel = (ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0, 1.0], nu=2.5)
              + WhiteKernel(noise_level=0.05))
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=20,
        normalize_y=True, alpha=1e-4, random_state=random_state
    )


# ============================================================
# Acquisition functions
# ============================================================

def expected_improvement(mu, sigma, y_best, xi=0.01):
    """EI for maximization (scalar-safe)."""
    mu, sigma = mu.item(), sigma.item()
    if sigma < 1e-10:
        return 0.0
    z = (mu - y_best - xi) / sigma
    return float((mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z))


def upper_confidence_bound(mu, sigma, kappa=2.0):
    """UCB for maximization."""
    return (mu + kappa * sigma).item()


# ============================================================
# Constrained acquisition optimization
# ============================================================

def optimize_acquisition(gp, y_best, bounds, acq='ei', xi=0.03, kappa=2.0,
                         n_restarts=200):
    """
    Find composition maximizing the acquisition function.

    Optimizes 3 free variables (LiTFSI, FEC, MDFA) with TFB = 1 - sum.
    """
    b3 = [(bounds[0, 0], bounds[0, 1]),
          (bounds[1, 0], bounds[1, 1]),
          (bounds[2, 0], bounds[2, 1])]

    best_x, best_val, best_mu, best_sigma = None, np.inf, None, None

    for _ in range(n_restarts):
        # Find a random feasible starting point
        for __ in range(500):
            x0 = np.array([np.random.uniform(*b) for b in b3])
            tfb = 1.0 - x0.sum()
            if bounds[3, 0] <= tfb <= bounds[3, 1]:
                break
        else:
            continue

        def objective(x3):
            tfb = 1.0 - x3[0] - x3[1] - x3[2]
            if tfb < bounds[3, 0] or tfb > bounds[3, 1]:
                return 1e10
            comp = np.array([[x3[0], x3[1], x3[2], tfb]])
            comp = comp / comp.sum()
            x_alr = alr_transform(comp)
            mu, sigma = gp.predict(x_alr, return_std=True)
            if acq == 'ei':
                return -expected_improvement(mu[0], sigma[0], y_best, xi=xi)
            else:
                return -upper_confidence_bound(mu[0], sigma[0], kappa=kappa)

        res = minimize(objective, x0, bounds=b3, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-14})

        if res.fun < best_val and res.fun < 1e9:
            best_val = res.fun
            best_x = res.x.copy()
            tfb = 1.0 - best_x.sum()
            comp = np.array([[best_x[0], best_x[1], best_x[2], tfb]])
            comp = comp / comp.sum()
            x_alr = alr_transform(comp)
            mu, sigma = gp.predict(x_alr, return_std=True)
            best_mu, best_sigma = mu[0].item(), sigma[0].item()

    if best_x is None:
        return None, None, None, None

    tfb = 1.0 - best_x.sum()
    best_comp = np.array([best_x[0], best_x[1], best_x[2], tfb])
    return best_comp / best_comp.sum(), -best_val, best_mu, best_sigma


def suggest_candidates(gp, y_best, bounds, n=5, acq='ei', xi=0.03, kappa=2.0,
                       dedup_threshold=0.005, n_restarts=500):
    """
    Suggest n diverse candidates by repeated acquisition optimization,
    increasing xi between iterations to encourage diversity.
    """
    candidates = []
    current_xi = xi
    for i in range(n):
        comp, ei, mu, sigma = optimize_acquisition(
            gp, y_best, bounds, acq=acq, xi=current_xi, kappa=kappa,
            n_restarts=n_restarts
        )
        if comp is None:
            current_xi = xi * (1.5 + 0.5 * i)
            continue

        # Deduplicate
        dup = False
        for prev in candidates:
            if np.abs(comp - prev['composition']).max() < dedup_threshold:
                dup = True
                break
        if dup:
            # Retry with higher xi
            current_xi = xi * (1.5 + 0.5 * i) * 3.0
            comp2, ei2, mu2, sigma2 = optimize_acquisition(
                gp, y_best, bounds, acq=acq, xi=current_xi, kappa=kappa,
                n_restarts=n_restarts
            )
            if comp2 is not None:
                dup2 = False
                for prev in candidates:
                    if np.abs(comp2 - prev['composition']).max() < dedup_threshold:
                        dup2 = True
                        break
                if not dup2:
                    comp, ei, mu, sigma = comp2, ei2, mu2, sigma2
                else:
                    continue
            else:
                continue

        candidates.append({
            'composition': comp,
            'ei': ei,
            'mu': mu,
            'sigma': sigma,
        })

        # Increase xi for next iteration to encourage diversity
        current_xi = xi * (1.5 + 0.5 * i)

    return candidates


# ============================================================
# Visualization
# ============================================================

def make_figure(df_train, gp, bounds, target, candidates,
                gp_other=None, other_label='', output_path=None):
    """
    Generate a 4-panel diagnostic figure:
      1. Response contour (MDFA-TFB plane)
      2. GP fit (predicted vs observed)
      3. Training data scatter (CIP vs AGG)
      4. TFB effect curve
    """
    components = ['LiTFSI', 'FEC', 'MDFA', 'TFB']
    X_train = df_train[components].values
    X_train = X_train / X_train.sum(axis=1, keepdims=True)
    y_train = np.log(df_train[target].values.clip(min=1e-10))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Panel 1: Response contour on MDFA-TFB plane ----
    ax = axes[0, 0]
    li_fix = 0.08
    fec_fix = 0.0
    mdfa_grid = np.linspace(0.35, 1.0 - li_fix - fec_fix - 0.001, 60)
    tfb_grid = np.linspace(0.001, 1.0 - li_fix - fec_fix - 0.35, 60)
    MM, TT = np.meshgrid(mdfa_grid, tfb_grid)
    surf = np.full_like(MM, np.nan)

    for i in range(len(tfb_grid)):
        for j in range(len(mdfa_grid)):
            m, t = MM[i, j], TT[i, j]
            li = 1.0 - m - t - fec_fix
            if not (bounds[0, 0] <= li <= bounds[0, 1]):
                continue
            if not (bounds[3, 0] <= t <= bounds[3, 1]):
                continue
            comp = np.array([[li, fec_fix, m, t]])
            xa = alr_transform(comp)
            surf[i, j] = np.exp(gp.predict(xa)[0].item())

    im = ax.contourf(MM, TT, surf, levels=20, cmap='Blues' if target == 'cip' else 'Reds')
    sc = ax.scatter(X_train[:, 2], X_train[:, 3], c=np.exp(y_train),
                    s=60, edgecolors='black', linewidth=0.8,
                    cmap='Blues' if target == 'cip' else 'Reds', zorder=5)

    # Mark candidates
    for i, c in enumerate(candidates):
        ax.scatter(c['composition'][2], c['composition'][3],
                   c='gold', s=100, marker='D', edgecolors='black',
                   zorder=10)
        ax.annotate(str(i + 1), (c['composition'][2], c['composition'][3]),
                    fontsize=7, ha='center', va='bottom', color='darkred',
                    fontweight='bold')

    # Mark best observed
    best_idx = np.exp(y_train).argmax()
    ax.scatter(X_train[best_idx, 2], X_train[best_idx, 3],
               c='red', s=120, marker='*', edgecolors='black', zorder=10)

    plt.colorbar(im, ax=ax, label=f'Predicted {target.upper()}')
    ax.set_xlabel('MDFA')
    ax.set_ylabel('TFB')
    ax.set_title(f'{target.upper()} response (LiTFSI={li_fix}, FEC=0)\n'
                 f'* = best observed,  ◇ = candidates')

    # ---- Panel 2: GP fit ----
    ax = axes[0, 1]
    Xa_train = alr_transform(X_train)
    y_pred = np.exp(gp.predict(Xa_train))
    ax.scatter(np.exp(y_train), y_pred, c='steelblue', s=50, edgecolors='black')
    ax.plot([0, y_pred.max() * 1.1], [0, y_pred.max() * 1.1], 'k--', alpha=0.5)
    ax.set_xlabel(f'Observed {target.upper()}')
    ax.set_ylabel(f'Predicted {target.upper()}')
    score = gp.score(Xa_train, y_train)
    ax.set_title(f'GP Fit: {target.upper()} (R² = {score:.3f})')

    # ---- Panel 3: Training data overview ----
    ax = axes[1, 0]
    if 'cip' in df_train.columns and 'agg' in df_train.columns:
        cip_vals = df_train['cip'].values.clip(min=1e-10)
        agg_vals = df_train['agg'].values.clip(min=1e-10)
        ax.scatter(cip_vals, agg_vals, c='steelblue', s=60, edgecolors='black',
                   zorder=5)
        for i in range(len(df_train)):
            label = str(df_train.iloc[i]['id'])
            ax.annotate(label, (cip_vals[i], agg_vals[i]),
                        fontsize=7, ha='center', va='bottom')
        ax.set_xlabel('CIP')
        ax.set_ylabel('AGG')
        ax.set_title('Training Data: CIP vs AGG')
    else:
        ax.text(0.5, 0.5, f'{target.upper()} data only', transform=ax.transAxes,
                ha='center', va='center')

    # ---- Panel 4: Component effects ----
    ax = axes[1, 1]
    li = 0.08
    fec = 0.0
    tfb_range = np.linspace(bounds[3, 0] + 0.001, bounds[3, 1], 100)
    preds, los, his = [], [], []
    for t in tfb_range:
        m = 1.0 - li - fec - t
        if m < 0 or m > bounds[2, 1]:
            continue
        comp = np.array([[li, fec, m, t]])
        xa = alr_transform(comp)
        mu, si = gp.predict(xa, return_std=True)
        mu, si = mu.item(), si.item()
        preds.append(np.exp(mu))
        los.append(np.exp(mu - 1.96 * si))
        his.append(np.exp(mu + 1.96 * si))
    tr = tfb_range[:len(preds)]
    ax.plot(tr, preds, 'b-', lw=2, label=target.upper())
    ax.fill_between(tr, los, his, alpha=0.15, color='b')
    ax.set_xlabel('TFB mole fraction')
    ax.set_ylabel(f'{target.upper()}')
    ax.set_title(f'TFB Effect (LiTFSI={li}, FEC=0)')
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {output_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for Electrolyte Screening'
    )
    parser.add_argument('--data', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--target', required=True, choices=['cip', 'agg'],
                        help='Target to maximize: cip or agg')
    parser.add_argument('--rounds', default=None,
                        help='Comma-separated round labels to include (default: all)')
    parser.add_argument('--exclude-rounds', default=None,
                        help='Comma-separated round labels to exclude')
    parser.add_argument('--n-candidates', type=int, default=5,
                        help='Number of candidates to suggest (default: 5)')
    parser.add_argument('--acq', default='ei', choices=['ei', 'ucb'],
                        help='Acquisition function (default: ei)')
    parser.add_argument('--xi', type=float, default=0.03,
                        help='Exploration parameter for EI (default: 0.03)')
    parser.add_argument('--kappa', type=float, default=2.0,
                        help='Exploration parameter for UCB (default: 2.0)')
    parser.add_argument('--output', default='.',
                        help='Output directory (default: current)')
    parser.add_argument('--bounds-li', default='0.03,0.15',
                        help='LiTFSI bounds: min,max (default: 0.03,0.15)')
    parser.add_argument('--bounds-fec', default='0.0,0.60',
                        help='FEC bounds: min,max (default: 0.0,0.60)')
    parser.add_argument('--bounds-mdfa', default='0.0,0.95',
                        help='MDFA bounds: min,max (default: 0.0,0.95)')
    parser.add_argument('--bounds-tfb', default='0.001,0.50',
                        help='TFB bounds: min,max (default: 0.001,0.50)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--use-log', action='store_true', default=True,
                        help='Model log(target) instead of raw target (default)')
    parser.add_argument('--no-log', dest='use_log', action='store_false',
                        help='Model raw target without log transform')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # ---- Parse bounds ----
    bounds = np.array([
        [float(x) for x in args.bounds_li.split(',')],
        [float(x) for x in args.bounds_fec.split(',')],
        [float(x) for x in args.bounds_mdfa.split(',')],
        [float(x) for x in args.bounds_tfb.split(',')],
    ])

    # ---- Load data ----
    df = pd.read_csv(args.data)
    components = ['LiTFSI', 'FEC', 'MDFA', 'TFB']

    # Filter by rounds
    if args.rounds:
        include_rounds = [r.strip() for r in args.rounds.split(',')]
        df = df[df['round'].isin(include_rounds)]
    if args.exclude_rounds:
        exclude_rounds = [r.strip() for r in args.exclude_rounds.split(',')]
        df = df[~df['round'].isin(exclude_rounds)]

    # Separate training data (rows with measured target) from pending
    df_train = df.dropna(subset=[args.target]).copy()
    df_pending = df[df[args.target].isna()].copy()

    # Validate training data
    if len(df_train) == 0:
        print(f"ERROR: No rows with measured '{args.target}' found.")
        print("CSV must contain columns: round,id,LiTFSI,FEC,MDFA,TFB,ssip,cip,agg")
        print("Rows with measured values are used for training.")
        sys.exit(1)

    # Convert composition columns to numeric
    for c in components:
        df_train[c] = pd.to_numeric(df_train[c], errors='coerce')
    df_train[args.target] = pd.to_numeric(df_train[args.target], errors='coerce')
    df_train = df_train.dropna(subset=components + [args.target])

    # Remove invalid rows: skip rows with non-numeric or empty IDs
    # (but allow numeric strings like '1', '2')
    df_train = df_train.reset_index(drop=True)

    print(f"Training data: {len(df_train)} points")
    if len(df_pending) > 0:
        print(f"Pending (unmeasured): {len(df_pending)} points")

    # ---- Prepare training data ----
    X_raw = df_train[components].values
    X_raw = X_raw / X_raw.sum(axis=1, keepdims=True)
    X_alr = alr_transform(X_raw)

    if args.use_log:
        y_raw = df_train[args.target].values.clip(min=1e-10)
        y_train = np.log(y_raw)
    else:
        y_train = df_train[args.target].values

    best_idx = y_train.argmax()
    best_obs_val = df_train.iloc[best_idx][args.target]
    print(f"Best observed {args.target.upper()}: {best_obs_val:.4f}")
    print(f"  at {dict(zip(components, X_raw[best_idx].round(4)))}")

    # ---- Train GP ----
    print(f"\nTraining GP on {args.target.upper()}...")
    gp = build_gp(random_state=args.seed)
    gp.fit(X_alr, y_train)
    score = gp.score(X_alr, y_train)
    print(f"GP R² = {score:.4f}")

    # ---- Optimize acquisition ----
    print(f"\nOptimizing acquisition ({args.acq}, n_restarts=500)...")
    candidates = suggest_candidates(
        gp, y_train.max(), bounds, n=args.n_candidates,
        acq=args.acq, xi=args.xi, kappa=args.kappa
    )
    print(f"Generated {len(candidates)} unique candidates.\n")

    # ---- Print results ----
    print("=" * 70)
    print(f"RECOMMENDED FORMULATIONS — Maximize {args.target.upper()}")
    print("=" * 70)

    rows = []
    for i, c in enumerate(candidates):
        comp = c['composition']
        mu, sigma, ei = c['mu'], c['sigma'], c['ei']
        pred = np.exp(mu) if args.use_log else mu
        ci_lo = np.exp(mu - 1.96 * sigma) if args.use_log else mu - 1.96 * sigma
        ci_hi = np.exp(mu + 1.96 * sigma) if args.use_log else mu + 1.96 * sigma

        print(f"\n  Candidate #{i + 1}: EI = {ei:.4f}")
        print(f"    LiTFSI = {comp[0]:.4f}   FEC = {comp[1]:.4f}   "
              f"MDFA = {comp[2]:.4f}   TFB = {comp[3]:.4f}")
        print(f"    Predicted {args.target.upper()} = {pred:.4f}  "
              f"[95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"    Uncertainty σ = {sigma:.4f}")

        rows.append({
            'candidate': i + 1,
            'round': f'BO-{args.target.upper()}',
            'id': i + 1,
            'LiTFSI': round(comp[0], 6),
            'FEC': round(comp[1], 6),
            'MDFA': round(comp[2], 6),
            'TFB': round(comp[3], 6),
            'ssip': '',
            'cip': '',
            'agg': '',
            f'pred_{args.target}': round(pred, 6),
            f'{args.target}_ci_low': round(ci_lo, 6),
            f'{args.target}_ci_high': round(ci_hi, 6),
            'EI': round(ei, 6),
            'sigma_log': round(sigma, 6),
        })

    # ---- Export candidates ----
    os.makedirs(args.output, exist_ok=True)
    df_out = pd.DataFrame(rows)
    csv_out = os.path.join(args.output, f'candidates_{args.target}.csv')
    df_out.to_csv(csv_out, index=False)
    print(f"\nCandidates exported: {csv_out}")

    # ---- Optional: also train and report cross-target predictions ----
    other_target = 'agg' if args.target == 'cip' else 'cip'
    has_other = other_target in df_train.columns and df_train[other_target].notna().any()
    gp_other = None
    if has_other:
        y_other_raw = df_train[other_target].values.clip(min=1e-10)
        y_other = np.log(y_other_raw) if args.use_log else y_other_raw
        gp_other = build_gp(random_state=args.seed + 1)
        gp_other.fit(X_alr, y_other)
        print(f"\nGP-{other_target.upper()} training R² = {gp_other.score(X_alr, y_other):.4f} (for cross-prediction)")

        print(f"\nCross-predicted {other_target.upper()} for each candidate:")
        for i, c in enumerate(candidates):
            xa = alr_transform(c['composition'].reshape(1, -1))
            mu_o = gp_other.predict(xa).item()
            pred_o = np.exp(mu_o) if args.use_log else mu_o
            print(f"  #{i + 1}: {other_target.upper()} = {pred_o:.4f}")

    # ---- Generate figure ----
    if not args.no_plot:
        png_out = os.path.join(args.output, f'optimization_{args.target}.png')
        make_figure(
            df_train, gp, bounds, args.target, candidates,
            gp_other=gp_other, other_label=other_target,
            output_path=png_out
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
