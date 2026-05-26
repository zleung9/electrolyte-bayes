#!/usr/bin/env python3
"""
Bayesian Optimization for Electrolyte Formulation Screening — V2
================================================================
Optimizes electrolyte composition to maximize CIP or AGG fraction, using
Gaussian-Process regression with simplex constraints on BOTH the input
(composition) and the output (speciation) simplices.

Differences vs V1 (bayesian_opt.py):
  - Output-simplex constraint: ssip + cip + agg = 1 is enforced by modeling
    the 2-D ALR of the output simplex
        y_c = log(cip/ssip),  y_a = log(agg/ssip)
    with two GPs.  Predictions (ssip, cip, agg) always sum to 1, and the
    CIP/AGG/SSIP trade-off is shared across both GPs.
  - EI computed via Monte-Carlo (the back-transform is non-linear, so no
    closed-form EI in the original target).
  - Default exploration parameter xi: 0.01 (V1 default was 0.03).

Usage (same as V1):
    python bayesian_opt_v2.py --data electrolyte_data.csv --target cip
    python bayesian_opt_v2.py --data electrolyte_data.csv --target agg --n-candidates 8
    python bayesian_opt_v2.py --data electrolyte_data.csv --target cip --xi 0.01

Input CSV format:
    round,id,LiTFSI,FEC,MDFA,TFB,ssip,cip,agg
    Round 0,1,0.0989,0.0,0.9011,0.0,0.2939,0.5425,0.1636
    ...

    Rows with ssip, cip, agg ALL present are used for GP training.
    Rows with any of those missing are treated as pending (excluded from training).
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-10


# ============================================================
# Input (composition) simplex transform — same as V1
# ============================================================

def alr_in(X):
    """Additive Log-Ratio: (n,4) simplex -> (n,3) unconstrained, ref = TFB."""
    return np.log((X[:, :3] + EPS) / (X[:, 3:] + EPS))


def alr_in_inv(Y):
    """Inverse ALR: (n,3) unconstrained -> (n,4) simplex."""
    ey = np.exp(Y)
    denom = 1.0 + ey.sum(axis=1, keepdims=True)
    return np.hstack([ey / denom, 1.0 / denom])


# ============================================================
# Output (speciation) simplex transform — NEW in V2
# ============================================================

def alr_out(F):
    """(n,3) [ssip,cip,agg] simplex -> (n,2) log-ratios with ssip as reference."""
    F = F.clip(min=EPS)
    return np.column_stack([np.log(F[:, 1] / F[:, 0]),
                            np.log(F[:, 2] / F[:, 0])])


def alr_out_inv(Y):
    """(n,2) -> (n,3) simplex with ssip as reference."""
    e = np.exp(Y)
    denom = 1.0 + e[:, 0] + e[:, 1]
    return np.column_stack([1.0 / denom, e[:, 0] / denom, e[:, 1] / denom])


# ============================================================
# Joint GP on the output-simplex ALR
# ============================================================

def build_gp(seed=42):
    """Create a Gaussian Process regressor with Matérn 5/2 kernel."""
    kernel = (ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0, 1.0], nu=2.5)
              + WhiteKernel(noise_level=0.05))
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=20,
        normalize_y=True, alpha=1e-4, random_state=seed
    )


class JointGP:
    """Two GPs on the output-simplex ALR.  The constraint ssip+cip+agg=1 is
    enforced by construction, and the CIP/AGG/SSIP trade-off is shared via
    the common ssip-reference denominator."""

    def __init__(self, seed=42):
        self.gp_c = build_gp(seed)
        self.gp_a = build_gp(seed + 1)
        self._X_alr = None
        self._Y = None

    def fit(self, X_in, F_out):
        Xa = alr_in(X_in)
        Y = alr_out(F_out)                  # (n,2) — log-ratios
        self.gp_c.fit(Xa, Y[:, 0])
        self.gp_a.fit(Xa, Y[:, 1])
        self._X_alr = Xa
        self._Y = Y
        return self

    def _predict_ratios(self, X_in, return_std=False):
        Xa = alr_in(X_in)
        if return_std:
            mu_c, sd_c = self.gp_c.predict(Xa, return_std=True)
            mu_a, sd_a = self.gp_a.predict(Xa, return_std=True)
            return mu_c, sd_c, mu_a, sd_a
        return self.gp_c.predict(Xa), self.gp_a.predict(Xa)

    def predict_simplex(self, X_in):
        """Posterior-mean (ssip, cip, agg).  Sums to 1 by construction."""
        mu_c, mu_a = self._predict_ratios(X_in)
        return alr_out_inv(np.column_stack([mu_c, mu_a]))

    def predict_target(self, X_in, target, n_mc=400, rng=None):
        """MC estimate of (mean, std, samples) for the requested target."""
        if rng is None:
            rng = np.random.default_rng(0)
        mu_c, sd_c, mu_a, sd_a = self._predict_ratios(X_in, return_std=True)
        n = X_in.shape[0]
        Yc = mu_c[:, None] + sd_c[:, None] * rng.standard_normal((n, n_mc))
        Ya = mu_a[:, None] + sd_a[:, None] * rng.standard_normal((n, n_mc))
        e_c, e_a = np.exp(Yc), np.exp(Ya)
        denom = 1.0 + e_c + e_a
        if   target == 'ssip': samp = 1.0 / denom
        elif target == 'cip':  samp = e_c / denom
        elif target == 'agg':  samp = e_a / denom
        else:                  raise ValueError(f"unknown target: {target}")
        return samp.mean(axis=1), samp.std(axis=1), samp

    def score(self, X_in, F_out, target):
        """R² of the back-transformed posterior mean against measured target."""
        F_hat = self.predict_simplex(X_in)
        col = {'ssip': 0, 'cip': 1, 'agg': 2}[target]
        y_true = F_out[:, col]
        y_pred = F_hat[:, col]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2) + EPS
        return 1.0 - ss_res / ss_tot


# ============================================================
# Acquisition functions (Monte-Carlo)
# ============================================================

def expected_improvement_mc(samples, y_best, xi=0.01):
    """EI = E[max(target - y_best - xi, 0)] from MC samples (n_points, n_mc)."""
    return np.maximum(samples - y_best - xi, 0.0).mean(axis=1)


def upper_confidence_bound_mc(samples, kappa=2.0):
    """UCB on the empirical posterior of the target."""
    return samples.mean(axis=1) + kappa * samples.std(axis=1)


# ============================================================
# Constrained acquisition optimization
# ============================================================

def optimize_acquisition(model, target, y_best, bounds, acq='ei',
                         xi=0.01, kappa=2.0, n_restarts=200, n_mc=400, rng=None):
    """Maximize the acquisition over the input simplex with TFB = 1 - sum."""
    if rng is None:
        rng = np.random.default_rng(0)
    b3 = [(bounds[0, 0], bounds[0, 1]),
          (bounds[1, 0], bounds[1, 1]),
          (bounds[2, 0], bounds[2, 1])]

    best_x, best_neg_val = None, np.inf

    for _ in range(n_restarts):
        for __ in range(500):
            x0 = np.array([rng.uniform(*b) for b in b3])
            tfb = 1.0 - x0.sum()
            if bounds[3, 0] <= tfb <= bounds[3, 1]:
                break
        else:
            continue

        def neg_acq(x3):
            tfb = 1.0 - x3[0] - x3[1] - x3[2]
            if tfb < bounds[3, 0] or tfb > bounds[3, 1]:
                return 1e10
            comp = np.array([[x3[0], x3[1], x3[2], tfb]])
            comp = comp / comp.sum()
            _, _, samp = model.predict_target(comp, target, n_mc=n_mc, rng=rng)
            if acq == 'ei':
                v = expected_improvement_mc(samp, y_best, xi=xi)[0]
            else:
                v = upper_confidence_bound_mc(samp, kappa=kappa)[0]
            return -float(v)

        res = minimize(neg_acq, x0, bounds=b3, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.fun < best_neg_val and res.fun < 1e9:
            best_neg_val = res.fun
            best_x = res.x.copy()

    if best_x is None:
        return None, None
    tfb = 1.0 - best_x.sum()
    comp = np.array([best_x[0], best_x[1], best_x[2], tfb])
    return comp / comp.sum(), -best_neg_val


def suggest_candidates(model, target, y_best, bounds, n=5, acq='ei',
                       xi=0.01, kappa=2.0, dedup_threshold=0.005,
                       n_restarts=200, n_mc=400, rng=None):
    """Suggest n diverse candidates by repeated acquisition optimization."""
    if rng is None:
        rng = np.random.default_rng(0)
    candidates = []
    cur_xi = xi
    for i in range(n):
        comp, val = optimize_acquisition(
            model, target, y_best, bounds, acq=acq,
            xi=cur_xi, kappa=kappa, n_restarts=n_restarts, n_mc=n_mc, rng=rng
        )
        if comp is None:
            cur_xi *= 2.0
            continue
        if any(np.abs(comp - p['composition']).max() < dedup_threshold
               for p in candidates):
            cur_xi *= 3.0
            comp, val = optimize_acquisition(
                model, target, y_best, bounds, acq=acq,
                xi=cur_xi, kappa=kappa, n_restarts=n_restarts, n_mc=n_mc, rng=rng
            )
            if comp is None:
                continue
            if any(np.abs(comp - p['composition']).max() < dedup_threshold
                   for p in candidates):
                continue
        mu, sd, _ = model.predict_target(comp.reshape(1, -1), target,
                                         n_mc=2000, rng=rng)
        candidates.append({
            'composition': comp,
            'acq_val': float(val),
            'mean': float(mu[0]),
            'std': float(sd[0]),
        })
        cur_xi = xi * (1.0 + 0.5 * (i + 1))
    return candidates


# ============================================================
# Visualization
# ============================================================

def make_figure(df_train, model, bounds, target, candidates, output_path=None,
                rng=None):
    """4-panel diagnostic figure (same layout as V1)."""
    if rng is None:
        rng = np.random.default_rng(0)
    components = ['LiTFSI', 'FEC', 'MDFA', 'TFB']
    X_train = df_train[components].values
    X_train = X_train / X_train.sum(axis=1, keepdims=True)
    F_train = df_train[['ssip', 'cip', 'agg']].values
    F_train = F_train / F_train.sum(axis=1, keepdims=True)
    col = {'ssip': 0, 'cip': 1, 'agg': 2}[target]
    y_train = F_train[:, col]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Panel 1: response contour on MDFA–TFB plane ----
    ax = axes[0, 0]
    li_fix, fec_fix = 0.08, 0.0
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
            surf[i, j] = model.predict_simplex(comp)[0, col]
    cmap = 'Blues' if target == 'cip' else ('Reds' if target == 'agg' else 'Greens')
    im = ax.contourf(MM, TT, surf, levels=20, cmap=cmap)
    ax.scatter(X_train[:, 2], X_train[:, 3], c=y_train,
               s=60, edgecolors='black', linewidth=0.8, cmap=cmap, zorder=5)
    for i, c in enumerate(candidates):
        ax.scatter(c['composition'][2], c['composition'][3],
                   c='gold', s=100, marker='D', edgecolors='black', zorder=10)
        ax.annotate(str(i + 1), (c['composition'][2], c['composition'][3]),
                    fontsize=7, ha='center', va='bottom',
                    color='darkred', fontweight='bold')
    best_idx = y_train.argmax()
    ax.scatter(X_train[best_idx, 2], X_train[best_idx, 3],
               c='red', s=120, marker='*', edgecolors='black', zorder=10)
    plt.colorbar(im, ax=ax, label=f'Predicted {target.upper()}')
    ax.set_xlabel('MDFA'); ax.set_ylabel('TFB')
    ax.set_title(f'{target.upper()} response (LiTFSI={li_fix}, FEC=0)\n'
                 f'* = best observed,  ◇ = candidates')

    # ---- Panel 2: GP fit (back-transformed) ----
    ax = axes[0, 1]
    F_hat = model.predict_simplex(X_train)
    y_pred = F_hat[:, col]
    ax.scatter(y_train, y_pred, c='steelblue', s=50, edgecolors='black')
    mx = max(y_train.max(), y_pred.max()) * 1.1
    ax.plot([0, mx], [0, mx], 'k--', alpha=0.5)
    ax.set_xlabel(f'Observed {target.upper()}')
    ax.set_ylabel(f'Predicted {target.upper()}')
    score = model.score(X_train, F_train, target)
    ax.set_title(f'GP Fit: {target.upper()} (R² = {score:.3f})')

    # ---- Panel 3: training data CIP vs AGG ----
    ax = axes[1, 0]
    cip_vals = F_train[:, 1]; agg_vals = F_train[:, 2]
    ax.scatter(cip_vals, agg_vals, c='steelblue', s=60,
               edgecolors='black', zorder=5)
    for i in range(len(df_train)):
        ax.annotate(str(df_train.iloc[i]['id']), (cip_vals[i], agg_vals[i]),
                    fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('CIP'); ax.set_ylabel('AGG')
    ax.set_title('Training Data: CIP vs AGG')

    # ---- Panel 4: TFB effect (target ± 95% CI) ----
    ax = axes[1, 1]
    li, fec = 0.08, 0.0
    tfb_range = np.linspace(bounds[3, 0] + 0.001, bounds[3, 1], 80)
    means, los, his = [], [], []
    grid_pts = []
    for t in tfb_range:
        m = 1.0 - li - fec - t
        if m < 0 or m > bounds[2, 1]:
            continue
        grid_pts.append([li, fec, m, t])
    grid_pts = np.array(grid_pts)
    mu, sd, _ = model.predict_target(grid_pts, target, n_mc=2000, rng=rng)
    means = mu; los = np.clip(mu - 1.96 * sd, 0, 1); his = np.clip(mu + 1.96 * sd, 0, 1)
    tr = grid_pts[:, 3]
    color = 'b' if target == 'cip' else ('r' if target == 'agg' else 'g')
    ax.plot(tr, means, color + '-', lw=2, label=target.upper())
    ax.fill_between(tr, los, his, alpha=0.15, color=color)
    ax.set_xlabel('TFB mole fraction')
    ax.set_ylabel(target.upper())
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
        description='Bayesian Optimization for Electrolyte Screening (V2)'
    )
    parser.add_argument('--data', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--target', required=True, choices=['cip', 'agg'],
                        help='Target to maximize: cip or agg')
    parser.add_argument('--rounds', default=None,
                        help='Comma-separated round labels to include')
    parser.add_argument('--exclude-rounds', default=None,
                        help='Comma-separated round labels to exclude')
    parser.add_argument('--n-candidates', type=int, default=5,
                        help='Number of candidates to suggest (default: 5)')
    parser.add_argument('--acq', default='ei', choices=['ei', 'ucb'],
                        help='Acquisition function (default: ei)')
    parser.add_argument('--xi', type=float, default=0.01,
                        help='Exploration parameter for EI (default: 0.01)')
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
    parser.add_argument('--n-mc', type=int, default=400,
                        help='MC samples for acquisition (default: 400)')
    parser.add_argument('--n-restarts', type=int, default=200,
                        help='Acquisition optimization restarts (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

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
    species = ['ssip', 'cip', 'agg']

    if args.rounds:
        df = df[df['round'].isin([r.strip() for r in args.rounds.split(',')])]
    if args.exclude_rounds:
        df = df[~df['round'].isin([r.strip() for r in
                                   args.exclude_rounds.split(',')])]

    # V2 requires ALL three speciation values to enforce ssip+cip+agg=1
    df_train = df.dropna(subset=species).copy()
    df_pending = df[df[species].isna().any(axis=1)].copy()

    if len(df_train) == 0:
        print(f"ERROR: no rows with all of ssip/cip/agg present.")
        print("CSV must contain: round,id,LiTFSI,FEC,MDFA,TFB,ssip,cip,agg")
        print("V2 requires all three speciation values to enforce the simplex.")
        sys.exit(1)

    for c in components + species:
        df_train[c] = pd.to_numeric(df_train[c], errors='coerce')
    df_train = df_train.dropna(subset=components + species).reset_index(drop=True)

    print(f"Training data: {len(df_train)} points")
    if len(df_pending) > 0:
        print(f"Pending (missing ssip/cip/agg): {len(df_pending)} points")

    X = df_train[components].values
    X = X / X.sum(axis=1, keepdims=True)
    F = df_train[species].values
    F = F / F.sum(axis=1, keepdims=True)            # normalize to simplex

    best_idx = F[:, {'cip': 1, 'agg': 2}[args.target]].argmax()
    best_obs = F[best_idx, {'cip': 1, 'agg': 2}[args.target]]
    print(f"Best observed {args.target.upper()}: {best_obs:.4f}")
    print(f"  at {dict(zip(components, X[best_idx].round(4)))}")

    # ---- Train joint GP (output-simplex ALR) ----
    print(f"\nTraining v2 joint GP (output-simplex ALR)...")
    model = JointGP(seed=args.seed).fit(X, F)
    r2 = model.score(X, F, args.target)
    print(f"GP R² ({args.target.upper()}) = {r2:.4f}")
    F_hat = model.predict_simplex(X)
    print(f"In-sample mean |pred-meas|:  "
          f"ssip={np.mean(np.abs(F_hat[:, 0] - F[:, 0])):.4f}  "
          f"cip={np.mean(np.abs(F_hat[:, 1] - F[:, 1])):.4f}  "
          f"agg={np.mean(np.abs(F_hat[:, 2] - F[:, 2])):.4f}")

    # ---- Optimize acquisition ----
    print(f"\nOptimizing acquisition ({args.acq}, n_restarts={args.n_restarts})...")
    candidates = suggest_candidates(
        model, args.target, best_obs, bounds,
        n=args.n_candidates, acq=args.acq,
        xi=args.xi, kappa=args.kappa,
        n_restarts=args.n_restarts, n_mc=args.n_mc, rng=rng
    )
    print(f"Generated {len(candidates)} unique candidates.\n")

    # ---- Print + export ----
    print("=" * 70)
    print(f"RECOMMENDED FORMULATIONS — Maximize {args.target.upper()}")
    print("=" * 70)
    rows = []
    other_target = 'agg' if args.target == 'cip' else 'cip'
    for i, c in enumerate(candidates):
        comp = c['composition']
        mu, sd = c['mean'], c['std']
        ci_lo = max(0.0, mu - 1.96 * sd)
        ci_hi = min(1.0, mu + 1.96 * sd)
        # also predict the other target and ssip at this point
        mu_o, sd_o, _ = model.predict_target(comp.reshape(1, -1), other_target,
                                             n_mc=2000, rng=rng)
        mu_s, sd_s, _ = model.predict_target(comp.reshape(1, -1), 'ssip',
                                             n_mc=2000, rng=rng)
        print(f"\n  Candidate #{i + 1}: acq = {c['acq_val']:.4f}")
        print(f"    LiTFSI = {comp[0]:.4f}   FEC = {comp[1]:.4f}   "
              f"MDFA = {comp[2]:.4f}   TFB = {comp[3]:.4f}")
        print(f"    Predicted {args.target.upper()} = {mu:.4f}  "
              f"[95% CI: {ci_lo:.4f}, {ci_hi:.4f}]   σ = {sd:.4f}")
        print(f"    Cross-predicted SSIP = {mu_s[0]:.4f}±{sd_s[0]:.4f}   "
              f"{other_target.upper()} = {mu_o[0]:.4f}±{sd_o[0]:.4f}")

        rows.append({
            'candidate': i + 1,
            'round': f'BO-{args.target.upper()}',
            'id': i + 1,
            'LiTFSI': round(comp[0], 6),
            'FEC':    round(comp[1], 6),
            'MDFA':   round(comp[2], 6),
            'TFB':    round(comp[3], 6),
            'ssip': '', 'cip': '', 'agg': '',
            f'pred_{args.target}':       round(mu, 6),
            f'{args.target}_ci_low':     round(ci_lo, 6),
            f'{args.target}_ci_high':    round(ci_hi, 6),
            f'pred_{other_target}':      round(float(mu_o[0]), 6),
            f'sd_{other_target}':        round(float(sd_o[0]), 6),
            'pred_ssip':                 round(float(mu_s[0]), 6),
            'sd_ssip':                   round(float(sd_s[0]), 6),
            'acq': round(c['acq_val'], 6),
            f'sigma_{args.target}':      round(sd, 6),
        })

    os.makedirs(args.output, exist_ok=True)
    df_out = pd.DataFrame(rows)
    csv_out = os.path.join(args.output, f'candidates_{args.target}_v2.csv')
    df_out.to_csv(csv_out, index=False)
    print(f"\nCandidates exported: {csv_out}")

    # ---- Figure ----
    if not args.no_plot:
        png_out = os.path.join(args.output, f'optimization_{args.target}_v2.png')
        make_figure(df_train, model, bounds, args.target, candidates,
                    output_path=png_out, rng=rng)

    print("\nDone.")


if __name__ == '__main__':
    main()
