"""Toy validation: run v1 (independent log-GPs) vs v2 (output-simplex constrained)
on a synthetic ground truth and compare convergence + prediction accuracy.

Ground truth:
  Given composition r = (LiTFSI, FEC, MDFA, TFB) on the input simplex, define
      u_ssip = 0
      u_cip  =  1.5*MDFA - 1.0*TFB - 2.0*FEC + 0.3
      u_agg  = -0.5*MDFA + 2.0*TFB - 2.5*FEC + 0.5*LiTFSI - 0.2
  and (ssip, cip, agg) = softmax(u_ssip, u_cip, u_agg) (+ small noise on logits).
This is a smooth 4D->simplex mapping that captures the qualitative chemistry:
MDFA promotes CIP, TFB promotes AGG, FEC suppresses both (boosts SSIP).
"""
import numpy as np, pandas as pd
from bayesian_opt_v2 import JointGP, suggest_candidates, alr_in, alr_out_inv

# v1-style independent log-GP target prediction (matches bayesian_opt.py)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern
def build_gp(seed):
    k = (ConstantKernel(1.0)*Matern(length_scale=[1,1,1], nu=2.5)
         + WhiteKernel(noise_level=0.05))
    return GaussianProcessRegressor(kernel=k, n_restarts_optimizer=20,
                                    normalize_y=True, alpha=1e-4, random_state=seed)


RNG_SEED = 7
N_INIT   = 10
N_ROUNDS = 5
N_PER_ROUND_PER_TARGET = 2     # 2 cip + 2 agg per round, like the real workflow
BOUNDS = np.array([[0.03, 0.15],
                   [0.00, 0.60],
                   [0.00, 0.95],
                   [0.001, 0.50]])


# ---------- ground truth -----------------------------------------------------
def ground_truth(comp, noise=0.02, rng=None):
    """comp: (n,4) array on simplex.  Returns (n,3) [ssip,cip,agg]."""
    rng = rng or np.random.default_rng(0)
    li, fec, mdfa, tfb = comp[:,0], comp[:,1], comp[:,2], comp[:,3]
    u_s = np.zeros_like(li)
    u_c =  1.5*mdfa - 1.0*tfb - 2.0*fec + 0.3
    u_a = -0.5*mdfa + 2.0*tfb - 2.5*fec + 0.5*li - 0.2
    U = np.column_stack([u_s, u_c, u_a]) + noise*rng.standard_normal((len(li),3))
    E = np.exp(U - U.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


# ---------- random feasible initial set --------------------------------------
def sample_feasible(n, rng):
    out = []
    while len(out) < n:
        li  = rng.uniform(*BOUNDS[0])
        fec = rng.uniform(*BOUNDS[1])
        mdfa= rng.uniform(*BOUNDS[2])
        tfb = 1.0 - li - fec - mdfa
        if BOUNDS[3,0] <= tfb <= BOUNDS[3,1]:
            out.append([li, fec, mdfa, tfb])
    return np.array(out)


# ---------- v1 wrapper: independent log-GPs ----------------------------------
class IndependentLogGP:
    def __init__(self, seed=42):
        self.gp = {'cip': build_gp(seed), 'agg': build_gp(seed+1)}
    def fit(self, X_in, F):
        Xa = alr_in(X_in)
        self.gp['cip'].fit(Xa, np.log(F[:,1].clip(min=1e-10)))
        self.gp['agg'].fit(Xa, np.log(F[:,2].clip(min=1e-10)))
        return self
    def predict_target(self, X_in, target, n_mc=400, rng=None):
        rng = rng or np.random.default_rng(0)
        Xa = alr_in(X_in)
        mu, sd = self.gp[target].predict(Xa, return_std=True)
        n = X_in.shape[0]
        samples = np.exp(mu[:,None] + sd[:,None]*rng.standard_normal((n, n_mc)))
        return samples.mean(axis=1), samples.std(axis=1), samples


# v1-style suggest_candidates
from scipy.optimize import minimize
def suggest_v1(model, target, y_best, xi=0.03, n=4, rng=None):
    rng = rng or np.random.default_rng(0)
    b3 = [(BOUNDS[0,0],BOUNDS[0,1]),(BOUNDS[1,0],BOUNDS[1,1]),(BOUNDS[2,0],BOUNDS[2,1])]
    candidates = []
    cur_xi = xi
    for i in range(n):
        best_x, best_val = None, np.inf
        for _ in range(120):
            for __ in range(500):
                x0 = np.array([rng.uniform(*b) for b in b3])
                tfb = 1.0 - x0.sum()
                if BOUNDS[3,0] <= tfb <= BOUNDS[3,1]: break
            else: continue
            def neg(x3):
                tfb = 1.0 - x3[0] - x3[1] - x3[2]
                if tfb < BOUNDS[3,0] or tfb > BOUNDS[3,1]: return 1e10
                comp = np.array([[x3[0],x3[1],x3[2],tfb]])
                comp /= comp.sum()
                _,_,samp = model.predict_target(comp, target, n_mc=400, rng=rng)
                return -float(np.maximum(samp[0] - y_best - cur_xi, 0).mean())
            r = minimize(neg, x0, bounds=b3, method='L-BFGS-B',
                         options={'maxiter':200,'ftol':1e-12})
            if r.fun < best_val and r.fun < 1e9:
                best_val, best_x = r.fun, r.x.copy()
        if best_x is None: continue
        tfb = 1.0 - best_x.sum()
        comp = np.array([best_x[0],best_x[1],best_x[2],tfb]); comp /= comp.sum()
        if any(np.abs(comp - p['composition']).max() < 0.005 for p in candidates):
            cur_xi *= 3.0; continue
        candidates.append({'composition': comp, 'ei': -best_val})
        cur_xi = xi*(1.0 + 0.5*(i+1))
    return candidates


# ---------- run one BO trajectory --------------------------------------------
def run_trajectory(model_class, label, xi, seed):
    rng = np.random.default_rng(seed)
    X = sample_feasible(N_INIT, rng)
    F = ground_truth(X, rng=rng)
    history = []
    for rd in range(N_ROUNDS + 1):
        # incumbents
        best_cip = F[:,1].max(); best_agg = F[:,2].max()
        # prediction MAE on a fresh held-out set
        Xh = sample_feasible(200, np.random.default_rng(seed + 1000 + rd))
        Fh = ground_truth(Xh, noise=0.0, rng=np.random.default_rng(seed+2000+rd))
        mdl = model_class(seed=seed).fit(X, F)
        mae_cip = np.mean(np.abs(mdl.predict_target(Xh, 'cip', n_mc=200, rng=rng)[0] - Fh[:,1]))
        mae_agg = np.mean(np.abs(mdl.predict_target(Xh, 'agg', n_mc=200, rng=rng)[0] - Fh[:,2]))
        history.append({'round': rd, 'n': len(X),
                        'best_cip': best_cip, 'best_agg': best_agg,
                        'mae_cip': mae_cip, 'mae_agg': mae_agg})
        if rd == N_ROUNDS: break

        # suggest 2 for cip and 2 for agg
        if label == 'v2':
            cand_c = suggest_candidates(mdl, 'cip', best_cip, BOUNDS,
                                        n=N_PER_ROUND_PER_TARGET, xi=xi, rng=rng)
            cand_a = suggest_candidates(mdl, 'agg', best_agg, BOUNDS,
                                        n=N_PER_ROUND_PER_TARGET, xi=xi, rng=rng)
        else:
            cand_c = suggest_v1(mdl, 'cip', best_cip, xi=xi,
                                n=N_PER_ROUND_PER_TARGET, rng=rng)
            cand_a = suggest_v1(mdl, 'agg', best_agg, xi=xi,
                                n=N_PER_ROUND_PER_TARGET, rng=rng)

        # "measure" each suggestion
        new_comps = np.array([c['composition'] for c in (cand_c + cand_a)])
        new_F     = ground_truth(new_comps, rng=rng)
        X = np.vstack([X, new_comps]); F = np.vstack([F, new_F])
    return pd.DataFrame(history)


# ---------- driver -----------------------------------------------------------
def main():
    # establish the toy's true maxima by random search
    rng = np.random.default_rng(99)
    Xs = sample_feasible(20000, rng)
    Fs = ground_truth(Xs, noise=0.0, rng=rng)
    print(f"Toy ground-truth (random-search estimate of optima):")
    print(f"  max CIP ≈ {Fs[:,1].max():.4f}    max AGG ≈ {Fs[:,2].max():.4f}\n")

    print("Running v1 (independent log-GPs, xi=0.03) — 3 seeds...")
    v1 = pd.concat([run_trajectory(IndependentLogGP, 'v1', xi=0.03, seed=s)
                    .assign(seed=s) for s in (1,2,3)], ignore_index=True)
    print("Running v2 (constrained output-simplex GPs, xi=0.01) — 3 seeds...")
    v2 = pd.concat([run_trajectory(JointGP,         'v2', xi=0.01, seed=s)
                    .assign(seed=s) for s in (1,2,3)], ignore_index=True)

    def summarize(df, label):
        g = df.groupby('round')
        m = g[['best_cip','best_agg','mae_cip','mae_agg']].mean()
        print(f"\n--- {label} (mean over 3 seeds) ---")
        print(m.round(4).to_string())
    summarize(v1, 'v1: independent log-GPs, xi=0.03')
    summarize(v2, 'v2: output-simplex constrained, xi=0.01')

    # head-to-head at final round
    f1 = v1[v1['round']==N_ROUNDS]; f2 = v2[v2['round']==N_ROUNDS]
    print("\n===== Head-to-head at final round =====")
    print(f"  best CIP  v1={f1['best_cip'].mean():.4f} ± {f1['best_cip'].std():.4f}"
          f"   v2={f2['best_cip'].mean():.4f} ± {f2['best_cip'].std():.4f}")
    print(f"  best AGG  v1={f1['best_agg'].mean():.4f} ± {f1['best_agg'].std():.4f}"
          f"   v2={f2['best_agg'].mean():.4f} ± {f2['best_agg'].std():.4f}")
    print(f"  MAE  CIP  v1={f1['mae_cip'].mean():.4f}"
          f"   v2={f2['mae_cip'].mean():.4f}")
    print(f"  MAE  AGG  v1={f1['mae_agg'].mean():.4f}"
          f"   v2={f2['mae_agg'].mean():.4f}")

    v1.to_csv('toy_results_v1.csv', index=False)
    v2.to_csv('toy_results_v2.csv', index=False)
    print("\nSaved: toy_results_v1.csv, toy_results_v2.csv")


if __name__ == '__main__':
    main()
