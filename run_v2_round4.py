"""Run v2 (output-simplex constrained, xi=0.01) on the full measured dataset
to produce Round 4 recommendations: 2 candidates maximizing CIP, 2 maximizing AGG.
"""
import numpy as np, pandas as pd
from bayesian_opt_v2 import JointGP, suggest_candidates

DATA = "/Users/zliang/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/zleung_62c5/msg/file/2026-05/electrolyte_data.csv"

BOUNDS = np.array([[0.03, 0.15],     # LiTFSI
                   [0.00, 0.60],     # FEC
                   [0.00, 0.95],     # MDFA
                   [0.001, 0.50]])   # TFB
COMP   = ['LiTFSI','FEC','MDFA','TFB']
XI     = 0.01
N_CIP  = 2
N_AGG  = 2
SEED   = 42

df = pd.read_csv(DATA)
df = df.dropna(subset=['ssip','cip','agg']).reset_index(drop=True)
for c in COMP + ['ssip','cip','agg']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna().reset_index(drop=True)
print(f"Training points: {len(df)}  (rounds: {sorted(df['round'].unique())})\n")

X = df[COMP].values
X = X / X.sum(axis=1, keepdims=True)
F = df[['ssip','cip','agg']].values
F = F / F.sum(axis=1, keepdims=True)        # numerical safety: enforce simplex

best_cip = F[:, 1].max(); best_agg = F[:, 2].max()
print(f"Incumbent best CIP = {best_cip:.4f}   best AGG = {best_agg:.4f}\n")

print("Training v2 joint GP (output-simplex ALR)...")
rng = np.random.default_rng(SEED)
model = JointGP(seed=SEED).fit(X, F)

# Sanity: in-sample fit
F_hat = model.predict_simplex(X)
print(f"In-sample mean |pred-meas|:  ssip={np.mean(np.abs(F_hat[:,0]-F[:,0])):.4f}  "
      f"cip={np.mean(np.abs(F_hat[:,1]-F[:,1])):.4f}  "
      f"agg={np.mean(np.abs(F_hat[:,2]-F[:,2])):.4f}\n")

print(f"Suggesting {N_CIP} CIP candidates (xi={XI}) ...")
cand_cip = suggest_candidates(model, 'cip', best_cip, BOUNDS, n=N_CIP, xi=XI, rng=rng)
print(f"Suggesting {N_AGG} AGG candidates (xi={XI}) ...")
cand_agg = suggest_candidates(model, 'agg', best_agg, BOUNDS, n=N_AGG, xi=XI, rng=rng)

def fmt(cands, target, prefix):
    rows = []
    print(f"\n=== Round-4 candidates — maximize {target.upper()} ===")
    for i, c in enumerate(cands, 1):
        comp = c['composition']
        # also predict the *other* target's posterior at the same point
        mu_s, sd_s, _ = model.predict_target(comp.reshape(1,-1), 'ssip', n_mc=4000, rng=rng)
        mu_c, sd_c, _ = model.predict_target(comp.reshape(1,-1), 'cip',  n_mc=4000, rng=rng)
        mu_a, sd_a, _ = model.predict_target(comp.reshape(1,-1), 'agg',  n_mc=4000, rng=rng)
        print(f"  #{prefix}{i}  LiTFSI={comp[0]:.4f}  FEC={comp[1]:.4f}  "
              f"MDFA={comp[2]:.4f}  TFB={comp[3]:.4f}")
        print(f"        EI(target={target}) = {c['ei']:.4f}")
        print(f"        predicted  ssip={mu_s[0]:.3f}±{sd_s[0]:.3f}   "
              f"cip={mu_c[0]:.3f}±{sd_c[0]:.3f}   agg={mu_a[0]:.3f}±{sd_a[0]:.3f}")
        rows.append({
            'round':'Round 4','id': f'{prefix}{i}','target':target,
            'LiTFSI':round(comp[0],4),'FEC':round(comp[1],4),
            'MDFA':round(comp[2],4),'TFB':round(comp[3],4),
            'pred_ssip':round(float(mu_s[0]),4),'sd_ssip':round(float(sd_s[0]),4),
            'pred_cip' :round(float(mu_c[0]),4),'sd_cip' :round(float(sd_c[0]),4),
            'pred_agg' :round(float(mu_a[0]),4),'sd_agg' :round(float(sd_a[0]),4),
            'EI':round(float(c['ei']),6),
        })
    return rows

rows = fmt(cand_cip, 'cip', 'B4C') + fmt(cand_agg, 'agg', 'B4A')
out = pd.DataFrame(rows)
out.to_csv('round4_candidates_v2.csv', index=False)
print(f"\nSaved: round4_candidates_v2.csv")
