"""Compare BO predictions vs measured experimental results.

File A: electrolyte_data.csv               — measured values (ground truth)
File B: electrolyte_data_计算分子数(2).csv — same compositions; for B2x/B3x rows
                                              the target column holds the GP prediction
                                              that was used to pick the candidate.
"""
import os
import numpy as np
import pandas as pd

A = "/Users/zliang/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/zleung_62c5/msg/file/2026-05/electrolyte_data.csv"
B = "/Users/zliang/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/zleung_62c5/msg/file/2026-05/electrolyte_data_计算分子数(2).csv"

meas = pd.read_csv(A)
pred = pd.read_csv(B)
# Drop unnamed columns in pred (computed molecule counts etc.)
pred = pred.loc[:, ~pred.columns.str.contains("^Unnamed")]
pred = pred.dropna(subset=["id"]).copy()

# Per-round best target trajectory (measured)
print("===== Per-round best measured CIP / AGG (over all entries in that round) =====")
for r, g in meas.groupby("round", sort=False):
    print(f"  {r:8s}  n={len(g):2d}   best CIP = {g['cip'].max():.4f}   "
          f"best AGG = {g['agg'].max():.4f}")

# Round-0 baseline best
r0 = meas[meas["round"] == "Round 0"]
best_cip_r0 = r0["cip"].max()
best_agg_r0 = r0["agg"].max()
print(f"\nRound-0 baseline:  best CIP = {best_cip_r0:.4f},  best AGG = {best_agg_r0:.4f}")

# BO suggestions only (ids starting with B)
def is_bo(idv):
    s = str(idv)
    return s.startswith("B") and len(s) >= 2 and s[1].isdigit()

bo_meas = meas[meas["id"].apply(is_bo)].copy()
bo_pred = pred[pred["id"].apply(is_bo)].copy()

# Identify which target each BO row was optimized for: whichever of cip/agg is
# non-empty in the prediction file is the target.
bo_pred["target"] = np.where(bo_pred["cip"].notna() & bo_pred["cip"].astype(str).str.strip().ne(""),
                             "cip",
                             "agg")
# A row could in theory have both filled — disambiguate by checking which is numeric.
def pick_target(row):
    cip_ok = pd.notna(row["cip"]) and str(row["cip"]).strip() != ""
    agg_ok = pd.notna(row["agg"]) and str(row["agg"]).strip() != ""
    if cip_ok and not agg_ok: return "cip"
    if agg_ok and not cip_ok: return "agg"
    if cip_ok and agg_ok:    return "both"
    return None
bo_pred["target"] = bo_pred.apply(pick_target, axis=1)

# Build comparison frame
rows = []
for _, p in bo_pred.iterrows():
    m = bo_meas[bo_meas["id"] == p["id"]]
    if m.empty: continue
    m = m.iloc[0]
    t = p["target"]
    if t in ("cip", "agg"):
        pred_val = float(p[t])
        meas_val = float(m[t])
    else:
        continue
    rows.append({
        "round": m["round"],
        "id": p["id"],
        "target": t,
        "predicted": pred_val,
        "measured": meas_val,
        "error":  meas_val - pred_val,         # +ve = measurement exceeded prediction
        "abs_err": abs(meas_val - pred_val),
        "rel_err": abs(meas_val - pred_val)/max(meas_val, 1e-9),
        "other_target_meas_cip": m["cip"],
        "other_target_meas_agg": m["agg"],
    })

cmp = pd.DataFrame(rows)
print("\n===== Predictions vs measurements (BO suggestions only) =====")
print(cmp[["round","id","target","predicted","measured","error","rel_err"]]
      .to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

# Aggregate accuracy
print("\n===== Prediction accuracy by target =====")
for t in ("cip","agg"):
    sub = cmp[cmp["target"] == t]
    if len(sub)==0: continue
    rmse = np.sqrt(((sub["predicted"]-sub["measured"])**2).mean())
    mae  = sub["abs_err"].mean()
    bias = sub["error"].mean()
    # Pearson r if at least 2 points
    if len(sub) >= 2:
        r = np.corrcoef(sub["predicted"], sub["measured"])[0,1]
    else:
        r = float("nan")
    print(f"  {t.upper()}  n={len(sub)}   MAE={mae:.4f}   RMSE={rmse:.4f}   "
          f"bias(meas-pred)={bias:+.4f}   Pearson r={r:.3f}")

# Direction-of-progress: per BO round, what was the best newly measured target?
print("\n===== Trajectory of the best newly-measured target =====")
all_rounds = ["Round 0","Round 1","Round 2","Round 3"]
hi_cip = -np.inf; hi_agg = -np.inf
for r in all_rounds:
    g = meas[meas["round"] == r]
    bc = g["cip"].max(); ba = g["agg"].max()
    new_cip = bc - hi_cip if hi_cip != -np.inf else 0
    new_agg = ba - hi_agg if hi_agg != -np.inf else 0
    print(f"  {r:8s}  best CIP in round={bc:.4f} (Δ vs prev best={new_cip:+.4f})   "
          f"best AGG in round={ba:.4f} (Δ vs prev best={new_agg:+.4f})")
    hi_cip = max(hi_cip, bc); hi_agg = max(hi_agg, ba)

# Did each BO recommendation BEAT the prior best for its declared target?
print("\n===== Did each recommendation outperform the prior incumbent? =====")
# Build incumbent (best seen up to but not including this row's round)
round_order = {"Round 0":0,"Round 1":1,"Round 2":2,"Round 3":3}
bo_meas2 = bo_meas.copy()
bo_meas2["rd"] = bo_meas2["round"].map(round_order)

def incumbent(target, rd):
    prior = meas[meas["round"].map(round_order) < rd]
    return prior[target].max()

for _, r in cmp.iterrows():
    rd = round_order[r["round"]]
    inc = incumbent(r["target"], rd)
    flag = "BEAT" if r["measured"] > inc else "no  "
    print(f"  {r['round']:8s} {r['id']:>3s}  target={r['target']}  "
          f"meas={r['measured']:.4f}  vs prior best {inc:.4f}  -> {flag}")
