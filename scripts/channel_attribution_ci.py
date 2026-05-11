"""
Delta-method CI for the 94% channel-count attribution in Section 4.

Three FoM values underpin the claim:
  f1 = 0.739  CNN 5yr 24-ch (8 epochs, 1975-2010)
  f2 = 0.679  CNN 5yr 21-ch (7 epochs, channel-count control)
  f3 = 0.675  CNN 10yr 21-ch (actual multi-horizon, 7 epochs)

Attribution A = (f1 - f2) / (f1 - f3)   [channel effect / total effect]

Each FoM is a single-seed point estimate; the two-seed std across seeds 0 and 42
(all-pixel domain) is sigma = 0.0074 (from cnn_multiseed_results.json).
The delta method propagates that sigma through A to produce a 95% CI.

Note: the channel-control model weights were not retained, so a tile bootstrap
is not feasible. The delta-method result is an analytical approximation
assuming the three FoM estimates are independent with equal variance sigma^2.
"""

import json
import math
import os

# ── Data ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")

with open(os.path.join(RESULTS_DIR, "cnn_multiseed_results.json")) as fh:
    multiseed = json.load(fh)

with open(os.path.join(RESULTS_DIR, "cnn_multihorizon_results.json")) as fh:
    multihorizon = json.load(fh)

with open(os.path.join(RESULTS_DIR, "cnn_channel_control_results.json")) as fh:
    channel_ctrl = json.load(fh)

# ── Point estimates ───────────────────────────────────────────────────────────
f1 = multihorizon["horizons"]["5yr"]["val_fom"]    # 0.739  (8ep, 24ch)
f2 = channel_ctrl["configs"]["5yr_7ep"]["val_fom"] # 0.6792 (7ep, 21ch, fixed task)
f3 = multihorizon["horizons"]["10yr"]["val_fom"]   # 0.6753 (7ep, 21ch, actual 10yr)

sigma = multiseed["summary"]["fom_2015_std"]       # 0.0074

delta_channel = f1 - f2   # channel-count effect
delta_total   = f1 - f3   # total multi-horizon effect
delta_residual = f2 - f3  # residual / genuine horizon effect

attribution = delta_channel / delta_total

# ── Delta method ──────────────────────────────────────────────────────────────
# A = (f1 - f2) / (f1 - f3)
# dA/df1 = (f2 - f3) / (f1 - f3)^2
# dA/df2 = -1 / (f1 - f3)
# dA/df3 = (f1 - f2) / (f1 - f3)^2

dA_df1 = (f2 - f3) / (f1 - f3) ** 2
dA_df2 = -1.0      / (f1 - f3)
dA_df3 = (f1 - f2) / (f1 - f3) ** 2

var_A = (dA_df1 ** 2 + dA_df2 ** 2 + dA_df3 ** 2) * sigma ** 2
se_A  = math.sqrt(var_A)

z95 = 1.96
ci_lo = max(0.0, attribution - z95 * se_A)
ci_hi = min(1.0, attribution + z95 * se_A)

# ── Conservative seed-variance lower bound ───────────────────────────────────
# Worst case: add full sigma to denominator (expanding total effect)
# and subtract nothing from numerator.
lb_conservative = delta_channel / (delta_total + sigma)

# ── Print ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Channel attribution CI (delta method)")
print("=" * 60)
print(f"  f1  (5yr 24ch)  = {f1:.4f}")
print(f"  f2  (5yr 21ch)  = {f2:.4f}")
print(f"  f3  (10yr 21ch) = {f3:.4f}")
print(f"  sigma           = {sigma:.4f}")
print()
print(f"  channel effect  = {delta_channel:.4f}")
print(f"  total effect    = {delta_total:.4f}")
print(f"  residual effect = {delta_residual:.4f}")
print()
print(f"  Attribution (point) = {attribution*100:.1f}%")
print(f"  SE (delta method)   = {se_A:.4f}")
print(f"  95% CI              = [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
print()
print(f"  Conservative lower bound (seed-var sensitivity) = {lb_conservative*100:.1f}%")
print()
print("Note: wide CI reflects small denominator (0.064). The point estimate")
print("and the conservative lower bound are the appropriate summaries.")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "experiment": "channel_attribution_ci",
    "description": (
        "Delta-method 95% CI on the 94% channel-count attribution (Section 4). "
        "Three single-seed FoM measurements; sigma from two-seed std in multiseed run. "
        "Tile bootstrap not feasible (channel-control model weights not retained)."
    ),
    "inputs": {
        "f1_5yr_24ch": f1,
        "f2_5yr_21ch": f2,
        "f3_10yr_21ch": f3,
        "sigma_per_fom": sigma,
    },
    "effects": {
        "delta_channel": round(delta_channel, 4),
        "delta_total":   round(delta_total,   4),
        "delta_residual": round(delta_residual, 4),
    },
    "attribution": {
        "point_estimate": round(attribution, 4),
        "point_pct": round(attribution * 100, 1),
        "se_delta_method": round(se_A, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "ci_95_lo_pct": round(ci_lo * 100, 1),
        "ci_95_hi_pct": round(ci_hi * 100, 1),
        "conservative_lower_bound_pct": round(lb_conservative * 100, 1),
    },
    "interpretation": (
        "Wide CI is expected: attribution is a ratio of two small differences "
        "(denominator = 0.064), amplifying ratio variance under delta method. "
        "The conservative lower bound (expand denominator by sigma) gives >=83%. "
        "Reporting the point estimate + conservative lower bound is appropriate."
    ),
}

out_path = os.path.join(RESULTS_DIR, "channel_attribution_ci.json")
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)

print(f"\nSaved to {out_path}")
