#!/usr/bin/env python3
"""Consolidate individual results JSONs into all_results.json.

Run after any experiment script to keep all_results.json in sync:
    python scripts/consolidate_results.py
"""

import json
from pathlib import Path

METRICS_DIR = Path(__file__).parent.parent / "results" / "metrics"
ALL_RESULTS = METRICS_DIR / "all_results.json"


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def consolidate():
    all_res = load_json(ALL_RESULTS)
    if all_res is None:
        print("ERROR: all_results.json not found")
        return

    updated = []

    # --- training_3ch_history.json → main_results.convlstm_3ch ---
    train = load_json(METRICS_DIR / "training_3ch_history.json")
    if train:
        entry = all_res["main_results"]["convlstm_3ch"]
        entry["val_mse"] = train["val_mse"]
        entry["val_mae"] = train["val_mae"]
        entry["val_rmse"] = train["val_rmse"]
        entry["params"] = train["params"]
        entry["training_epochs_used"] = train["num_epochs"]
        if "val_fom" in train:
            entry["val_fom"] = train["val_fom"]
        entry.pop("note", None)
        updated.append("convlstm_3ch")

    # --- sota_baselines_results.json → main_results.sleuth_ca ---
    sota = load_json(METRICS_DIR / "sota_baselines_results.json")
    if sota:
        if "sleuth_ca" in sota:
            s = sota["sleuth_ca"]
            entry = all_res["main_results"]["sleuth_ca"]
            entry["val_mse"] = s["val_mse"]
            entry["val_mae"] = s["val_mae"]
            entry["val_rmse"] = s["val_rmse"]
            entry["calibrated_params"] = s["calibrated_params"]
            entry["calibration_time_min"] = s["training_time_min"]
            if "val_fom" in s:
                entry["val_fom"] = s["val_fom"]
            updated.append("sleuth_ca")

    # --- ablation_3ch_results.json → ablation_study + linear_extrapolation ---
    abl = load_json(METRICS_DIR / "ablation_3ch_results.json")
    if abl:
        mapping = {
            "ablation_builtup_only": "builtup_only_1ch",
            "ablation_volume_only": "volume_only_1ch",
            "ablation_population_only": "population_only_1ch",
            "ablation_builtup_volume_2ch": "builtup_volume_2ch",
            "ablation_1layer_3ch": "convlstm_1layer_3ch",
            "baseline_cnn_3ch": "cnn_3ch",
            "baseline_unet_3ch": "unet_3ch",
        }
        for src_key, dst_key in mapping.items():
            if src_key in abl:
                s = abl[src_key]
                entry = {
                    "val_mse": s["val_mse"],
                    "val_mae": s["val_mae"],
                    "val_rmse": s["val_rmse"],
                    "params": s["params"],
                }
                if "val_fom" in s:
                    entry["val_fom"] = s["val_fom"]
                all_res["ablation_study"][dst_key] = entry
                updated.append(f"ablation.{dst_key}")

        if "linear_extrapolation" in abl:
            s = abl["linear_extrapolation"]
            entry = all_res["main_results"]["linear_extrapolation_2step"]
            entry["val_mse"] = s["val_mse"]
            entry["val_mae"] = s["val_mae"]
            entry["val_rmse"] = s["val_rmse"]
            if "val_fom" in s:
                entry["val_fom"] = s["val_fom"]
            updated.append("linear_extrapolation_2step")

    # --- validation_2015_results.json → temporal_validation_2015 ---
    val = load_json(METRICS_DIR / "validation_2015_results.json")
    if val:
        tv = all_res["temporal_validation_2015"]
        tv["n_eval_pixels"] = val["n_eval_pixels"]
        # convlstm — base metrics (extended with FoM/growth/SSIM in second pass below)
        tv["convlstm"] = {
            "mse": round(val["convlstm"]["mse"], 6),
            "r2": round(val["convlstm"]["r2"], 3),
        }
        # linear — include all metrics present in the results file
        lin = val["linear_extrapolation"]
        tv["linear_extrapolation"] = {
            "mse": round(lin["mse"], 6),
            "r2":  round(lin["r2"],  3),
        }
        for field in ("fom", "ssim", "growth_mse", "stable_mse"):
            if field in lin:
                tv["linear_extrapolation"][field] = lin[field]
        # persistence
        per = val["persistence"]
        tv["persistence"] = {"mse": round(per["mse"], 6)}
        for field in ("fom", "ssim", "growth_mse"):
            if field in per:
                tv["persistence"][field] = per[field]
        updated.append("temporal_validation_2015")

    # --- uncertainty_summary.json → uncertainty_quantification ---
    unc = load_json(METRICS_DIR / "uncertainty_summary.json")
    if unc and "uncertainty_stats" in unc:
        s = unc["uncertainty_stats"]
        uq = all_res["uncertainty_quantification"]
        uq["mean_std"] = s["mean_std"]
        uq["mean_cv"] = s["mean_cv"]
        uq["mean_ci95_width"] = s["mean_ci95_width"]
        if "calibration" in unc:
            uq["calibration_pearson_r"] = unc["calibration"]["pearson_r_std_vs_error"]
        updated.append("uncertainty")

    # --- ablation_3ch_results.json → sequence length ablation ---
    abl = load_json(METRICS_DIR / "ablation_3ch_results.json")
    if abl:
        seqlen_keys = [k for k in abl if k.startswith("seqlen_")]
        if seqlen_keys:
            if "sequence_length_ablation" not in all_res:
                all_res["sequence_length_ablation"] = {}
            for k in seqlen_keys:
                s = abl[k]
                all_res["sequence_length_ablation"][k] = {
                    "n_timesteps": s.get("n_timesteps"),
                    "epochs_used": s.get("epochs_used"),
                    "val_mse": s["val_mse"],
                    "val_mae": s["val_mae"],
                    "val_rmse": s["val_rmse"],
                }
                updated.append(f"seqlen.{k}")

    # --- validation_2015_results.json → new metrics (FoM, growth-region, SSIM) ---
    val2 = load_json(METRICS_DIR / "validation_2015_results.json")
    if val2 and "convlstm" in val2:
        tv = all_res["temporal_validation_2015"]
        c  = val2["convlstm"]
        # Extend with new metrics if present
        for field in ("fom", "ssim", "growth_mse", "growth_r2", "stable_mse"):
            if field in c:
                tv["convlstm"][field] = c[field]
        if "fom_vs_linear_pp" in val2:
            tv["fom_vs_linear_pp"] = val2["fom_vs_linear_pp"]
        if "growth_mse_vs_linear_pct" in val2:
            tv["growth_mse_vs_linear_pct"] = val2["growth_mse_vs_linear_pct"]

    # --- multihorizon_results.json → multi_horizon (new section) ---
    mh = load_json(METRICS_DIR / "multihorizon_results.json")
    if mh and "horizons" in mh:
        if "multi_horizon" not in all_res:
            all_res["multi_horizon"] = {}
        for horizon_name, hr in mh["horizons"].items():
            conv = hr["convlstm"]
            entry = {
                "years_ahead": hr["years_ahead"],
                "last_input_year": hr["last_input_year"],
                "convlstm_mse": conv.get("val_mse", conv.get("mse")),
                "convlstm_r2":  conv.get("r2"),
                "linear_ols_mse": hr["linear_regression_baseline"]["mse"],
                "linear_ols_r2":  hr["linear_regression_baseline"]["r2"],
                "improvement_vs_linear_ols_pct": hr["improvement_vs_linear_pct"],
            }
            if "val_fom" in hr["convlstm"]:
                entry["convlstm_fom"] = hr["convlstm"]["val_fom"]
            if "fom" in hr["linear_regression_baseline"]:
                entry["linear_ols_fom"] = hr["linear_regression_baseline"]["fom"]
            all_res["multi_horizon"][horizon_name] = entry
        updated.append("multi_horizon")

    # --- cnn_multihorizon_results.json → multi_horizon (add CNN columns) ---
    cnn_mh = load_json(METRICS_DIR / "cnn_multihorizon_results.json")
    if cnn_mh and "horizons" in cnn_mh:
        if "multi_horizon" not in all_res:
            all_res["multi_horizon"] = {}
        for horizon_name, hr in cnn_mh["horizons"].items():
            entry = all_res["multi_horizon"].setdefault(horizon_name, {})
            entry["cnn_mse"]    = hr["val_mse"]
            entry["cnn_r2"]     = hr["val_r2"]
            entry["cnn_fom"]    = hr["val_fom"]
            entry["cnn_mae"]    = hr["val_mae"]
            entry["cnn_params"] = hr["params"]
        # Store comparison note
        if "key_finding" in cnn_mh:
            all_res["multi_horizon"]["_cnn_key_finding"] = cnn_mh["key_finding"]
        updated.append("multi_horizon.cnn")

    # --- cnn_channel_control_results.json → new cnn_channel_control section ---
    cnn_cc = load_json(METRICS_DIR / "cnn_channel_control_results.json")
    if cnn_cc:
        all_res["cnn_channel_control"] = {
            "description": cnn_cc.get("description", ""),
            "configs": {},
            "paired_comparisons": cnn_cc.get("paired_comparisons", {}),
        }
        for cfg_name, cfg in cnn_cc.get("configs", {}).items():
            all_res["cnn_channel_control"]["configs"][cfg_name] = {
                "val_mse":       cfg["val_mse"],
                "val_fom":       cfg["val_fom"],
                "val_r2":        cfg["val_r2"],
                "n_channels":    cfg["n_channels"],
                "n_input_epochs": cfg["n_input_epochs"],
                "horizon_years": cfg["horizon_years"],
                "params":        cfg["params"],
            }
        updated.append("cnn_channel_control")

    # --- validation_2020_results.json → temporal_holdout_2020 ---
    val20 = load_json(METRICS_DIR / "validation_2020_results.json")
    if val20:
        th = all_res.setdefault("temporal_holdout_2020", {})
        c20 = val20["convlstm_best"]
        th["n_eval_pixels"] = val20["n_eval_pixels"]
        th["pct_growth_pixels"] = val20["pct_growth"]
        th["convlstm_best"] = {
            "mse":  round(c20["mse"], 6),
            "mae":  round(c20["mae"], 6),
            "r2":   round(c20["r2"], 4),
            "fom":  round(c20["fom"], 4),
            "ssim": round(c20["ssim"], 4),
            "growth_mse": c20["growth_mse"],
            "growth_r2":  c20["growth_r2"],
            "stable_mse": c20["stable_mse"],
        }
        lin20 = val20["linear_extrapolation"]
        th["linear_extrapolation"] = {
            "mse": round(lin20["mse"], 6),
            "r2":  round(lin20["r2"], 4),
            "fom": round(lin20["fom"], 4),
            "growth_mse": lin20["growth_mse"],
        }
        per20 = val20["persistence"]
        th["persistence"] = {
            "mse": round(per20["mse"], 6),
            "fom": 0.0,
            "growth_mse": per20["growth_mse"],
        }
        th["improvement_vs_linear_mse_pct"] = val20["improvement_convlstm_vs_linear_mse_pct"]
        th["fom_gap_vs_linear_pp"] = val20["fom_gap_vs_linear_pp"]
        th["growth_mse_improvement_pct"] = val20["growth_mse_improvement_pct"]
        updated.append("temporal_holdout_2020")

    # Write consolidated file
    with open(ALL_RESULTS, "w") as f:
        json.dump(all_res, f, indent=2)
        f.write("\n")

    print(f"Updated {len(updated)} entries in all_results.json:")
    for u in updated:
        print(f"  - {u}")


if __name__ == "__main__":
    consolidate()
