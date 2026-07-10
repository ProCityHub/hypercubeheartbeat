#!/usr/bin/env python3
"""OSF Ultimatum experiment for the Lattice Validation Laboratory.

All lattice features are constructed from information available before facial
feedback. The target is never used to define O, A, or B.
"""
from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/osf_ultimatum/source"
OUT = ROOT / "results/osf_ultimatum"
PHI = (1 + math.sqrt(5)) / 2

def z_by_global(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0

def load_and_align():
    m1 = pd.read_csv(DATA / "M1_single.csv")
    m2 = pd.read_csv(DATA / "M2_single.csv")
    m3 = pd.read_csv(DATA / "M3_single.csv")

    # R script coding: 1=accept, 2=reject. Convert to 1/0.
    m2["accept"] = (m2["decision"] == 1).astype(float)
    m3["accept"] = (m3["decision"] == 1).astype(float)
    m3["target_frn"] = m3["N2"] - m3["P2"]

    keys = ["VP_nr", "urtrial"]
    m2_cols = keys + ["offer", "P2", "N2", "P3a", "P3b", "Theta", "accept"]
    m3_cols = keys + ["fac_feed", "target_frn", "P3b", "N170"]
    df = m2[m2_cols].merge(m3[m3_cols], on=keys, how="inner", suffixes=("_offer", "_feedback"))
    df = df.sort_values(keys).reset_index(drop=True)
    return m1, m2, m3, df

def build_features(df):
    neural = ["P2", "N2", "P3a", "P3b_offer", "Theta"]
    # P3b_offer is created by merge suffix; other M2 names remain unsuffixed.
    if "P3b_offer" not in df:
        df = df.rename(columns={"P3b": "P3b_offer"})
    neural = ["P2", "N2", "P3a", "P3b_offer", "Theta"]

    for c in neural:
        df[f"z_{c}"] = z_by_global(df[c])

    # O: root-mean-square pre-feedback neural salience, compressed to [0,1].
    rms = np.sqrt(np.nanmean(np.square(df[[f"z_{c}" for c in neural]].to_numpy()), axis=1))
    df["O"] = 1 - np.exp(-np.nan_to_num(rms, nan=0.0))

    # A: decision/offer decisiveness. Extreme offers and offer-consistent decisions score higher.
    offer_extreme = np.abs(df["offer"] - 2.5) / 2.5
    expected_accept = df["offer"] / 5.0
    decision_alignment = 1 - np.abs(df["accept"] - expected_accept)
    df["A"] = np.clip(0.5 * offer_extreme + 0.5 * decision_alignment, 0, 1)

    # B: temporal integration before feedback. Compare current M2 neural vector with previous trial.
    zcols = [f"z_{c}" for c in neural]
    prev = df.groupby("VP_nr")[zcols].shift(1)
    dist = np.sqrt(np.nanmean(np.square(df[zcols].to_numpy() - prev.to_numpy()), axis=1))
    df["B"] = np.exp(-np.nan_to_num(dist, nan=np.nanmedian(dist[np.isfinite(dist)])))
    df["OAB"] = df["O"] * df["A"] * df["B"]
    df["O_plus_A_plus_B"] = df["O"] + df["A"] + df["B"]
    df["phi_OAB"] = PHI * df["OAB"]

    # Hostile control: destroy temporal meaning while preserving participant-level B distribution.
    rng = np.random.default_rng(20260709)
    df["B_shuffled"] = df.groupby("VP_nr")["B"].transform(
        lambda x: pd.Series(rng.permutation(x.to_numpy()), index=x.index)
    )
    df["OAB_shuffled_B"] = df["O"] * df["A"] * df["B_shuffled"]
    return df

def grouped_cv(df, cols):
    mask = df["target_frn"].notna()
    work = df.loc[mask].copy()
    X = work[cols]
    y = work["target_frn"].to_numpy()
    groups = work["VP_nr"].to_numpy()
    n_splits = min(10, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
    return {
        "features": cols,
        "n": int(len(y)),
        "participants": int(len(np.unique(groups))),
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred))
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m1, m2, m3, df = load_and_align()
    df = build_features(df)

    # Basic reproduction quantities from the supplied coding.
    m1_correct = (m1["fac_feed"] < 70).astype(float)
    descriptives = {
        "M1_rows": int(len(m1)),
        "M2_rows": int(len(m2)),
        "M3_rows": int(len(m3)),
        "aligned_rows": int(len(df)),
        "participants_aligned": int(df["VP_nr"].nunique()),
        "M1_correct_rate": float(m1_correct.mean()),
        "M3_accept_rate": float((m3["decision"] == 1).mean()),
        "M3_mean_feedback_FRN": float((m3["N2"] - m3["P2"]).mean())
    }

    models = {
        "intercept_context": ["offer", "accept"],
        "raw_offer_eeg": ["offer", "accept", "P2", "N2", "P3a", "P3b_offer", "Theta"],
        "oab_only": ["OAB"],
        "additive_oab": ["O", "A", "B"],
        "context_plus_oab": ["offer", "accept", "OAB"],
        "context_plus_additive": ["offer", "accept", "O", "A", "B"],
        "context_plus_shuffled_bridge": ["offer", "accept", "OAB_shuffled_B"]
    }
    results = {name: grouped_cv(df, cols) for name, cols in models.items()}

    # Phi identifiability: positive scalar multiplication leaves ranking unchanged,
    # and standardization/linear coefficients absorb it exactly.
    constants = [1.0, 1.5, math.pi/2, PHI, 2.0]
    correlations = {}
    for c in constants:
        correlations[str(c)] = float(np.corrcoef(df["OAB"], c * df["OAB"])[0,1])
    phi_test = {
        "verdict": "INVALID_TEST",
        "reason": "Multiplying one feature by a positive constant does not change its rank and is absorbed by a fitted linear coefficient or standardization.",
        "constant_correlations_with_unscaled_OAB": correlations,
        "required_future_test": "Phi must alter model structure or create a fixed, preregistered nonlinear prediction that competing constants do not reproduce."
    }

    best_baseline = max(results["intercept_context"]["r2"], results["raw_offer_eeg"]["r2"])
    lattice_gain = results["context_plus_oab"]["r2"] - best_baseline
    shuffle_gap = results["context_plus_oab"]["r2"] - results["context_plus_shuffled_bridge"]["r2"]
    if lattice_gain > 0.01 and shuffle_gap > 0.005:
        verdict = "PARTIALLY_SUPPORTED"
    elif lattice_gain <= 0:
        verdict = "NOT_SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "experiment": "LL-01-OSF-001",
        "descriptives": descriptives,
        "models": results,
        "phi_identifiability": phi_test,
        "primary_verdict": verdict,
        "lattice_r2_gain_over_best_baseline": float(lattice_gain),
        "bridge_shuffle_r2_gap": float(shuffle_gap),
        "limitations": [
            "O, A, and B are exploratory operational definitions.",
            "This analysis does not reproduce mixed-effects inferential statistics from the original R script.",
            "Negative held-out R2 can occur when trial-level EEG is difficult to predict across unseen participants.",
            "No consciousness claim follows from any result."
        ]
    }
    (OUT / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(results).T.to_csv(OUT / "model_comparison.csv")
    df[["VP_nr","urtrial","offer","accept","O","A","B","OAB","target_frn"]].to_csv(
        OUT / "derived_trial_features.csv", index=False
    )

    report = [
        "# OSF Ultimatum Lattice Challenge — Initial Report",
        "",
        f"Primary verdict: **{verdict}**",
        "",
        "## Reproduction descriptives",
        "",
        f"- M1 correct-response rate: {descriptives['M1_correct_rate']:.4f}",
        f"- Aligned M2/M3 trials: {descriptives['aligned_rows']}",
        f"- Participants: {descriptives['participants_aligned']}",
        "",
        "## Held-out participant results",
        "",
        "| Model | R² | MAE |",
        "|---|---:|---:|"
    ]
    for name, r in results.items():
        report.append(f"| {name} | {r['r2']:.6f} | {r['mae']:.6f} |")
    report += [
        "",
        "## Phi finding",
        "",
        "**INVALID TEST in this formulation.** A positive constant multiplier is mathematically non-identifiable in a standardized or coefficient-fitted linear model. The present analysis therefore refuses to count φ scaling as evidence.",
        "",
        "## Interpretation",
        "",
        f"- Lattice gain over the strongest listed baseline: {lattice_gain:.6f} R².",
        f"- Gap between true temporal Bridge and shuffled-Bridge control: {shuffle_gap:.6f} R².",
        "",
        "This result is a challenge to the operational model, not a verdict on every philosophical use of the Lattice Law."
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
