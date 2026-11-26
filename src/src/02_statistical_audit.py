"""
Two-Stage Statistical Integrity Audit: Benford's Law & Z-Score Bot Filtering
Validates engagement distribution integrity and purges anomalous click-farm records (Z > 3.5).
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import chisquare


def benford_expected_probabilities() -> np.ndarray:
    """Computes theoretical first-digit probabilities P(d) = log10(1 + 1/d) for d in 1..9."""
    digits = np.arange(1, 10)
    return np.log10(1.0 + 1.0 / digits)


def extract_leading_digits(series: pd.Series) -> pd.Series:
    """Extracts the first significant non-zero digit from positive numeric values."""
    clean = series.dropna()
    clean = clean[clean > 0]
    first_digits = clean.astype(str).str.lstrip("0.").str[0]
    return pd.to_numeric(first_digits, errors="coerce").dropna().astype(int)


def run_benford_audit(series: pd.Series) -> dict:
    """Executes Chi-square Goodness-of-Fit test against Benford's Law."""
    digits = extract_leading_digits(series)
    observed_counts = np.array([(digits == d).sum() for d in range(1, 10)], dtype=float)
    expected_counts = benford_expected_probabilities() * observed_counts.sum()
    chi2_stat, p_val = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    return {
        "n_evaluated": int(observed_counts.sum()),
        "chi2_statistic": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 6),
        "conforms_benford_p01": bool(p_val >= 0.01)
    }


def filter_zscore_anomalies(df: pd.DataFrame, target_cols: list[str], threshold: float = 3.5) -> pd.DataFrame:
    """Purges records exceeding |Z| > threshold across audited engagement metrics."""
    mask = pd.Series(True, index=df.index)
    for col in target_cols:
        if col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std(ddof=1)
            if col_std > 0:
                z_scores = (df[col] - col_mean).abs() / col_std
                mask &= (z_scores <= threshold)
    return df[mask].copy()


def main():
    parser = argparse.ArgumentParser(description="Benford's Law and Z-Score Statistical Audit")
    parser.add_argument(
        "--input",
        type=str,
        default="douyin_suspense_audiobook_audited_dataset.csv",
        help="Path to dataset CSV"
    )
    parser.add_argument("--z-threshold", type=float, default=3.5, help="Z-score anomaly cutoff")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded dataset: {args.input} | Records: {len(df)}")

    # Identify engagement or circulation numeric columns available in the dataset
    candidate_cols = [c for c in ["Ei", "global_engagement", "likes", "Vcirc", "circulation_velocity"] if c in df.columns]
    if not candidate_cols:
        candidate_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:3]

    print("\n--- Stage 1: Benford's Logarithmic Law Audit ---")
    for col in candidate_cols:
        audit_res = run_benford_audit(df[col])
        print(f"Variable '{col}': N={audit_res['n_evaluated']}, Chi2={audit_res['chi2_statistic']}, p={audit_res['p_value']}")

    print(f"\n--- Stage 2: Z-Score Anomaly Filtering (|Z| <= {args.z_threshold}) ---")
    df_clean = filter_zscore_anomalies(df, candidate_cols, threshold=args.z_threshold)
    purged = len(df) - len(df_clean)
    print(f"Retained valid records: {len(df_clean)} | Flagged/Purged outliers: {purged}")


if __name__ == "__main__":
    main()
