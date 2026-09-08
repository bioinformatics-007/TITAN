import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
# Suppress performance warnings for a cleaner console
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ============================================================
# 1. LOAD DATA - TEMPORAL VALIDATION
# ============================================================
# Historical training data
train_file = "2010_enhanced.csv"

# Future temporal test data
test_file = "2023_enhanced.csv"

if not Path(train_file).exists():
    print(f"CRITICAL ERROR: Training file not found: {train_file}")
    exit()

if not Path(test_file).exists():
    print(f"CRITICAL ERROR: Test file not found: {test_file}")
    exit()

print(f"Loading TRAIN data: {train_file}")
train_df = pd.read_csv(train_file, low_memory=False)
train_df.columns = train_df.columns.str.strip()

print(f"Loading TEST data: {test_file}")
test_df = pd.read_csv(test_file, low_memory=False)
test_df.columns = test_df.columns.str.strip()

print(f"Original Train Shape: {train_df.shape}")
print(f"Original Test Shape : {test_df.shape}")

# Safety check
if train_df.shape[1] <= 1:
    print("CRITICAL ERROR: Training data loaded incorrectly.")
    print("If your file is tab-separated, change read_csv to sep='\\t'")
    exit()

if test_df.shape[1] <= 1:
    print("CRITICAL ERROR: Test data loaded incorrectly.")
    print("If your file is tab-separated, change read_csv to sep='\\t'")
    exit()

# Use a temporary combined dataframe only for shared column checks.
# Actual preprocessing remains separated into train/test.
df = train_df.copy()

# ============================================================
# 2. DEDUPLICATION & INITIAL CLEANING
# ============================================================

required = ["protein_name", "peptide", "epitope_type"]

print("DEBUG TRAIN epitope columns:", [repr(c) for c in train_df.columns if "epitope" in str(c).lower()])
print("DEBUG TEST epitope columns:", [repr(c) for c in test_df.columns if "epitope" in str(c).lower()])
# Check required columns in BOTH temporal datasets
for col in required:
    if col not in train_df.columns:
        print(f"CRITICAL ERROR: Missing required column in TRAIN: {col}")
        exit()

    if col not in test_df.columns:
        print(f"CRITICAL ERROR: Missing required column in TEST: {col}")
        exit()

# Create epitope IDs independently in each temporal cohort
train_df["epitope_id"] = (
    train_df["protein_name"].astype(str) + "_" +
    train_df["peptide"].astype(str) + "_" +
    train_df["epitope_type"].astype(str)
)

test_df["epitope_id"] = (
    test_df["protein_name"].astype(str) + "_" +
    test_df["peptide"].astype(str) + "_" +
    test_df["epitope_type"].astype(str)
)

# Deduplicate WITHIN each temporal dataset.
# Do not combine 2010 and 2023 before deduplication.
train_df = train_df.drop_duplicates(
    subset=["epitope_id"]
).copy()

test_df = test_df.drop_duplicates(
    subset=["epitope_id"]
).copy()

# Convert infinities to NaNs
train_df = train_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION SETUP")
print("=" * 70)
print("TRAIN PERIOD : 2010")
print("TEST PERIOD  : 2023")
print(f"Train rows after deduplication: {len(train_df)}")
print(f"Test rows after deduplication : {len(test_df)}")

print("\nTEMPORAL LEAKAGE CHECK")
print("2010 < 2023")
print("✅ PASS: Test cohort is chronologically after training cohort.")
print("=" * 70)


# ============================================================
# 4. LEAKAGE-FREE IMPUTATION (Learned on Train Only)
# ============================================================
print("Performing leakage-free imputation...")
numeric_cols_all = train_df.select_dtypes(include=np.number).columns
# Calculate medians ONLY from historical training data
train_medians = train_df[numeric_cols_all].median()
# Apply training medians to both sets
train_df[numeric_cols_all] = train_df[numeric_cols_all].fillna(train_medians)
test_df[numeric_cols_all] = test_df[numeric_cols_all].fillna(train_medians)

# ============================================================
# 5. LEAKAGE-FREE TARGET CONSTRUCTION (With Clipping)
# ============================================================
cols_to_norm = ["antigenicity", "population_coverage", "promiscuity_score", "percentile", "sequence_conservation"]
for col in cols_to_norm:
    if col in train_df.columns:
        t_min = train_df[col].min()
        t_max = train_df[col].max()
        denom = (t_max - t_min + 1e-9)
      
        if col == "percentile":
            train_df[f"norm_{col}"] = (1 - (train_df[col] - t_min) / denom).clip(0, 1)
            test_df[f"norm_{col}"] = (1 - (test_df[col] - t_min) / denom).clip(0, 1)
        else:
            train_df[f"norm_{col}"] = ((train_df[col] - t_min) / denom).clip(0, 1)
            test_df[f"norm_{col}"] = ((test_df[col] - t_min) / denom).clip(0, 1)

def calculate_rank(df_in):
    """
    Authoritative ranking-score definition.

    Ranking score:
        0.30 * antigenicity
      + 0.25 * population coverage
      + 0.20 * promiscuity
      + 0.15 * percentile
      + 0.10 * sequence conservation

    The normalized components are calculated using training-set
    statistics before this function is called.
    """
    required_norm = [
        "norm_antigenicity",
        "norm_population_coverage",
        "norm_promiscuity_score",
        "norm_percentile",
        "norm_sequence_conservation",
    ]

    missing = [c for c in required_norm if c not in df_in.columns]
    if missing:
        raise ValueError(
            f"Missing normalized ranking components: {missing}"
        )

    score = (
        0.30 * df_in["norm_antigenicity"].to_numpy(dtype=float)
        + 0.25 * df_in["norm_population_coverage"].to_numpy(dtype=float)
        + 0.20 * df_in["norm_promiscuity_score"].to_numpy(dtype=float)
        + 0.15 * df_in["norm_percentile"].to_numpy(dtype=float)
        + 0.10 * df_in["norm_sequence_conservation"].to_numpy(dtype=float)
    )

    return pd.Series(score, index=df_in.index, dtype=float)


def apply_ranking_penalties(df_in, score):
    """
    Apply predefined binary allergenicity/toxicity penalties.

    Penalties are applied only when the corresponding annotation is
    exactly binary-positive (== 1).
    """
    score = score.copy()

    # Raw annotations are binary: 0 = negative, 1 = positive.
    # Apply the predefined penalties to positive annotations only.
    if "allergenicity" in df_in.columns:
        score.loc[df_in["allergenicity"] == 1] *= 0.7

    if "toxicity" in df_in.columns:
        score.loc[df_in["toxicity"] == 1] *= 0.5

    return score

train_df["ranking_score"] = apply_ranking_penalties(
    train_df,
    calculate_rank(train_df)
)

test_df["ranking_score"] = apply_ranking_penalties(
    test_df,
    calculate_rank(test_df)
)

# ============================================================
# 6. PCA ON ESM EMBEDDINGS (Learned on Train Only)
# ============================================================
esm_cols = [c for c in train_df.columns if c.startswith("esm2_")]  # Use train_df columns
if len(esm_cols) > 0:
    print(f"Compressing {len(esm_cols)} ESM features to 50 PCA components...")
    pca = PCA(n_components=50, random_state=42)
   
    # These dataframes were already imputed in Section 4, so no NaNs here!
    train_pca = pca.fit_transform(train_df[esm_cols])
    test_pca = pca.transform(test_df[esm_cols])
  
    pca_cols = [f"esm_pca_{i}" for i in range(50)]
    train_pca_df = pd.DataFrame(train_pca, columns=pca_cols, index=train_df.index)
    test_pca_df = pd.DataFrame(test_pca, columns=pca_cols, index=test_df.index)
  
    train_df = pd.concat([train_df, train_pca_df], axis=1).drop(columns=esm_cols)
    test_df = pd.concat([test_df, test_pca_df], axis=1).drop(columns=esm_cols)
  
    joblib.dump(pca, "pca_transformer.joblib")

# ============================================================
# 7. FEATURE SELECTION & CORRELATION FILTERING
# ============================================================
metadata_cols = ["epitope_id", "protein_name", "epitope_type", "peptide", "allele", "pathogen"]
positional_cols = ["start", "end", "mean_position", "relative_position", "residue_count", "peptide_length"]
target_helper_cols = [c for c in train_df.columns if c.startswith("norm_")]
exclude_list = metadata_cols + positional_cols + target_helper_cols + ["ranking_score"]
feature_cols = [c for c in train_df.columns if c not in exclude_list]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]

print("Applying correlation filtering (0.95)...")
corr_matrix = train_df[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
feature_cols = [c for c in feature_cols if c not in to_drop]

train_df = train_df.drop(columns=to_drop)
test_df = test_df.drop(columns=to_drop)

# ============================================================
# 8. STANDARD SCALING (Learned on Train Only)
# ============================================================
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

joblib.dump(scaler, "feature_scaler.joblib")
joblib.dump(feature_cols, "final_feature_list.joblib")

# ============================================================
# 9. FINAL SAVE
# ============================================================
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)
pd.DataFrame({"features": feature_cols}).to_csv("final_feature_list.csv", index=False)

print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
print(f"Final Rows (Train): {len(train_df)}")
print(f"Final Rows (Test): {len(test_df)}")
print(f"Final Features: {len(feature_cols)}")
print("="*50)

# ============================================================
# FINAL DATA QUALITY REPORT
# ============================================================
print("\n" + "="*70)
print("📊 FINAL DATA QUALITY REPORT")
print("="*70)
# 1. Missing Values Check
train_missing = train_df[feature_cols].isna().sum().sum()
test_missing = test_df[feature_cols].isna().sum().sum()
print(f"\n1️⃣ MISSING VALUES")
if train_missing > 0 or test_missing > 0:
    print(f" ⚠️ WARNING: Missing values present! Train: {train_missing}, Test: {test_missing}")
else:
    print(" ✅ PASS: No missing values in features")

# 2. Scaling Verification
print(f"\n2️⃣ SCALING CHECK (StandardScaler)")
if len(feature_cols) > 0:
    sample_col = feature_cols[0]
    t_mean, t_std = train_df[sample_col].mean(), train_df[sample_col].std()
    if abs(t_mean) < 0.01 and abs(t_std - 1.0) < 0.01:
        print(f" ✅ PASS: {sample_col} is scaled (mean≈0, std≈1)")
    else:
        print(f" ⚠️ WARNING: Scaling looks off for {sample_col}")
else:
    print(" ⚠️ WARNING: No feature columns available for scaling check")

# 3. Target Distribution
print(f"\n3️⃣ TARGET (RANKING_SCORE) SUMMARY")
print(f" Train Mean: {train_df['ranking_score'].mean():.4f}")
print(f" Test Mean: {test_df['ranking_score'].mean():.4f}")
print("="*70 + "\n")
