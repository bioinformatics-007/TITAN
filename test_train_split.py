import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
import warnings
# Suppress performance warnings for a cleaner console
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ============================================================
# 1. LOAD DATA
# ============================================================
# UPDATE THIS: Use your new combined file
input_file = "FINAL_COMBINED_EPITOPE_DATA_with_cross_reactivity.csv"
df = pd.read_csv(input_file, low_memory=False)
df.columns = df.columns.str.strip()
print(f"Original Shape: {df.shape}")

# Safety check
if df.shape[1] <= 1:
    print("CRITICAL ERROR: Data loaded incorrectly. Only 1 column found.")
    print("If your file is tab-separated, change the read_csv line to sep='\\t'")
    exit()

# ============================================================
# 2. DEDUPLICATION & INITIAL CLEANING
# ============================================================
required = ["protein_name", "peptide", "epitope_type"]
for col in required:
    if col not in df.columns:
        print(f"CRITICAL ERROR: Missing required column: {col}")
        exit()

df["epitope_id"] = (
    df["protein_name"].astype(str) + "_" +
    df["peptide"].astype(str) + "_" +
    df["epitope_type"].astype(str)
)
df = df.drop_duplicates(subset=["epitope_id"]).copy()
# Convert infinities to NaNs for proper median handling later
df = df.replace([np.inf, -np.inf], np.nan)

# ============================================================
# 3. PATHOGEN-AWARE SPLIT (Zero-Leakage Point)
# ============================================================
split_col = "pathogen"
# NEW: Fix for the NaN error
nan_count = df[split_col].isna().sum()
if nan_count > 0:
    print(f"⚠️ Removing {nan_count} rows with missing pathogen labels...")
    df = df.dropna(subset=[split_col])

if split_col not in df.columns:
    print(f"ERROR: {split_col} column not found.")
    exit()

# Ensure pathogen names are strings
df[split_col] = df[split_col].astype(str)
print(f"Splitting data based on {df[split_col].nunique()} unique pathogens...")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df[split_col]))
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# ============================================================
# 4. LEAKAGE-FREE IMPUTATION (Learned on Train Only)
# ============================================================
print("Performing leakage-free imputation...")
numeric_cols_all = train_df.select_dtypes(include=np.number).columns
# Calculate medians ONLY from training pathogens
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
