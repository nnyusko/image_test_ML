import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import joblib

# =====================================================================================
# UTILS: Provided in the competition description
# =====================================================================================
tqdm.pandas()

def convert_age(val):
    if pd.isna(val): return np.nan
    try:
        base = int(str(val)[:-1])
        return base if str(val)[-1] == "a" else base + 5
    except:
        return np.nan

def split_testdate(val):
    try:
        v = int(val)
        return v // 100, v % 100
    except:
        return np.nan, np.nan

def seq_mean(series):
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )

def seq_std(series):
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

def seq_rate(series, target="1"):
    return series.fillna("").progress_apply(
        lambda x: str(x).split(",").count(target) / len(x.split(',')) if x else np.nan
    )

def masked_mean_from_csv_series(cond_series, val_series, mask_val):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)
    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts==0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

def masked_mean_in_set_series(cond_series, val_series, mask_set):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)
    mask = np.isin(cond_arr, list(mask_set))
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

# =====================================================================================
# PREPROCESS: Provided in the competition description
# =====================================================================================
def preprocess_A(df_A):
    df = df_A.copy()
    print("Preprocessing A: Step 1: Deriving Age, TestDate...")
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    print("Preprocessing A: Step 2: A1 features...")
    feats["A1_resp_rate"] = seq_rate(df["A1-3"], "1")
    feats["A1_rt_mean"]   = seq_mean(df["A1-4"])
    feats["A1_rt_std"]    = seq_std(df["A1-4"])
    feats["A1_rt_left"]   = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 1)
    feats["A1_rt_right"]  = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 2)
    feats["A1_rt_side_diff"] = feats["A1_rt_left"] - feats["A1_rt_right"]
    feats["A1_rt_slow"]   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 1)
    feats["A1_rt_fast"]   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 3)
    feats["A1_rt_speed_diff"] = feats["A1_rt_slow"] - feats["A1_rt_fast"]

    print("Preprocessing A: Step 3: A2 features...")
    feats["A2_resp_rate"] = seq_rate(df["A2-3"], "1")
    feats["A2_rt_mean"]   = seq_mean(df["A2-4"])
    feats["A2_rt_std"]    = seq_std(df["A2-4"])
    feats["A2_rt_cond1_diff"] = masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 1) - \
                                masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 3)
    feats["A2_rt_cond2_diff"] = masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 1) - \
                                masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 3)

    print("Preprocessing A: Step 4: A3 features...")
    s = df["A3-5"].fillna("")
    total   = s.apply(lambda x: len(x.split(",")) if x else 0)
    valid   = s.apply(lambda x: sum(v in {"1","2"} for v in x.split(",")) if x else 0)
    invalid = s.apply(lambda x: sum(v in {"3","4"} for v in x.split(",")) if x else 0)
    correct = s.apply(lambda x: sum(v in {"1","3"} for v in x.split(",")) if x else 0)
    feats["A3_valid_ratio"]   = (valid / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_invalid_ratio"] = (invalid / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_correct_ratio"] = (correct / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_resp2_rate"] = seq_rate(df["A3-6"], "1")
    feats["A3_rt_mean"]    = seq_mean(df["A3-7"])
    feats["A3_rt_std"]     = seq_std(df["A3-7"])
    feats["A3_rt_size_diff"] = masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 1) - \
                               masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 2)
    feats["A3_rt_side_diff"] = masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 1) - \
                               masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 2)

    print("Preprocessing A: Step 5: A4 features...")
    feats["A4_acc_rate"]   = seq_rate(df["A4-3"], "1")
    feats["A4_resp2_rate"] = seq_rate(df["A4-4"], "1")
    feats["A4_rt_mean"]    = seq_mean(df["A4-5"])
    feats["A4_rt_std"]     = seq_std(df["A4-5"])
    feats["A4_stroop_diff"] = masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2) - \
                              masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)
    feats["A4_rt_color_diff"] = masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 1) - \
                                masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 2)

    print("Preprocessing A: Step 6: A5 features...")
    feats["A5_acc_rate"]   = seq_rate(df["A5-2"], "1")
    feats["A5_resp2_rate"] = seq_rate(df["A5-3"], "1")
    feats["A5_acc_nonchange"] = masked_mean_from_csv_series(df["A5-1"], df["A5-2"], 1)
    feats["A5_acc_change"]    = masked_mean_in_set_series(df["A5-1"], df["A5-2"], {2,3,4})

    print("Preprocessing A: Step 7: Dropping sequence columns...")
    seq_cols = [c for c in df.columns if c.startswith('A') and '-' in c]
    out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    print("Finished preprocessing for A.")
    return out

def preprocess_B(df_B):
    df = df_B.copy()
    print("Preprocessing B: Step 1: Deriving Age, TestDate...")
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    print("Preprocessing B: Step 2: B1 features...")
    feats["B1_acc_task1"] = seq_rate(df["B1-1"], "1")
    feats["B1_rt_mean"]   = seq_mean(df["B1-2"])
    feats["B1_rt_std"]    = seq_std(df["B1-2"])
    feats["B1_acc_task2"] = seq_rate(df["B1-3"], "1")

    print("Preprocessing B: Step 3: B2 features...")
    feats["B2_acc_task1"] = seq_rate(df["B2-1"], "1")
    feats["B2_rt_mean"]   = seq_mean(df["B2-2"])
    feats["B2_rt_std"]    = seq_std(df["B2-2"])
    feats["B2_acc_task2"] = seq_rate(df["B2-3"], "1")

    print("Preprocessing B: Step 4: B3 features...")
    feats["B3_acc_rate"] = seq_rate(df["B3-1"], "1")
    feats["B3_rt_mean"]  = seq_mean(df["B3-2"])
    feats["B3_rt_std"]   = seq_std(df["B3-2"])

    print("Preprocessing B: Step 5: B4 features...")
    feats["B4_acc_rate"] = seq_rate(df["B4-1"], "1")
    feats["B4_rt_mean"]  = seq_mean(df["B4-2"])
    feats["B4_rt_std"]   = seq_std(df["B4-2"])

    print("Preprocessing B: Step 6: B5 features...")
    feats["B5_acc_rate"] = seq_rate(df["B5-1"], "1")
    feats["B5_rt_mean"]  = seq_mean(df["B5-2"])
    feats["B5_rt_std"]   = seq_std(df["B5-2"])

    print("Preprocessing B: Step 7: B6-B8 features...")
    feats["B6_acc_rate"] = seq_rate(df["B6"], "1")
    feats["B7_acc_rate"] = seq_rate(df["B7"], "1")
    feats["B8_acc_rate"] = seq_rate(df["B8"], "1")

    print("Preprocessing B: Step 8: Dropping sequence columns...")
    seq_cols = [c for c in df.columns if c.startswith('B') and ('-' in c or c in ['B6','B7','B8'])]
    out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    print("Finished preprocessing for B.")
    return out

# =====================================================================================
# FEATURE ENGINEERING: Provided in the competition description
# =====================================================================================
def _has(df, cols): return all(c in df.columns for c in cols)
def _safe_div(a, b, eps=1e-6): return a / (b + eps)

def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6
    if _has(feats, ["Year","Month"]): feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]
    if _has(feats, ["A1_rt_mean","A1_resp_rate"]): feats["A1_speed_acc_tradeoff"] = _safe_div(feats["A1_rt_mean"], feats["A1_resp_rate"], eps)
    if _has(feats, ["A2_rt_mean","A2_resp_rate"]): feats["A2_speed_acc_tradeoff"] = _safe_div(feats["A2_rt_mean"], feats["A2_resp_rate"], eps)
    if _has(feats, ["A4_rt_mean","A4_acc_rate"]): feats["A4_speed_acc_tradeoff"] = _safe_div(feats["A4_rt_mean"], feats["A4_acc_rate"], eps)
    for k in ["A1","A2","A3","A4"]: 
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]): feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)
    for name, base in [("A1_rt_side_gap_abs", "A1_rt_side_diff"), ("A1_rt_speed_gap_abs", "A1_rt_speed_diff"), ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"), ("A2_rt_cond2_gap_abs", "A2_rt_cond2_diff"), ("A4_stroop_gap_abs", "A4_stroop_diff"), ("A4_color_gap_abs", "A4_rt_color_diff")]:
        if base in feats.columns: feats[name] = feats[base].abs()
    if _has(feats, ["A3_valid_ratio","A3_invalid_ratio"]): feats["A3_valid_invalid_gap"] = feats["A3_valid_ratio"] - feats["A3_invalid_ratio"]
    if _has(feats, ["A3_correct_ratio","A3_invalid_ratio"]): feats["A3_correct_invalid_gap"] = feats["A3_correct_ratio"] - feats["A3_invalid_ratio"]
    if _has(feats, ["A5_acc_change", "A5_acc_nonchange"]): feats["A5_change_nonchange_gap"] = feats["A5_acc_change"] - feats["A5_acc_nonchange"]
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6
    if _has(feats, ["Year","Month"]): feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]
    for k, acc_col, rt_col in [("B1", "B1_acc_task1", "B1_rt_mean"), ("B2", "B2_acc_task1", "B2_rt_mean"), ("B3", "B3_acc_rate", "B3_rt_mean"), ("B4", "B4_acc_rate", "B4_rt_mean"), ("B5", "B5_acc_rate", "B5_rt_mean")]:
        if _has(feats, [rt_col, acc_col]): feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[rt_col], feats[acc_col], eps)
    for k in ["B1","B2","B3","B4","B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]): feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)
    parts = []
    for k in ["B4","B5"]: 
        if _has(feats, [f"{k}_rt_cv"]): parts.append(0.25 * feats[f"{k}_rt_cv"].fillna(0))
    for k in ["B3","B4","B5"]:
        acc = f"{k}_acc_rate" if k not in ["B1","B2"] else None
        if k in ["B1","B2"]:
            acc = f"{k}_acc_task1"
        if acc in feats: parts.append(0.25 * (1 - feats[acc].fillna(0)))
    for k in ["B1","B2"]:
        tcol = f"{k}_speed_acc_tradeoff"
        if tcol in feats: parts.append(0.25 * feats[tcol].fillna(0))
    if parts: feats["RiskScore_B"] = sum(parts)
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

# =====================================================================================
# MAIN TRAINING LOGIC
# =====================================================================================

DROP_COLS = ["Test_id","Test","PrimaryKey","Age","TestDate"]

def main():
    # ---- Load Data ----
    print("Loading data...")
    BASE_DIR = 'C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data'
    train_meta = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    train_A_raw = pd.read_csv(os.path.join(BASE_DIR, "train", "A.csv"))
    train_B_raw = pd.read_csv(os.path.join(BASE_DIR, "train", "B.csv"))

    # ---- Preprocess and Feature Engineer ----
    print("Processing A...")
    train_A_processed = preprocess_A(train_A_raw)
    train_A_featured = add_features_A(train_A_processed)
    
    print("Processing B...")
    train_B_processed = preprocess_B(train_B_raw)
    train_B_featured = add_features_B(train_B_processed)

    # ---- Merge with Meta and Split ----
    meta_A = train_meta[train_meta["Test"]=="A"].reset_index(drop=True)
    meta_B = train_meta[train_meta["Test"]=="B"].reset_index(drop=True)

    # Align dataframes
    data_A = meta_A.drop(columns=['Test']).merge(train_A_featured, on="Test_id")
    data_B = meta_B.drop(columns=['Test']).merge(train_B_featured, on="Test_id")

    X_A = data_A.drop(columns=DROP_COLS + ['Label'], errors="ignore")
    y_A = data_A["Label"].values
    X_B = data_B.drop(columns=DROP_COLS + ['Label'], errors="ignore")
    y_B = data_B["Label"].values

    X_train_A, X_val_A, y_train_A, y_val_A = train_test_split(X_A, y_A, test_size=0.2, stratify=y_A, random_state=42)
    X_train_B, X_val_B, y_train_B, y_val_B = train_test_split(X_B, y_B, test_size=0.2, stratify=y_B, random_state=42)

    # ---- Train Models ----
    def train_and_eval(X_train, y_train, X_val, y_val, group_label):
        model = lgb.LGBMClassifier(
            objective="binary", metric="auc", n_estimators=3000,
            learning_rate=0.05, n_jobs=-1, random_state=42,
        )
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)]
        )
        val_pred = model.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, val_pred)
        print(f"[{group_label}] Validation AUC: {auc:.4f}")
        return model

    print("Training model for A...")
    model_A = train_and_eval(X_train_A, y_train_A, X_val_A, y_val_A, "A")
    print("Training model for B...")
    model_B = train_and_eval(X_train_B, y_train_B, X_val_B, y_val_B, "B")

    # ---- Save Models ----
    print("Saving models...")
    MODEL_DIR = "C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/model"
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model_A, os.path.join(MODEL_DIR, "lgbm_A.pkl"))
    joblib.dump(model_B, os.path.join(MODEL_DIR, "lgbm_B.pkl"))
    print(f"Models saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
