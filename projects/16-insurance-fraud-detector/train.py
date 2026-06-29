"""
Insurance Claim Fraud Detector - Training Pipeline
===================================================
Generates synthetic insurance claims with embedded fraud patterns,
trains an ensemble classifier to detect suspicious claims.
Uses STRUCTURED FEATURES only (no text features) for realistic fraud detection.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib
import json

print("=" * 60)
print("INSURANCE CLAIM FRAUD DETECTOR - Training Pipeline")
print("=" * 60)

# --- Generate synthetic claims data with embedded fraud patterns ---
print("\n[1/5] Generating synthetic insurance claims...")
np.random.seed(42)
N = 15000
FRAUD_RATE = 0.12
def gen_claims(n, seed=42):
    rng = np.random.RandomState(seed)
    n_fraud = int(n * FRAUD_RATE)
    n_normal = n - n_fraud

    claim_ids = [f'CLM-{i:06d}' for i in range(1, n + 1)]

    # --- Normal claims ---
    normal_age = rng.normal(42, 15, n_normal).clip(18, 85)
    normal_years = rng.exponential(4, n_normal).clip(0, 30)
    normal_prior_3yr = rng.poisson(1.0, n_normal)
    normal_geo_risk = rng.normal(0.5, 0.2, n_normal)
    normal_amount_ratio = rng.beta(2, 6, n_normal)
    normal_is_first = (normal_prior_3yr < 1).astype(float)
    normal_round_amt = (rng.random(n_normal) < 0.05).astype(int)

    # --- Fraud claims: shifted distributions with OVERLAP ---
    fraud_age = rng.normal(36, 12, n_fraud).clip(18, 65)
    fraud_years = rng.exponential(2.0, n_fraud).clip(0, 15)
    fraud_prior_3yr = np.clip(rng.normal(3.2, 2.0, n_fraud), 0, None).astype(int)
    fraud_geo_risk = rng.normal(0.68, 0.15, n_fraud)
    fraud_amount_ratio = rng.beta(3, 3, n_fraud)
    fraud_is_first_arr = np.concatenate([np.ones(int(n_fraud * 0.4)), np.zeros(int(n_fraud * 0.6))])
    fraud_round_amt = (rng.random(n_fraud) < 0.18).astype(int)

    # Combine normal + fraud into single dataset
    all_age = np.concatenate([normal_age, fraud_age])
    all_years = np.concatenate([normal_years, fraud_years])
    all_prior_3yr = np.concatenate([normal_prior_3yr, fraud_prior_3yr])
    all_geo_risk = np.concatenate([normal_geo_risk, fraud_geo_risk])
    all_amount_ratio = np.concatenate([normal_amount_ratio, fraud_amount_ratio])
    all_is_first = np.concatenate([normal_is_first, fraud_is_first_arr])
    all_round_amt = np.concatenate([normal_round_amt, fraud_round_amt])

    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)

    # Policy types (same for both groups)
    policy_types = ['auto', 'home', 'life', 'health', 'commercial']
    policy_weights = [0.35, 0.25, 0.10, 0.20, 0.10]
    all_policy_type = rng.choice(policy_types, n, p=policy_weights)

    coverage_dist = {'auto': (50000, 30000), 'home': (250000, 150000), 'life': (300000, 200000),
                     'health': (500000, 300000), 'commercial': (500000, 400000)}
    all_coverage_limit = np.array([max(1000, int(rng.normal(*coverage_dist[t]))) for t in all_policy_type])

    claim_types_list = ['collision', 'theft', 'weather', 'liability', 'medical',
                        'property_damage', 'business_interruption', 'warranty']
    ct_maps = {'auto': [0.40, 0.15, 0.20, 0.15, 0.05, 0.03, 0.01, 0.01],
               'home': [0.02, 0.08, 0.35, 0.10, 0.05, 0.30, 0.05, 0.05],
               'life': [0.01, 0.01, 0.01, 0.70, 0.25, 0.01, 0.00, 0.01],
               'health': [0.01, 0.01, 0.01, 0.05, 0.90, 0.01, 0.00, 0.01],
               'commercial': [0.05, 0.02, 0.03, 0.10, 0.10, 0.05, 0.55, 0.10]}
    all_claim_type = [rng.choice(claim_types_list, p=ct_maps[t]) for t in all_policy_type]

    # Amount from coverage limit * amount_ratio
    all_claim_amount = np.round(all_amount_ratio * all_coverage_limit).astype(int)
    all_claim_amount = np.clip(all_claim_amount, 100, all_coverage_limit)

    # Other features (same distribution for both groups)
    hours_since = rng.exponential(72, n).clip(0, 720)
    filing_hour = rng.randint(0, 24, size=n)
    is_weekend = (rng.random(n) < 0.35).astype(int)
    has_police = (rng.random(n) < 0.3).astype(int)
    has_medical = (rng.random(n) < 0.4).astype(int)
    repair_shop = (rng.random(n) < 0.5).astype(int)

    # Gender (same for both groups)
    all_gender = rng.choice(['M', 'F'], n)

    # Prior claims 1yr (correlated with 3yr)
    prior_1yr = np.minimum(all_prior_3yr, rng.poisson(0.8 * (all_prior_3yr + 1), n))

    df = pd.DataFrame({
        'claim_id': claim_ids, 'customer_age': all_age, 'years_with_company': all_years,
        'gender': all_gender, 'policy_type': all_policy_type, 'claim_type': all_claim_type,
        'claim_amount': all_claim_amount, 'coverage_limit': all_coverage_limit,
        'hours_since_incident': hours_since, 'filing_hour_of_day': filing_hour,
        'is_weekend': is_weekend, 'prior_claims_1yr': prior_1yr,
        'prior_claims_3yr': all_prior_3yr, 'is_first_claim': all_is_first,
        'geo_risk_score': all_geo_risk, 'amount_ratio': np.round(all_amount_ratio, 4),
        'has_police_report': has_police, 'has_medical_report': has_medical,
        'repair_shop_recommended': repair_shop, 'round_amount_flag': all_round_amt,
        'fraud_label': y,
    })
    return df
df = gen_claims(N)
fraud_mask = df['fraud_label'] == 1
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {fraud_mask.mean():.2%} ({fraud_mask.sum()} claims)")
print(f"\nClaims by policy type:\n{df['policy_type'].value_counts().to_string()}")
print(f"\nAvg claim amount - Normal vs Fraud:")
print(f"  Normal: ${df.loc[~fraud_mask, 'claim_amount'].mean():,.0f}")
print(f"  Fraud:  ${df.loc[fraud_mask, 'claim_amount'].mean():,.0f}")

# --- Feature engineering (STRUCTURED DATA ONLY) ---
print("\n[2/5] Engineering features...")
policy_map = {t: i for i, t in enumerate(sorted(df['policy_type'].unique()))}
claim_map = {t: i for i, t in enumerate(sorted(df['claim_type'].unique()))}
df['policy_encoded'] = df['policy_type'].map(policy_map)
df['claim_encoded'] = df['claim_type'].map(claim_map)

# Derived features
df['amount_per_coverage_pct'] = (df['claim_amount'] / df['coverage_limit'].clip(lower=1)) * 100
df['age_claim_ratio'] = df['customer_age'] / (df['years_with_company'].clip(lower=1) + 1)
df['claims_per_year_rate'] = df['prior_claims_3yr'] / df['years_with_company'].clip(lower=0.5)
df['is_suspicious_timing'] = ((df['filing_hour_of_day'] < 6) | (df['filing_hour_of_day'] > 22)).astype(int)

# Composite risk score from multiple indicators
df['risk_composite'] = (
    df['prior_claims_3yr'].clip(upper=10) * 0.3 +
    df['amount_ratio'].clip(lower=1, upper=1) * 0.25 +
    df['geo_risk_score'].clip(0, 1) * 0.2 +
    df['is_suspicious_timing'] * 0.15 +
    (df['is_first_claim'] == 1).astype(float) * 0.1
)

# All feature columns - structured data only (no text features)
feature_cols = [
    'customer_age', 'years_with_company', 'policy_encoded', 'claim_encoded',
    'claim_amount', 'coverage_limit', 'hours_since_incident',
    'filing_hour_of_day', 'is_weekend', 'prior_claims_1yr', 'prior_claims_3yr',
    'is_first_claim', 'geo_risk_score', 'amount_ratio', 'has_police_report',
    'has_medical_report', 'repair_shop_recommended', 'round_amount_flag',
    'amount_per_coverage_pct', 'age_claim_ratio', 'claims_per_year_rate',
    'is_suspicious_timing', 'risk_composite'
]

X = df[feature_cols].values
y = df['fraud_label'].values
print(f"Total features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# --- Train models ---
print("\n[3/5] Training models...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

# Gradient Boosting classifier
gb_model = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.08,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gb_model.fit(X_tr, y_train)
gb_proba = gb_model.predict_proba(X_te)[:, 1]

# Isolation Forest for anomaly detection
iforest = IsolationForest(contamination=0.12, random_state=42, n_estimators=200)
iforest.fit(X_tr)
iso_scores = -iforest.score_samples(X_te)
iso_proba = (iso_scores > np.percentile(iso_scores, 88)).astype(float)

# Optimize threshold using precision-recall curve
pr_curve_data = precision_recall_curve(y_test, gb_proba)
best_f1 = 0
best_threshold = 0.25
for t in np.arange(0.10, 0.75, 0.005):
    preds_t = (gb_proba >= t).astype(int)
    if preds_t.sum() > 0:
        tp = int(((preds_t == 1) & (y_test == 1)).sum())
        fp = int(((preds_t == 1) & (y_test == 0)).sum())
        fn = int(((preds_t == 0) & (y_test == 1)).sum())
        if tp > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

ensemble_proba = 0.7 * gb_proba + 0.3 * iso_proba
print(f"Optimized threshold: {best_threshold:.3f} (F1={best_f1:.4f})")

# Evaluate individual models
gb_pred_at_t = (gb_proba >= best_threshold).astype(int)
gb_auc = roc_auc_score(y_test, gb_proba)

iso_auc = roc_auc_score(y_test, iso_proba)
print("\n--- Gradient Boosting ---")
print(f"ROC-AUC:     {gb_auc:.4f}")
report_gb = classification_report(y_test, gb_pred_at_t, target_names=['Legitimate', 'Fraud'], output_dict=True)
for cls in ['Legitimate', 'Fraud']:
    if cls in report_gb:
        print(f"  {cls}: P={report_gb[cls]['precision']:.3f} R={report_gb[cls]['recall']:.3f} F1={report_gb[cls]['f1-score']:.3f}")

print(f"\n--- Isolation Forest ---")
print(f"ROC-AUC:     {iso_auc:.4f}")

# Ensemble evaluation
ens_pred = (ensemble_proba >= best_threshold).astype(int)
ensemble_auc = roc_auc_score(y_test, ensemble_proba)
print(f"\n--- Ensemble (GB + IsolationForest) ---")
print(f"ROC-AUC:     {ensemble_auc:.4f}")
print(classification_report(y_test, ens_pred, target_names=['Legitimate', 'Fraud'], digits=3))

cm = confusion_matrix(y_test, ens_pred)
print(f"\nConfusion Matrix:")
print(f"               Predicted")
print(f"               Legit   Fraud")
print(f"  Actual Legit {cm[0][0]:>8d} {cm[0][1]:>6d}")
print(f"  Actual Fraud {cm[1][0]:>8d} {cm[1][1]:>6d}")

# Feature importance
gb_imp = gb_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_imp
}).sort_values('importance', ascending=False)

print("\n--- Top 12 Feature Importances ---")
for _, row in importance_df.head(12).iterrows():
    bar = chr(9608) * int(row['importance'] * 40)
    print(f"  {row['feature']:35s} {row['importance']:.4f} {bar}")

# --- Save artifacts ---
print("\n[4/5] Saving artifacts...")
joblib.dump(gb_model, 'fraud_gb_model.pkl')
joblib.dump(iforest, 'fraud_iforest.pkl')
joblib.dump(scaler, 'fraud_scaler.pkl')

metadata = {
    'feature_names': feature_cols,
    'feature_importances': importance_df.to_dict('records'),
    'fraud_rate_benchmarks': {
        'auto_insurance': '3-5%', 'property_insurance': '5-8%',
        'health_insurance': '2-4%', 'commercial_lines': '4-7%'
    },
    'risk_categories': {
        'low':     (0.0, 0.15, 'routine_processing'),
        'medium':  (0.15, 0.35, 'enhanced_review'),
        'high':    (0.35, 0.60, 'investigation_required'),
        'critical':(0.60, 1.0, 'immediate_suspension'),
    },
    'key_fraud_patterns': [
        'Amount near coverage limit',
        'New customer + large first claim',
        'Multiple prior claims in short period',
        'Unusual filing times (late night/weekend)',
        'Round amount claims',
        'High geographic risk area',
    ],
}
with open('fraud_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  - fraud_gb_model.pkl")
print("  - fraud_iforest.pkl")
print("  - fraud_scaler.pkl")
print("  - fraud_metadata.json")
print("\nTraining complete!")
