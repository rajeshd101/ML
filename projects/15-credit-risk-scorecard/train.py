import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json

print("=" * 60)
print("CREDIT RISK SCORECARD - Training Pipeline")
print("=" * 60)

# --- Generate realistic credit dataset ---
print("\n[1/4] Generating synthetic credit dataset...")
np.random.seed(42)
N = 8000

def gen_credit_data(n):
    records = []
    for _ in range(n):
        age = int(np.clip(np.random.normal(38, 12), 18, 85))
        income = max(20000, np.random.lognormal(10.8, 0.6))
        income = round(income / 1000) * 1000
        employment_years = max(0, int(np.random.exponential(5)))
        if age < 25:
            employment_years = min(employment_years, age - 18)

        credit_score = min(850, max(300, int(np.random.normal(680, 90))))
        total_debt = round(max(0, income * np.random.uniform(0.2, 3.5)))
        num_credit_lines = max(1, int(np.random.normal(6, 3)))
        credit_utilization = min(1.0, abs(np.random.normal(0.35, 0.2)))
        delinq_2yrs = np.random.choice([0, 1, 2, 3, 4], p=[0.70, 0.18, 0.06, 0.04, 0.02])

        public_records = min(5, int(np.random.exponential(0.5)))
        recent_inquiries = max(0, int(np.random.normal(3, 2)))
        months_since_last_derog = int(np.random.uniform(1, 24)) if delinq_2yrs > 0 else 999

        loan_purposes = ['home_improvement', 'debt_consolidation', 'medical',
                         'business', 'auto', 'education', 'other']
        weights = [0.25, 0.30, 0.10, 0.10, 0.12, 0.08, 0.05]
        loan_purpose = np.random.choice(loan_purposes, p=weights)
        loan_amount = round(max(1000, income * np.random.uniform(0.3, 4.0)))

        dti = min(0.95, max(0.02, total_debt / (income / 12) if income > 0 else 0.5))
        monthly_debt_payment = round(total_debt / 12 * np.random.uniform(0.05, 0.3))

        num_primary_accounts = max(1, int(np.random.normal(2, 1)))
        months_since_last_late = max(0, int(np.random.exponential(18 - delinq_2yrs * 4)))

        records.append({
            'age': age, 'annual_income': income, 'employment_years': employment_years,
            'credit_score': credit_score, 'total_debt': total_debt,
            'num_credit_lines': num_credit_lines, 'credit_utilization': round(credit_utilization, 3),
            'delinq_2yrs': delinq_2yrs, 'public_records': public_records,
            'recent_inquiries': recent_inquiries, 'months_since_last_derog': months_since_last_derog,
            'loan_amount': loan_amount, 'loan_purpose': loan_purpose,
            'dti_ratio': round(dti, 3), 'monthly_debt_payment': monthly_debt_payment,
            'num_primary_accounts': num_primary_accounts,
            'months_since_last_late': months_since_last_late,
        })

    df = pd.DataFrame(records)
    log_odds = (
        -0.009 * df['credit_score'] +
        3 * (df['total_debt'] / np.maximum(df['annual_income'], 1)) +
        0.4 * df['delinq_2yrs'] +
        1.5 * df['credit_utilization'] +
        0.3 * (df['public_records'] > 0).astype(float) +
        0.2 * (df['recent_inquiries'] > 5).astype(float) +
        3.8
    )
    prob_default = 1 / (1 + np.exp(log_odds))
    noise = np.random.normal(0, 0.005, len(df))
    final_prob = np.clip(prob_default + noise, 0.001, 0.999)
    default = (np.random.random(len(df)) < final_prob).astype(int)
    df['loan_approved'] = (default == 0).astype(int)
    return df, default

df, y_true = gen_credit_data(N)
print(f"Dataset shape: {df.shape}")
print(f"Default rate: {y_true.mean():.2%}")

# --- Feature engineering ---
print("\n[2/4] Engineering features...")
df['debt_to_income'] = df['total_debt'] / df['annual_income'].clip(lower=1)
df['income_per_line'] = df['annual_income'] / df['num_credit_lines'].clip(lower=1)
df['debt_per_line'] = df['total_debt'] / df['num_credit_lines'].clip(lower=1)
df['loan_to_income'] = df['loan_amount'] / df['annual_income'].clip(lower=1)
df['credit_risk_index'] = (
    df['delinq_2yrs'] * 3 + df['public_records'] * 4 +
    df['recent_inquiries'].clip(upper=5) + df['dti_ratio'].clip(upper=0.6) * 10
)

purpose_dummies = pd.get_dummies(df['loan_purpose'], prefix='purpose', drop_first=True)
df = pd.concat([df, purpose_dummies], axis=1)

feature_cols = [
    'age', 'annual_income', 'employment_years', 'credit_score',
    'total_debt', 'num_credit_lines', 'credit_utilization',
    'delinq_2yrs', 'public_records', 'recent_inquiries',
    'months_since_last_derog', 'loan_amount', 'dti_ratio',
    'monthly_debt_payment', 'num_primary_accounts',
    'months_since_last_late', 'debt_to_income', 'income_per_line',
    'debt_per_line', 'loan_to_income', 'credit_risk_index'
] + [c for c in df.columns if c.startswith('purpose_')]

X = df[feature_cols].values
y = df['loan_approved'].values
print(f"Total features: {len(feature_cols)}")

# --- Train models ---
print("\n[3/4] Training models...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
gb_model.fit(X_tr, y_train)
gb_proba = gb_model.predict_proba(X_te)[:, 1]

rf_model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
rf_model.fit(X_tr, y_train)
rf_proba = rf_model.predict_proba(X_te)[:, 1]

ens_proba = 0.6 * gb_proba + 0.4 * rf_proba
ens_pred = (ens_proba >= 0.5).astype(int)

print("\n--- Gradient Boosting ---")
print(f"Accuracy: {accuracy_score(y_test, (gb_proba >= 0.5).astype(int)):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, gb_proba):.4f}")
print("\n--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, (rf_proba >= 0.5).astype(int)):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, rf_proba):.4f}")
print("\n--- Ensemble (GB + RF) ---")
print(f"Accuracy: {accuracy_score(y_test, ens_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, ens_proba):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, ens_pred, target_names=['Approved', 'Default']))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, ens_pred)}")

gb_imp = gb_model.feature_importances_
importance_df = pd.DataFrame({'feature': feature_cols, 'importance': gb_imp}).sort_values('importance', ascending=False)
print("\n--- Top 10 Feature Importances ---")
for _, row in importance_df.head(10).iterrows():
    bar = chr(9608) * int(row['importance'] * 50)
    print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

# --- Save artifacts ---
print("\n[4/4] Saving artifacts...")
joblib.dump(gb_model, 'gb_credit_model.pkl')
joblib.dump(rf_model, 'rf_credit_model.pkl')
joblib.dump(scaler, 'credit_scaler.pkl')

metadata = {
    'feature_names': feature_cols,
    'feature_importances': importance_df.to_dict('records'),
    'credit_score_bands': {
        'excellent': (740, 850, 'lowest_rate'),
        'good':     (670, 739, 'low_rate'),
        'fair':     (580, 669, 'moderate_rate'),
        'poor':     (500, 579, 'high_rate'),
        'very_poor':(300, 499, 'denied_or_subprime')
    },
    'dti_thresholds': {
        'excellent': '< 15%', 'good': '< 25%', 'acceptable': '< 36%',
        'risky': '36% - 43%', 'denied': '> 43%'
    },
    'model_type': 'GradientBoosting + RandomForest ensemble',
}
with open('credit_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  - gb_credit_model.pkl")
print("  - rf_credit_model.pkl")
print("  - credit_scaler.pkl")
print("  - credit_metadata.json")
print("\nTraining complete!")
