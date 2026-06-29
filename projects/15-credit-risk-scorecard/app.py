"""Credit Risk Scorecard - Loan Application Risk Assessment API"""
import json
import joblib
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Credit Risk Scorecard")
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')
gb_model = joblib.load('gb_credit_model.pkl')
rf_model = joblib.load('rf_credit_model.pkl')
scaler = joblib.load('credit_scaler.pkl')

with open('credit_metadata.json') as f:
    meta = json.load(f)

CREDIT_BANDS = {
    'excellent':   (740, 850, 'lowest_rate'),
    'good':        (670, 739, 'low_rate'),
    'fair':        (580, 669, 'moderate_rate'),
    'poor':        (500, 579, 'high_rate'),
    'very_poor':   (300, 499, 'denied_or_subprime'),
}
DTI_THRESHOLDS = {
    '<=15%': (0, 0.15), '>15%-25%': (0.15, 0.25), '>25%-36%': (0.25, 0.36),
    '>36%-43%': (0.36, 0.43), '>43%': (0.43, 1.0)
}

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

def get_credit_band(score):
    for band in [('excellent','good','fair','poor','very_poor')]:
        pass
    bands = list(CREDIT_BANDS.items())
    for name, (lo, hi, rate) in CREDIT_BANDS.items():
        if lo <= score <= hi:
            return name, rate
    return 'unknown', 'unknown'

def compute_risk_score(default_prob):
    """Compute custom 1-1000 risk score from default probability."""
    # Higher prob = worse credit = lower score (inverted)
    # Clamp to [150, 850] range for realism
    score = int(850 * (1 - default_prob) + np.random.normal(0, 2))
    return max(300, min(850, score))

@app.post('/assess', response_class=HTMLResponse)
async def assess(request: Request, form: dict = Form(...)):
    try:
        age = int(form['age'])
        income = float(form['income'])
        emp_years = float(form.get('employment_years', 0))
        credit_score_val = int(form['credit_score'])
        total_debt = float(form['total_debt'])
        num_lines = max(1, int(form.get('num_credit_lines', 3)))
        util_pct = min(1.0, max(0.0, float(form.get('credit_utilization', 30)) / 100))
        delinq = int(form.get('delinq_2yrs', 0))
        pub_recs = int(form.get('public_records', 0))
        inquiries = max(0, int(form.get('recent_inquiries', 2)))
        loan_amt = float(form['loan_amount'])
        loan_purpose = form.get('loan_purpose', 'debt_consolidation')
        dti_val = min(0.95, max(0.01, total_debt / (income / 12) if income > 0 else 0.3))
        num_accounts = max(1, int(form.get('num_primary_accounts', 2)))

        debt_to_income = total_debt / income if income > 1 else 999
        income_per_line = income / num_lines if num_lines > 0 else 0
        loan_to_income = loan_amt / income if income > 1 else 999

        features = np.array([[
            age, income, emp_years, credit_score_val, total_debt,
            num_lines, util_pct, delinq, pub_recs, inquiries,
            999 if delinq == 0 else int(np.random.uniform(1, 24)),
            loan_amt, dti_val, total_debt * np.random.uniform(0.05, 0.3) / 12,
            num_accounts, 999 if delinq == 0 else int(np.random.exponential(12)),
            debt_to_income, income_per_line, total_debt / num_lines,
            loan_to_income,
            delinq * 3 + pub_recs * 4 + min(inquiries, 5) + dti_val * 10
        ]])

        # Add one-hot encoded purpose features (defaults to no extra cols for base case)
        purpose_map = {
            'home_improvement': [], 'debt_consolidation': [], 'medical': [],
            'business': [], 'auto': [], 'education': [], 'other': []
        }
        extra = purpose_map.get(loan_purpose, [])
        features_expanded = np.hstack([features, np.zeros((1, len(extra)))])

        scaled = scaler.transform(features_expanded)
        gb_proba = gb_model.predict_proba(scaled)[0, 1]
        rf_proba = rf_model.predict_proba(scaled)[0, 1]
        default_prob = 0.6 * gb_proba + 0.4 * rf_proba

        credit_band, rate_tier = get_credit_band(credit_score_val)
        risk_score = compute_risk_score(default_prob)
        approved = default_prob < 0.5
        dti_bracket = '>43%'
        for bracket, (lo, hi) in DTI_THRESHOLDS.items():
            if lo <= dti_val <= hi:
                dti_bracket = bracket

        recommendation = "Approved - Best Terms" if approved and credit_score_val >= 740 else \
                        ("Approved - Standard Terms" if approved else "Declined / Subprime")

        top_factors = sorted(zip(meta['feature_importances'][:5], ['credit_score', 'dti_ratio',
            'delinq_2yrs', 'public_records', 'credit_utilization']), key=lambda x: x[0]['importance'])

        return templates.TemplateResponse('results.html', {
            'request': request,
            'default_prob': round(default_prob * 100, 1),
            'risk_score': risk_score,
            'approved': approved,
            'credit_band': credit_band,
            'rate_tier': rate_tier,
            'dti_bracket': dti_bracket,
            'recommendation': recommendation,
            'gb_proba': round(gb_proba * 100, 1),
            'rf_proba': round(rf_proba * 100, 1),
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8005)
