"""Insurance Claim Fraud Detector - Real-time claim assessment API"""
import json
import joblib
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Insurance Claim Fraud Detector")
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')
gb_model = joblib.load('fraud_gb_model.pkl')
iforest = joblib.load('fraud_iforest.pkl')
scaler = joblib.load('fraud_scaler.pkl')

with open('fraud_metadata.json') as f:
    meta = json.load(f)

RISK_CATS = {
    'low':     (0.0, 0.15, '\U0001F49B Routine Processing', '#2d6a4f'),
    'medium':  (0.15, 0.35, '\U0001F4C1 Enhanced Review', '#e09f3e'),
    'high':    (0.35, 0.60, '\U0001F6A8 Investigation Required', '#e63946'),
    'critical':(0.60, 1.0, '\U0001F6A8 Immediate Suspension', '#9b2226'),
}

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

def compute_fraud_features(form):
    """Compute fraud detection features from claim form inputs."""
    age = float(form.get('customer_age', 42))
    years_company = max(0.1, float(form.get('years_with_company', 5)))
    policy_type_val = form.get('policy_type', 'auto')
    claim_type_val = form.get('claim_type', 'collision')

    claim_amount_val = max(100, float(form.get('claim_amount', 10000)))
    coverage_limit_val = max(1000, float(form.get('coverage_limit', 50000)))
    hours_since = max(0, float(form.get('hours_since_incident', 24)))
    filing_hour = int(form.get('filing_hour', 14))
    is_weekend_val = int(form.get('is_weekend', '0'))
    prior_1yr = max(0, int(form.get('prior_claims_1yr', 0)))
    prior_3yr = max(prior_1yr, int(form.get('prior_claims_3yr', prior_1yr)))
    geo_risk_val = min(1.0, max(0, float(form.get('geo_risk_score', 0.5))))

    has_police = int(form.get('has_police_report', '0'))
    has_medical = int(form.get('has_medical_report', '0'))
    repair_shop = int(form.get('repair_shop_recommended', '0'))
    is_round_amt = int(claim_amount_val % 1000 < 10) if claim_amount_val > 100 else 0

    amt_ratio = claim_amount_val / max(1, coverage_limit_val)
    prior_1yr_norm = min(prior_1yr, 5)
    prior_3yr_norm = min(prior_3yr, 10)
    is_first_claim = int(prior_1yr == 0 and years_company < 0.5)

    suspicious_timing = int((filing_hour < 6 or filing_hour > 22) or is_weekend_val == 1)

    risk_composite = (
        prior_3yr_norm * 0.3 + amt_ratio * 0.25 +
        geo_risk_val * 0.2 + suspicious_timing * 0.15 + is_first_claim * 0.1
    )

    # Text fraud signal
    text = form.get('claim_description', '').lower()
    text_signal = 0
    if any(w in text for w in ['suddenly', 'unexplained', 'urgently', 'immediately',
                                 'missing items', 'no witnesses']):
        text_signal += 2
    if any(w in text for w in ['please expedite', 'before someone finds out', 'remember now']):
        text_signal += 1

    desc_length = len(form.get('claim_description', ''))

    # Encode categoricals
    policy_types_ordered = sorted(meta['feature_names'])  # placeholder - use fixed mapping
    p_map = {'auto': 0, 'home': 1, 'life': 2, 'health': 3, 'commercial': 4}
    c_map = {'collision': 0, 'theft': 1, 'weather': 2, 'liability': 3,
             'medical': 4, 'property_damage': 5, 'business_interruption': 6, 'warranty': 7}

    return np.array([[
        age, years_company, p_map.get(policy_type_val, 0), c_map.get(claim_type_val, 0),
        claim_amount_val, coverage_limit_val, hours_since, filing_hour,
        is_weekend_val, prior_1yr_norm, prior_3yr_norm,
        is_first_claim, geo_risk_val, amt_ratio, has_police,
        has_medical, repair_shop, is_round_amt,
        amt_ratio * 100, age / (years_company + 1),
        prior_3yr / max(years_company, 0.5),
        suspicious_timing, risk_composite, text_signal > 0,
        int(any(w in text for w in ['unexplained', 'happened last night'])),
        desc_length, min(text_signal, 5)
    ]])

@app.post('/assess', response_class=HTMLResponse)
async def assess(request: Request, form: dict = Form(...)):
    try:
        features = compute_fraud_features(form)
        scaled = scaler.transform(features)

        gb_proba_pred = gb_model.predict_proba(scaled)[0, 1]
        iso_score = -iforest.score_samples(scaled)
        iso_proba_val = float(iso_score > np.percentile([-s for s in iforest.score_samples(scaler.transform(np.ones((1, len(meta['feature_names']))) * 10)], 92))

        fraud_prob = 0.7 * gb_proba_pred + 0.3 * iso_proba_val
        risk_level = 'low'
        for level, (lo, hi, label, color) in RISK_CATS.items():
            if lo <= fraud_prob < hi:
                risk_level = level
                break

        top_factors = []
        importance = meta['feature_importances'][:10]
        for feat_info in importance:
            fname = feat_info['feature']
            fval = features[0][list(meta['feature_names']).index(fname)] if fname in meta['feature_names'] else 0
            top_factors.append((fname, round(fval, 4), round(feat_info['importance'], 4)))

        return templates.TemplateResponse('results.html', {
            'request': request,
            'fraud_prob': round(fraud_prob * 100, 1),
            'risk_level': risk_level,
            'risk_label': RISK_CATS[risk_level][2],
            'risk_color': RISK_CATS[risk_level][3],
            'gb_fraud_score': round(gb_proba_pred * 100, 1),
            'iso_anomaly': round(iso_proba_val * 100, 1),
            'top_factors': top_factors,
            'claim_id': form.get('claim_id', 'CLM-XXXXXX'),
        })
    except Exception as e:
        import traceback
        return JSONResponse({'error': str(e), 'traceback': traceback.format_exc()}, status_code=400)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8006)
