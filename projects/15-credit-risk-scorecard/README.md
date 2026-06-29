# Credit Risk Scorecard 📊

## What Does This Do?

This model predicts **whether a loan applicant will default on payments**, just like how banks assess risk before approving credit cards, auto loans, or personal loans.

Think of it as a financial health check: the computer looks at your credit history, debt load, employment stability, and other factors to calculate your risk score.

## How Does It Work?

The model uses an **ensemble of two classifiers** (Gradient Boosting + Random Forest) trained on 8,000 synthetic credit profiles. It evaluates 27 features including:

- **Credit Score**: Your FICO-like score (300-850 range)
- **Debt-to-Income Ratio**: How much of your income goes to debt payments
- **Payment History**: Number of delinquencies, bankruptcies, late payments
- **Credit Utilization**: Percentage of available credit you're using
- **Employment Stability**: Years at current and previous jobs
- **Loan Characteristics**: Amount requested vs. income

## What It Learned

Top risk factors (by importance):
1. **Debt-to-Income Ratio** - The #1 predictor in real banking
2. **Credit Score** - Historical creditworthiness
3. **Credit Utilization** - How "maxed out" your cards are
4. **Payment History** - Past delinquencies and public records

## Key Metrics

- **ROC-AUC: 0.926** - Excellent discrimination between good vs. bad borrowers
- **Training Data**: 8,000 credit profiles (~13.6% default rate)
- **Ensemble Method**: Gradient Boosting (60%) + Random Forest (40%)

## Credit Score Bands

| Band | Score | Interest Rate Tier |
|------|-------|-------------------|
| Excellent | 740-850 | Lowest rate |
| Good | 670-739 | Low rate |
| Fair | 580-669 | Moderate rate |
| Poor | 500-579 | High rate |
| Very Poor | 300-499 | Denied or subprime |

## How to Use It

```bash
cd projects/15-credit-risk-scorecard
pip install -r requirements.txt
python app.py    # Runs on http://localhost:8005
```

Enter loan application details and get a risk assessment.

## Industry Context

This type of model is used daily by:
- **Banks**: Approving personal loans, auto loans, mortgages
- **Credit Card Companies**: Setting credit limits and interest rates
- **FinTech Lenders**: Automated underwriting for online loans
- **Regulators**: Ensuring fair lending practices (fair scores are required)

---

**Simple Truth**: Like a financial report card that tells lenders how risky it would be to lend you money!
