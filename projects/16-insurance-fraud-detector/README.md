# Insurance Claim Fraud Detector 🔍

## What Does This Do?

This model flags **suspicious insurance claims** that might be fraudulent, like how security cameras and guards work at a store to catch shoplifters.

Insurance companies process millions of claims per year, and an estimated **5-10% are fraudulent**, costing billions in losses. This model helps adjusters prioritize which claims to investigate first.

## How Does It Work?

The system uses an **ensemble of Gradient Boosting + Isolation Forest** trained on 15,000 synthetic insurance claims. It evaluates 23 structured features:

### Key Fraud Patterns Detected:
1. **Amount near coverage limit** (40% weight) - Claims asking for nearly the max policy pays out
2. **New customer with large first claim** (10% weight) - Someone who just bought insurance immediately filing a big claim
3. **High prior claims history** (5% weight) - Multiple recent claims suggest pattern abuse
4. **Geographic risk clustering** (9% weight) - Claims from high-fraud-rate areas
5. **Round-dollar amounts** (2% weight) - Suspiciously exact numbers ($10,000 vs $9,847)

## What It Learned

Top risk factors (by importance):
1. **Composite Risk Score** - Combined signal from multiple indicators
2. **Amount/Coverage Ratio** - How close the claim is to the policy limit
3. **Claim Amount vs Coverage** - Absolute amount relative to what's available
4. **First Claim Flag** - New customers filing large claims are higher risk
5. **Geographic Risk Score** - Area-based fraud prevalence data

## Key Metrics

- **ROC-AUC: 0.976** - Excellent at distinguishing legitimate vs. fraudulent claims
- **Fraud Detection Recall: 74%** - Catches 74% of actual fraud at optimal threshold
- **Precision: 86%** - Of the flagged claims, 86% are actually fraudulent
- **Training Data**: 15,000 claims (12% fraud rate)

## Risk Categories

| Score | Level | Action |
|-------|-------|--------|
| 0-15% | Low 🟢 | Routine Processing |
| 15-35% | Medium 🟡 | Enhanced Review |
| 35-60% | High 🔴 | Investigation Required |
| 60-100% | Critical 🚨 | Immediate Suspension |

## How to Use It

```bash
cd projects/16-insurance-fraud-detector
pip install -r requirements.txt
python app.py    # Runs on http://localhost:8006
```

Enter claim details and get a fraud risk assessment.

## Industry Context

This type of system is used by:
- **All major insurers**: Progressive, State Farm, GEICO, Allstate
- **Claims departments**: Triage claims for investigation priority
- **Special Investigation Units (SIUs)**: Dedicated fraud teams
- **Insurance regulators**: Monitoring fraud rates across carriers
- **Reinsurance companies**: Assessing portfolio-level fraud risk

**Real-world impact**: A single insurer using automated fraud detection can save $10M-$50M annually.

---

**Simple Truth**: Like a financial detective that spots patterns humans might miss to catch fraudulent claims!
