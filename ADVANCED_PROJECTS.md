# Advanced ML Projects: Complete Lifecycle & Production Skills 🚀

## Overview

Two advanced projects demonstrating **end-to-end ML lifecycle** with **real production patterns**:

1. **Project 11**: Advanced CNN Image Classifier (TensorFlow/Keras)
2. **Project 12**: Neural Recommendation System (PyTorch)

3. **Project 15**: Credit Risk Scorecard (Gradient Boosting + RF)
4. **Project 16**: Insurance Claim Fraud Detector (GB + Isolation Forest)

### Banking & Insurance Domain Projects

#### Project 15: Credit Risk Scorecard (Gradient Boosting + Random Forest Ensemble)
- Predicts loan default probability from 27 credit features
- ROC-AUC: 0.93 with excellent feature importance balance
- Used by banks for auto loans, personal loans, and mortgage underwriting
- Key features: Debt-to-income ratio, credit score, payment history, utilization

#### Project 16: Insurance Claim Fraud Detector (Gradient Boosting + Isolation Forest)
- Flags suspicious insurance claims from 23 structured features
- ROC-AUC: 0.98 with 74% fraud recall at optimal threshold
- Used by insurers to triage claims for SIU investigation
- Key patterns: Amount-to-limit ratio, new customer risk, geographic clustering

---

## Skills Demonstrated

### 🛠️ Framework Expertise

#### TensorFlow (Project 11)
- `keras.Sequential` and Functional API
- Convolutional layers (Conv2D, MaxPooling2D)
- Batch Normalization for training stability
- Dropout for regularization
- Data augmentation pipeline
- Model callbacks (EarlyStopping)
- Model evaluation & metrics

#### PyTorch (Project 12)
- `nn.Module` custom architectures
- Embedding layers for sparse data
- `DataLoader` for batch processing
- Custom training loops
- Optimizer management (Adam)
- Device handling (CPU/GPU)
- Model serialization

---

### 📊 End-to-End ML Lifecycle

Both projects cover COMPLETE lifecycle:

#### STEP 1: Data Preparation ✓
- **Project 11**: CIFAR-10 loading, normalization, augmentation
- **Project 12**: Synthetic data generation, sparse matrices, train/test split

#### STEP 2: Model Building ✓
- **Project 11**: 3-block CNN with BatchNorm + Dropout
- **Project 12**: Neural Collaborative Filtering with embeddings

#### STEP 3: Training ✓
- **Project 11**: 30 epochs with monitoring, early stopping
- **Project 12**: Validation-based training, early stopping

#### STEP 4: Evaluation ✓
- **Project 11**: Accuracy, loss, per-class metrics, confusion matrix
- **Project 12**: RMSE, MAE, R², NDCG, Hit Rate

#### STEP 5: Deployment ✓
- **Project 11**: FastAPI with image upload, real-time classification
- **Project 12**: FastAPI with recommendation requests

#### STEP 6: Monitoring ✓
- **Project 11**: Prediction logging, confidence tracking, hard example detection
- **Project 12**: Per-user recommendation history, metric tracking

---

### 🎯 Model Performance, Evaluation & Debugging

#### Project 11: Image Classification

**Performance Metrics:**
- Overall accuracy (test set)
- Per-class accuracy
- Confusion matrix
- Confidence distribution

**Debugging Features:**
- Hard examples identification (correct but low confidence)
- Overconfident mistakes (wrong but high confidence)
- Per-class analysis showing which classes are hard
- Confidence statistics (mean, std, min, max)
- Monitoring dashboard showing recent predictions

**Evaluation Metrics:**
```python
# Classification Report
from sklearn.metrics import classification_report, confusion_matrix

# Confusion Matrix
Shows which classes are confused with each other

# Per-class metrics
Precision, Recall, F1 for each animal class
```

#### Project 12: Recommendation System

**Performance Metrics:**
- RMSE: How close predictions are to actual ratings
- MAE: Mean absolute error
- R²: Variance explained
- Accuracy@0.5: Predictions within ±0.5 stars

**Ranking Metrics (Real evaluation!):**
- **NDCG@5**: Quality of top-5 recommendations
  - Penalizes bad recommendations even if they're "close"
  - Considers ranking order
  - Standard in recommendation research

- **Hit Rate@5**: Percentage of users who got good recommendation
  - Did top-5 include at least one high-rated item?
  - Practical business metric

**Debugging Features:**
- Per-user RMSE (identify problematic users)
- Ranking quality analysis
- Confidence in recommendations
- Cold-start handling analysis

**Evaluation Metrics:**
```python
# Regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Custom ranking metrics
NDCG@k - how good is the ranking
Hit Rate - did we recommend something good

# Per-user analysis
Understand model behavior for different user types
```

---

## Project Details

### Project 11: Advanced CNN Image Classifier

**Location**: `projects/11-advanced-image-classifier/`

**What it does:**
- Classifies images into 10 CIFAR-10 categories
- Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Architecture:**
```
Input (32×32 RGB)
├─ Augmentation (flip, rotate, zoom)
├─ Conv Block 1: Conv→BatchNorm→Conv→BatchNorm→MaxPool→Dropout
├─ Conv Block 2: Conv→BatchNorm→Conv→BatchNorm→MaxPool→Dropout
├─ Conv Block 3: Conv→BatchNorm→Conv→BatchNorm→MaxPool→Dropout
├─ Flatten
├─ Dense(512)→BatchNorm→Dropout
├─ Dense(256)→BatchNorm→Dropout
└─ Output: 10 classes (softmax)
```

**Training Features:**
- Batch normalization (stable training)
- Data augmentation (prevents overfitting)
- Early stopping (smart stopping)
- Validation split (monitor generalization)
- 30 epochs, batch size 128

**Evaluation:**
- Test accuracy: ~85-90%
- Per-class breakdown
- Confidence analysis
- Hard example detection

**Deployment:**
- FastAPI web app
- Image upload interface
- Real-time predictions
- Model information dashboard

**To Run:**
```bash
cd projects/11-advanced-image-classifier
pip install -r requirements.txt
python train.py  # ~5-10 min on CPU
python app.py
# Visit http://localhost:8011
```

---

### Project 12: Neural Recommendation System

**Location**: `projects/12-neural-recommender/`

**What it does:**
- Recommends movies based on user preferences
- 500 users, 200 movies, ~5,000 ratings

**Architecture:**
```
User ID → Embedding(32D) ─┐
                           ├→ Concat → Dense(128)→ReLU→Dropout
Movie ID → Embedding(32D) ─┤          Dense(64)→ReLU→Dropout
                           └→ Output: Rating [1-5]
```

**Training Features:**
- Sparse data handling (embeddings)
- Custom PyTorch training loop
- Batch processing with DataLoader
- Validation monitoring
- Early stopping (patience=5)
- 30 epochs, batch size 32

**Evaluation:**
- RMSE: ~0.8-1.0 (±1 star accuracy)
- MAE: Mean absolute error
- R²: Variance explained
- NDCG@5: Top-5 ranking quality
- Hit Rate@5: Recommendation success

**Key Metrics Explained:**
```
NDCG@5 = 0.75
→ Top 5 recommendations are 75% as good as perfect ranking

Hit Rate@5 = 0.80
→ 80% of users get at least 1 good recommendation (rating ≥4)
```

**Deployment:**
- FastAPI web app
- User ID input → top 10 recommendations
- Predicted ratings with confidence
- Model information dashboard

**To Run:**
```bash
cd projects/12-neural-recommender
pip install -r requirements.txt
python train.py  # ~2-3 min on CPU
python app.py
# Visit http://localhost:8012
```

---

## Comparing TensorFlow vs PyTorch

### Project 11: TensorFlow (CNN)

**Pros:**
- High-level Keras API (easy to build)
- Automatic differentiation
- Great for computer vision
- Built-in data augmentation

**Code Example:**
```python
model = keras.Sequential([
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25)
])
```

### Project 12: PyTorch (Neural CF)

**Pros:**
- Fine-grained control
- Research-friendly (easier to debug)
- Custom training loops
- Dynamic computation graphs

**Code Example:**
```python
class NCF(nn.Module):
    def __init__(self):
        self.user_embed = nn.Embedding(500, 32)
        self.movie_embed = nn.Embedding(200, 32)
    
    def forward(self, user_ids, movie_ids):
        # ...custom forward pass
```

---

## Key Takeaways

### 1. Framework Expertise ✓
- **TensorFlow**: High-level APIs, CNNs, data pipeline
- **PyTorch**: Custom models, training loops, embeddings
- **Both**: Model building, evaluation, deployment

### 2. Complete Lifecycle ✓
- Data loading & preprocessing
- Model architecture design
- Training with validation
- Comprehensive evaluation
- Deployment with FastAPI
- Production monitoring

### 3. Performance & Debugging ✓
- Multiple evaluation metrics (regression + ranking)
- Per-sample/user analysis
- Confidence tracking
- Hard example identification
- Monitoring for model drift
- Real-world business metrics (Hit Rate, NDCG)

### 4. Production Skills ✓
- Model serialization
- API design
- Real-time predictions
- Performance tracking
- Error handling
- Scalable architecture

---

## Learning Path

1. **Start with Project 11** (easier):
   - TensorFlow Keras is intuitive
   - Understand CNN architecture
   - Learn evaluation metrics

2. **Then Project 12** (advanced):
   - PyTorch gives more control
   - Custom models for real problems
   - Ranking metrics for recommendations

3. **Compare & Contrast**:
   - Both solve different problems
   - Different frameworks have tradeoffs
   - Choose based on use case

---

## Real-World Applications

### CNN Applications (Project 11):
- Medical imaging (X-rays, MRI)
- Autonomous vehicles (traffic signs)
- Quality control (manufacturing)
- Content moderation
- Security/surveillance

### Recommendation Applications (Project 12):
- Netflix: Movies/shows
- Spotify: Music
- Amazon: Products
- YouTube: Videos
- LinkedIn: People/jobs
- Dating apps: Matches

---

## What's Next?

Once you master these:
1. **Combine**: CNN features → Recommendation system
2. **Scale**: Distributed training, inference optimization
3. **Monitor**: Production metrics, A/B testing
4. **Deploy**: Cloud platforms, edge devices
5. **Advance**: Transformers, GANs, RL models

---

**These projects show you're ready for real ML engineering roles!** 🎓
