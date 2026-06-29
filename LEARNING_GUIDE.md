# Step-by-Step Learning Guide 📚

## Table of Contents
1. [Getting Started](#getting-started)
2. [Learning Path](#learning-path)
3. [Project 11: CNN Deep Dive](#project-11-cnn-deep-dive)
4. [Project 12: PyTorch Deep Dive](#project-12-pytorch-deep-dive)
5. [Project 13: scikit-learn Deep Dive](#project-13-scikit-learn-deep-dive)
6. [Hands-On Exercises](#hands-on-exercises)
7. [Common Mistakes & Solutions](#common-mistakes--solutions)
8. [Interview Prep](#interview-prep)

---

## Getting Started

### Prerequisites
- Python 3.8+ installed
- Basic understanding of ML concepts (optional but helpful)
- Git installed
- ~10 GB disk space (for datasets and models)

### Setup (5 minutes)

**Step 1: Navigate to project**
```bash
cd /Users/rjd/Work/ML
```

**Step 2: Check project structure**
```bash
ls -la projects/1{1,2,3}-*
```

You should see:
```
11-advanced-image-classifier/
12-neural-recommender/
13-automated-ml-pipeline/
```

**Step 3: Read the overview**
```bash
cat ADVANCED_PROJECTS.md
cat ML_FRAMEWORKS_COMPARISON.md
```

---

## Learning Path

### Timeline Overview
```
Day 1: Understand Project 13 (scikit-learn)
Day 2: Understand Project 11 (TensorFlow)
Day 3: Understand Project 12 (PyTorch)
Day 4: Compare & practice
Day 5: Interview prep
```

### Why This Order?

**Project 13 First** (scikit-learn)
- Simplest framework (classical ML)
- Easier to understand concepts without deep learning complexity
- ~2-3 minutes training time
- Perfect foundation

**Project 11 Second** (TensorFlow)
- Next complexity level
- High-level Keras API (easier to learn than PyTorch)
- Computer vision is intuitive
- ~5-10 minutes training time

**Project 12 Last** (PyTorch)
- Most control and complexity
- Custom implementations
- Build on TensorFlow knowledge
- ~2-3 minutes training time

---

# Project 11: CNN Deep Dive

## What This Project Teaches 🎓

**Frameworks**: TensorFlow/Keras
**Algorithms**: Convolutional Neural Networks (CNN)
**Data**: Image classification (CIFAR-10)
**Lifecycle**: Complete end-to-end with monitoring

## Step 1: Understand the Problem (10 minutes)

### CIFAR-10 Dataset
- 60,000 images (32×32 pixels)
- 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Real-world images (not perfectly cleaned)
- Standard ML benchmark

**Why images?**
- Visual patterns are different from tabular data
- Convolutional layers detect local features
- Progressive feature extraction (low-level → high-level)

### The Goal
Classify any image into one of 10 categories with >85% accuracy

**Read**: `projects/11-advanced-image-classifier/README.md`

## Step 2: Understand CNN Architecture (15 minutes)

### What's a CNN?

**Concept**: "Sliding window that learns features"

```
Input Image (32×32×3)
    ↓
[Conv 3×3] → Extract small patterns
    ↓
[Conv 3×3] → Combine patterns
    ↓
[MaxPool] → Keep important features
    ↓
[Conv 3×3] → Extract larger patterns
    ↓
[Dense] → Classification
    ↓
Output (10 classes)
```

### Key Components

**1. Convolutional Layer**
```
What: Applies filters to detect patterns
Why: Learns edge, texture, shape features
Parameters: kernel size (3×3), number of filters (64, 128, 256)
```

**2. Batch Normalization**
```
What: Normalizes layer inputs
Why: Stabilizes training, allows higher learning rates
Effect: Faster convergence, better generalization
```

**3. Dropout**
```
What: Randomly disables neurons during training (25%, 50%)
Why: Prevents overfitting (co-adaptation)
Effect: Model learns diverse features
```

**4. MaxPooling**
```
What: Takes maximum value in 2×2 window
Why: Reduces spatial dimensions, keeps important features
Effect: Faster training, translation invariance
```

### Project 11 Architecture
```
Input: 32×32 RGB

Block 1:
  Conv(64) → BatchNorm → Conv(64) → BatchNorm
  MaxPool → Dropout(0.25)

Block 2:
  Conv(128) → BatchNorm → Conv(128) → BatchNorm
  MaxPool → Dropout(0.25)

Block 3:
  Conv(256) → BatchNorm → Conv(256) → BatchNorm
  MaxPool → Dropout(0.25)

Dense Section:
  Flatten → Dense(512) → BatchNorm → Dropout(0.5)
          → Dense(256) → BatchNorm → Dropout(0.5)
          → Dense(10 output classes)

Total Parameters: ~1.2 million
```

**Visualization**: Each block gets progressively deeper features

## Step 3: Examine the Training Code (15 minutes)

**File**: `projects/11-advanced-image-classifier/train.py`

### Data Preparation Section
```python
# Line 32-47: Loading CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Line 50-53: Normalization (critical!)
x_train = x_train.astype('float32') / 255.0  # Pixel values [0,1]
x_test = x_test.astype('float32') / 255.0

# Line 56-60: Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

**Why normalize?** Gradients work better on [0,1] range
**Why augment?** Creates variation without more data (flip horizontal, rotate slightly)

### Model Building Section
```python
# Line 68-115: Sequential blocks
# Each block: Conv → BatchNorm → Conv → BatchNorm → MaxPool → Dropout
# Notice: Filters increase (64 → 128 → 256)
# Notice: Dropout increases (0.25 → 0.25 → 0.25 for conv, 0.5 for dense)
```

**Question**: Why increase dropout in dense layer?
**Answer**: Dense layers overfit more easily (every neuron sees all inputs)

### Training Section
```python
# Line 130-150: Training loop
history = model.fit(
    x_train, y_train_encoded,
    batch_size=128,
    epochs=30,
    validation_split=0.1,  # Hold out 10% for validation
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5  # Stop if val_loss doesn't improve for 5 epochs
        ),
    ]
)
```

**Key insight**: Early stopping prevents overfitting (when model memorizes training data)

### Evaluation Section
```python
# Line 165-200: Detailed metrics
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get class with highest probability

# Per-class metrics
for i, class_name in enumerate(class_names):
    class_accuracy = accuracy for animals in this class
```

**Why per-class?** Some classes are harder (is it a cat or dog?)

## Step 4: Run the Project (5-10 minutes)

**Step 1**: Install dependencies
```bash
cd projects/11-advanced-image-classifier
pip install -r requirements.txt
```

**Step 2**: Train the model
```bash
python train.py
```

**Expected output**:
```
[1/4] DATA PREPARATION...
  ✓ Training samples: 50000
  ✓ Test samples: 10000

[2/4] MODEL BUILDING...
  ✓ Model created with 1,234,567 parameters

[3/4] TRAINING...
  Epoch 1/30: Train Loss=2.1234, Val Loss=1.8934
  Epoch 2/30: Train Loss=1.8934, Val Loss=1.6234
  ...
  Early stopping at epoch 20

[4/4] EVALUATION...
  Test Accuracy: 87.23%
  Per-Class Metrics:
    airplane: 82.1%
    automobile: 89.3%
    ...

✅ Model saved: cifar10_model.h5
✅ Evaluation metrics saved: evaluation_metrics.json
```

**Time**: ~5-10 minutes on CPU (~1 min on GPU)

**Step 3**: Run the web app
```bash
python app.py
# Visit http://localhost:8011
```

**Try it**: Upload an image and see predictions!

## Step 5: Understand the Evaluation (15 minutes)

### Accuracy Metrics
```
Overall Accuracy = (Correct Predictions) / (Total Predictions)
Example: 8720 correct out of 10000 = 87.2%

Per-Class Accuracy = Correct for that class / Total of that class
Example: Airplane accuracy = 82.1% (might be confused with bird)
```

### Confusion Matrix
```
             Predicted
             airplane  automobile  bird  ...
Actual
airplane     821       10         5     ...
automobile   8         893        2     ...
bird         12        3          845   ...
...
```

**What it shows**:
- Diagonal = correct predictions
- Off-diagonal = mistakes
- Example: 10 airplanes misclassified as automobile

### Debugging Insights

**Hard Examples** (Correct but low confidence)
```python
correct_mask = y_test == y_pred_classes
confidence = np.max(y_pred, axis=1)
hard_correct = (correct_mask) & (confidence < 0.7)
```

**Why?** Model is uncertain even when right
**Action**: Might need more training or these images are ambiguous

**Overconfident Mistakes** (Wrong but high confidence)
```python
wrong_mask = y_test != y_pred_classes
overconfident = (wrong_mask) & (confidence > 0.9)
```

**Why?** Model makes wrong decision strongly
**Action**: Potential bug in training or mislabeled data

## Step 6: Key Learnings Checklist ✓

After completing Project 11, you should understand:

- [ ] What are CNNs and how they work
- [ ] Batch normalization purpose
- [ ] Dropout for regularization
- [ ] Data augmentation benefits
- [ ] Early stopping strategy
- [ ] Accuracy metrics
- [ ] Confusion matrix interpretation
- [ ] Per-class performance analysis
- [ ] How to debug model predictions
- [ ] Confidence vs correctness trade-off
- [ ] TensorFlow/Keras syntax
- [ ] Model serialization and loading
- [ ] FastAPI deployment basics

---

# Project 12: PyTorch Deep Dive

## What This Project Teaches 🎓

**Frameworks**: PyTorch
**Algorithms**: Neural Collaborative Filtering (embeddings)
**Data**: User-movie ratings (sparse)
**Lifecycle**: Custom training loop with advanced metrics

## Step 1: Understand Recommendation Problem (10 minutes)

### The Problem
- 500 users, 200 movies, 5,000 ratings
- User rates movies 1-5 stars
- **Task**: Predict ratings + recommend top movies

**Real-world examples**:
- Netflix: What movies would you like?
- Spotify: What songs would you like?
- Amazon: What products would you like?

### The Challenge

**Sparsity Problem**:
```
500 users × 200 movies = 100,000 possible ratings
We only have 5,000 ratings = 5% density!

Most user-movie pairs are unknown.
Can't use regular ML (not enough data per cell).
```

**Solution**: Embeddings!
```
User ID → 32-dim vector (representing their taste)
Movie ID → 32-dim vector (representing movie style)
Similarity = dot product of embeddings
```

**Read**: `projects/12-neural-recommender/README.md`

## Step 2: Understand Neural Collaborative Filtering (20 minutes)

### Traditional Approach (Matrix Factorization)
```
User Matrix (500×32) × Movie Matrix (200×32) = Predicted Ratings (500×200)
Issue: Linear only, limited expressiveness
```

### Neural Approach (Project 12)
```
User ID → Embedding(32) ─┐
                         ├→ Concat → Dense(128) → Dense(64) → Rating
Movie ID → Embedding(32) ─┘
```

**Why neural?** Non-linear combinations capture complex patterns

### Embeddings Explained

**Concept**: Represent categories as vectors

```
Movie Embeddings (learned by model):
  Sci-Fi Movie 1 → [0.8, 0.1, 0.2, -0.5, ...]  (high on "action", low on "romance")
  Romantic Comedy  → [0.1, 0.7, 0.3,  0.2, ...]  (high on "romance")
  
User Embeddings (learned by model):
  Action Fan      → [0.9, 0.05, 0.1, -0.4, ...] (likes action, not romance)
  Romance Fan     → [0.1, 0.85, 0.2,  0.3, ...] (likes romance)

Prediction = Neural Network(concat(user_embed, movie_embed))
```

**Similarity**:
- Action Fan + Sci-Fi Movie → HIGH predicted rating
- Action Fan + Romance Movie → LOW predicted rating

## Step 3: Examine PyTorch Code (20 minutes)

**File**: `projects/12-neural-recommender/train.py`

### Data Generation Section
```python
# Line 32-60: Create synthetic data
# Real data: MovieLens dataset
# Project: Synthetic for demo
n_users = 500
n_movies = 200
density = 0.1  # 10% of all pairs have ratings

# Create ratings (5,000 total)
user_ids, movie_ids, ratings = []
```

### Model Class Definition
```python
# Line 76-105: Define the model
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32):
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        x = torch.cat([user_embeds, movie_embeds], dim=1)  # Concatenate
        return self.dense_layers(x).squeeze()
```

**Key differences from TensorFlow**:
- Manual forward() method (custom logic)
- More control over computation
- Explicit parameter initialization
- Clearer data flow

### Training Loop Section
```python
# Line 150-180: Manual training loop (PyTorch style)
for epoch in range(30):
    model.train()  # Set to training mode
    for user_ids, movie_ids, ratings in train_loader:
        optimizer.zero_grad()  # Clear old gradients
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, ratings)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
    
    # Validation
    model.eval()  # Set to eval mode (disable dropout)
    with torch.no_grad():  # Don't compute gradients
        val_loss = compute_validation_loss()
```

**PyTorch vs TensorFlow**:
- TensorFlow: `model.fit()` handles loop automatically
- PyTorch: You write the loop explicitly

**Benefit of PyTorch**: Control! Can do custom things mid-training.

### Evaluation Section
```python
# Line 200+: Advanced metrics
from sklearn.metrics import mean_squared_error

# Regression metrics
rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

# Ranking metrics (unique to recommendations!)
def calculate_ranking_metrics(predictions, ratings, k=5):
    # NDCG@5: Quality of top-5 recommendations
    # Hit Rate@5: Did top-5 include a good movie?
```

**Why ranking metrics?** In recommendations, order matters!
- Showing best movies first is better than worst first
- NDCG penalizes bad movies in top positions

## Step 4: Understanding Ranking Metrics (15 minutes)

### NDCG (Normalized Discounted Cumulative Gain)

**Scenario**: User rated 10 movies. We rank predictions for top-5.

**Actual ratings** (sorted best to worst):
```
Movie A: 5 stars
Movie B: 5 stars
Movie C: 4 stars
Movie D: 2 stars
Movie E: 1 star
```

**Our predictions** (ranked 1-5):
```
Rank 1: Movie A (actual: 5) ✓ Good!
Rank 2: Movie D (actual: 2) ✗ Bad!
Rank 3: Movie B (actual: 5) ✓ Good!
Rank 4: Movie C (actual: 4) ✓ Good!
Rank 5: Movie E (actual: 1) ✗ Bad!
```

**DCG Calculation**:
```
Position 1: 1.0 / log2(2)   = 0.5  (got a 5-star movie, best position)
Position 2: 0.0 / log2(3)   = 0.0  (got a 2-star movie, wasted position)
Position 3: 1.0 / log2(4)   = 0.25 (got a 5-star movie, worse position)
Position 4: 1.0 / log2(5)   = 0.22 (got a 4-star movie, worse position)
Position 5: 0.0 / log2(6)   = 0.0  (got a 1-star movie, wasted position)
DCG = 0.5 + 0 + 0.25 + 0.22 + 0 = 0.97
```

**IDCG** (best possible):
```
Rank 1: 5-star movie → 0.5
Rank 2: 5-star movie → 0.25
Rank 3: 4-star movie → 0.22
Rank 4: ...
IDCG = max possible DCG
```

**NDCG = DCG / IDCG = 0.97 / 1.5 = 0.65**
- 0.0 = terrible ranking
- 1.0 = perfect ranking
- 0.65 = decent ranking

### Hit Rate

**Simpler metric**:
```
For each user:
  Get top-5 recommendations
  Check if ANY movie has rating ≥ 4 (good)
  
If yes → Hit
If no → Miss

Hit Rate = (# users with hits) / (# total users)

Example: 80% Hit Rate = 80% of users got at least 1 good recommendation
```

**Interpretation**:
- 50% Hit Rate = risky (half of users unhappy)
- 80% Hit Rate = acceptable
- 95% Hit Rate = excellent

## Step 5: Run the Project (5 minutes)

```bash
cd projects/12-neural-recommender
pip install -r requirements.txt
python train.py
```

**Expected output**:
```
[1/4] DATA PREPARATION...
  ✓ Generated 5000 ratings

[2/4] MODEL BUILDING...
  ✓ Model created: 201,728 parameters

[3/4] TRAINING...
  Epoch 1: Train Loss=1.8234, Val Loss=1.6234
  Epoch 2: Train Loss=1.6234, Val Loss=1.4567
  ...
  Early stopping at epoch 15

[4/4] EVALUATION...
  Regression Metrics:
    RMSE: 0.8234 (within ±0.8 stars)
    MAE: 0.6567 (average error)
    R²: 0.7234 (variance explained)
  
  Ranking Metrics:
    NDCG@5: 0.7456 (good ranking quality)
    Hit Rate@5: 0.8123 (81% got good recommendation)
```

**Time**: ~2-3 minutes

```bash
python app.py
# Visit http://localhost:8012
# Try User ID 0-499
```

## Step 6: Key Learnings Checklist ✓

- [ ] Collaborative filtering concept
- [ ] Sparsity problem in recommendations
- [ ] Embedding layers and their purpose
- [ ] Neural collaborative filtering architecture
- [ ] PyTorch model definition (nn.Module)
- [ ] Manual training loop
- [ ] Regression metrics (RMSE, MAE, R²)
- [ ] Ranking metrics (NDCG, Hit Rate)
- [ ] Why order matters in recommendations
- [ ] torch.nn, torch.optim usage
- [ ] DataLoader for batching
- [ ] Model evaluation modes (train vs eval)
- [ ] Gradient computation and optimization

---

# Project 13: scikit-learn Deep Dive

## What This Project Teaches 🎓

**Frameworks**: scikit-learn
**Algorithms**: 5 classical ML algorithms
**Data**: Tabular medical data (breast cancer)
**Lifecycle**: Hyperparameter tuning, model comparison, feature importance

## Step 1: Understand the Classification Problem (10 minutes)

### Medical Diagnosis Task
- Dataset: Breast Cancer Diagnostic Data
- 569 patients, 30 clinical measurements
- **Goal**: Predict if tumor is malignant or benign (binary classification)

**Real-world impact**:
- Correct diagnosis can save lives
- Model helps doctors make faster decisions
- Need high precision (avoid false positives - unnecessary surgery)
- Need high recall (avoid false negatives - miss cancer)

**Read**: `projects/13-automated-ml-pipeline/README.md`

## Step 2: Understand the ML Algorithms (30 minutes)

### Algorithm 1: Logistic Regression

**Concept**: "Probability-based classification"

```
Features → Linear combination → Sigmoid function → Probability [0,1]

If probability > 0.5 → Malignant
If probability ≤ 0.5 → Benign
```

**Pros**:
- Interpretable (coefficients = feature importance)
- Fast training
- Probabilistic output

**Cons**:
- Only linear boundaries
- Won't work for complex data

**Hyperparameters** (tuned in grid search):
```python
C: [0.001, 0.1, 1, 10]  # Regularization strength
   # Low C = more regularization (simpler model)
   # High C = less regularization (complex model)

penalty: ['l2']  # Type of regularization (L2 = weight penalty)
solver: ['lbfgs']  # Algorithm to find best weights
```

### Algorithm 2: Support Vector Machine (SVM)

**Concept**: "Find the widest margin separating classes"

```
Malignant points
   XX  ← Margin (maximized)
------- Decision boundary
   OO
Benign points
```

**Pros**:
- Good for high-dimensional data
- Non-linear with kernel trick
- Works with medium-sized datasets

**Cons**:
- Slower training
- Hard to interpret

**Hyperparameters**:
```python
C: [0.1, 1, 10]  # Balance between margin width and misclassifications
kernel: ['rbf']  # Type of boundary (rbf = non-linear)
gamma: ['scale', 'auto']  # Influence of single training point
```

### Algorithm 3: Random Forest

**Concept**: "Multiple decision trees voting"

```
           Training Data
                |
        /       |       \
       /        |        \
    Tree1     Tree2     Tree3
     |         |         |
   Pred1    Pred2     Pred3
     \___________|_________/
          Vote (majority)
            Final
```

**Pros**:
- Handles non-linear relationships
- Feature importance built-in
- Robust to outliers
- Good out-of-the-box performance

**Cons**:
- Hard to interpret
- Can overfit with deep trees

**Hyperparameters**:
```python
max_depth: [5, 10, 20]  # Maximum tree depth
           # Small = simpler, larger = complex
min_samples_split: [2, 5, 10]  # Minimum samples to split node
           # Small = overfitting risk, large = underfitting risk
n_estimators: [50, 100]  # Number of trees
           # More trees = better (but slower)
```

### Algorithm 4: Gradient Boosting

**Concept**: "Sequential trees, each fixes previous mistakes"

```
Tree1 prediction: [0.4, 0.3, 0.8, ...]
         Error: [0.1, -0.2, 0.2, ...]  ← Residuals

Tree2 fits residuals: [0.15, -0.25, 0.25, ...]

Final = Tree1 + Tree2 + Tree3 + ...
```

**Pros**:
- Best performance typically
- Handles complex relationships
- Gradual refinement

**Cons**:
- Slower training
- Easy to overfit

**Hyperparameters**:
```python
learning_rate: [0.01, 0.1]  # How much each tree contributes
                # Small = accurate but slow
                # Large = fast but less accurate
max_depth: [3, 5, 7]  # Depth of each tree
n_estimators: [50, 100]  # Number of trees
```

### Algorithm 5: K-Nearest Neighbors (KNN)

**Concept**: "Classify based on nearest neighbors"

```
New point X
    |
    ├─ Neighbor 1 (Malignant)
    ├─ Neighbor 2 (Malignant)
    ├─ Neighbor 3 (Benign)
    |
Vote: 2 Malignant, 1 Benign
Predict: Malignant
```

**Pros**:
- Simple and intuitive
- No training required
- Non-parametric

**Cons**:
- Slow prediction (check all points)
- Need to store all data
- Sensitive to feature scaling

**Hyperparameters**:
```python
n_neighbors: [3, 5, 7, 9]  # Number of neighbors to consider
                # Small = might overfit
                # Large = might underfit
weights: ['uniform', 'distance']  # How to weight neighbors
                # uniform = all same
                # distance = closer neighbors matter more
```

## Step 3: Understand Hyperparameter Tuning (20 minutes)

### The Problem
```
Random Forest with max_depth=5 → 85% accuracy
Random Forest with max_depth=10 → 88% accuracy
Random Forest with max_depth=20 → 85% accuracy (overfitting!)

Which depth is best? 10!
```

### GridSearchCV Solution

```python
GridSearchCV(
    model,
    param_grid={'max_depth': [5, 10, 20]},
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc'
)
```

**What happens**:
1. **Create grid**: [5, 10, 20] = 3 options
2. **For each value**:
   - Split data into 5 folds
   - Train 4 folds, test 1 fold (repeat 5 times)
   - Average the 5 scores
3. **Select best**: depth=10 wins
4. **Retrain**: Train final model on all data with depth=10

**Why cross-validation?**
- More robust than single train/test split
- Uses all data for training
- Catches overfitting

### Example: 3 Hyperparameters

```python
GridSearchCV(
    RandomForestClassifier(),
    {
        'max_depth': [5, 10],        # 2 options
        'min_samples_split': [2, 5], # 2 options
        'n_estimators': [50, 100]    # 2 options
    },
    cv=5
)

Total combinations: 2 × 2 × 2 = 8
Each with 5-fold CV: 8 × 5 = 40 models trained!
Time: ~30 seconds on CPU
```

**Trade-off**:
- More hyperparams = better results but slower
- More CV folds = more robust but slower

## Step 4: Understand Evaluation Metrics (15 minutes)

### Binary Classification Matrix

```
                 Predicted
              Negative    Positive
Actual
Negative      TN (900)    FP (50)     [All Benign cases]
Positive      FN (20)     TP (30)     [All Malignant cases]

[All Benign Predictions] [All Malignant Predictions]
```

### Key Metrics

**Accuracy** = (TP + TN) / Total
```
= (30 + 900) / 1000 = 93%
General "correctness" but ignores class importance
```

**Precision** = TP / (TP + FP)
```
= 30 / (30 + 50) = 37.5%
"Of predicted malignant cases, how many actually were?"
Question: When we predict cancer, how often are we right?
Use when: False positives are expensive (unnecessary surgery)
```

**Recall** = TP / (TP + FN)
```
= 30 / (30 + 20) = 60%
"Of actual malignant cases, how many did we find?"
Question: Did we catch all cancer cases?
Use when: False negatives are expensive (missed cancer)
```

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
```
= 2 × (0.375 × 0.60) / (0.375 + 0.60) = 0.46
Harmonic mean (balances precision and recall)
Use when: Precision and recall both important
```

**ROC-AUC** = Area under ROC curve
```
Score: 0.0 to 1.0
0.5 = random guessing
1.0 = perfect classifier
0.7-0.8 = good
0.8-0.9 = excellent
0.9+ = exceptional

Best for: Imbalanced classes (more benign than malignant)
Why: Threshold-independent (doesn't assume 0.5 cutoff)
```

### Medical Context

**Cancer Diagnosis Scenario**:
- Ignore high accuracy if low recall!
- Miss 1 cancer case → patient dies
- False positive → another test, not life-threatening

**Better strategy**:
- High recall (catch all cancers) - might accept lower precision
- Or optimize F1 if both errors serious

## Step 5: Feature Importance & Selection (15 minutes)

### Feature Importance (Tree-based models)

**Concept**: "How much does each feature contribute to predictions?"

```python
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_

# Output:
# Feature1: 0.25 (uses this 25% of decisions)
# Feature2: 0.15 (uses this 15% of decisions)
# Feature3: 0.10
# ...

Top feature is "worst radius" → strong cancer indicator
```

**Action**: 
- Focus data collection on important features
- Less important features might be removed
- Understand why model thinks they matter

### Feature Selection (SelectKBest)

**Problem**: Too many features → overfitting

**Solution**: Keep only top 20 features

```python
SelectKBest(f_classif, k=20)  # Statistical test to score features
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)  # Remove low-score features
```

**Benefits**:
- Faster training
- Less overfitting
- Simpler model (easier to interpret)

## Step 6: Cross-Validation Robustness (10 minutes)

### Why 10-Fold CV?

**Single train/test**:
```
Train: 70% → Model achieves 90% on this split
Test: 30% → Model achieves 87% on this split
Result: 87% ← Might vary with different split!
```

**10-Fold Cross-Validation**:
```
Fold1: Train on [2-10], Test on [1]  → 87%
Fold2: Train on [1,3-10], Test on [2] → 89%
Fold3: Train on [1-2,4-10], Test on [3] → 86%
...
Fold10: Train on [1-9], Test on [10] → 88%

Average: 87.4%
Std Dev: 1.2%
Range: 86%-89%

Conclusion: Model consistently achieves 86-89% regardless of split
```

**Interpretation**:
- Mean (87.4%) = expected performance
- Std (1.2%) = variability (low = stable model)
- Range (86-89%) = likely range in production

## Step 7: Run the Project (3-5 minutes)

```bash
cd projects/13-automated-ml-pipeline
pip install -r requirements.txt
python train.py
```

**Expected output**:
```
[1/6] DATA PREPARATION...
  ✓ Dataset: Breast Cancer Classification
  ✓ Samples: 569
  ✓ Features: 30

[2/6] FEATURE SELECTION...
  ✓ Selected top 20 features

[3/6] MODEL TUNING...
  Tuning Logistic Regression...
    Best Params: {'C': 1, 'penalty': 'l2'}
    Test Accuracy: 0.9649
    
  Tuning Random Forest...
    Best Params: {'max_depth': 10, 'n_estimators': 100}
    Test Accuracy: 0.9561
    
  ... (SVM, Gradient Boosting, KNN)

[4/6] BEST MODEL EVALUATION...
  Best Model: Gradient Boosting (ROC-AUC: 0.9876)
  
  Detailed Metrics:
    Accuracy: 0.9649
    Precision: 0.9722
    Recall: 0.9545
    F1: 0.9633
    ROC-AUC: 0.9876

[5/6] FEATURE IMPORTANCE...
  Top Features:
    1. worst perimeter: 0.342
    2. worst concave points: 0.289
    ...

[6/6] CROSS-VALIDATION...
  10-Fold Scores: [0.973, 0.966, 0.980, ...]
  Mean: 0.9721
  Std: 0.0052
  Range: 0.966-0.980
```

**Time**: ~2-3 minutes

```bash
python app.py
# Visit http://localhost:8013
# Try different feature values
# See model comparison
```

## Step 8: Key Learnings Checklist ✓

- [ ] 5 ML algorithms and when to use each
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for robustness
- [ ] Accuracy, Precision, Recall, F1, ROC-AUC
- [ ] Confusion matrix interpretation
- [ ] Feature selection and importance
- [ ] Avoiding overfitting
- [ ] Model comparison workflow
- [ ] scikit-learn Pipeline concept
- [ ] Ensemble methods (Random Forest, Gradient Boosting)
- [ ] When to optimize for precision vs recall
- [ ] Production ML considerations

---

# Hands-On Exercises

## Exercise 1: Modify CNN Architecture (30 minutes)

**Goal**: Understand how architecture changes affect performance

**Task 1.1**: Add regularization
```python
# In train.py, increase dropout from 0.25 to 0.4 in first block
x = layers.Dropout(0.4)(x)  # Was 0.25

# What happens?
# - Training loss: might be HIGHER (model underfits)
# - Test loss: might be LOWER (more regularization)
# - Best case: better generalization
```

**Task 1.2**: Reduce model capacity
```python
# Change Conv filters: 64→48, 128→96, 256→192
# What happens?
# - Training time: FASTER (fewer parameters)
# - Accuracy: LOWER (less capacity)
# - Conclusion: Trade-off between speed and accuracy
```

**Task 1.3**: Remove early stopping
```python
# Comment out EarlyStopping callback
# Train for full 30 epochs
# What happens?
# - Around epoch 15-20: validation loss stops improving
# - After: validation loss increases (overfitting!)
# - Conclusion: Early stopping is important
```

## Exercise 2: Analyze Recommendation Metrics (30 minutes)

**Goal**: Understand NDCG vs Hit Rate deeply

**Task 2.1**: Manually calculate NDCG
```python
# In Python REPL:
predictions = [5.0, 4.8, 3.2, 2.1, 1.9]  # Predicted ratings
actuals =     [5.0, 4.0, 3.0, 2.0, 1.0]  # Actual ratings

# Calculate DCG
import numpy as np
dcg = np.sum((predictions >= 3.5).astype(float) / np.log2(np.arange(2, 7)))
# Interpretation: How good is the ranking?
```

**Task 2.2**: Try different user profiles
```python
# In app.py:
# User 0: Try prediction
# User 100: Try prediction
# User 250: Try prediction
# Compare results
# Question: Do recommendations differ per user?
```

## Exercise 3: Hyperparameter Sensitivity (30 minutes)

**Goal**: See how hyperparameters affect accuracy

**Task 3.1**: Test different max_depth values
```python
# In train.py, modify GridSearchCV:
'max_depth': [2, 3, 5, 10, 15, 20, 30]  # More options

# Train and compare
# Plot accuracy vs depth
# Question: Where's the sweet spot?
# Answer: Usually around depth=10 (avoid overfitting)
```

**Task 3.2**: Test different numbers of features
```python
# Change SelectKBest from k=20 to different values:
for k in [5, 10, 20, 30]:
    selector = SelectKBest(f_classif, k=k)
    # Train and compare accuracy

# Question: How does feature count affect accuracy?
# Answer: 20-30 features optimal, diminishing returns after
```

## Exercise 4: Compare Model Interpretability (20 minutes)

**Goal**: Understand which models are interpretable

**Task 4.1**: Check model interpretability
```python
# Logistic Regression: Very interpretable
coefs = log_reg.coef_[0]
for i, coef in enumerate(coefs):
    print(f"Feature {i}: {coef:.4f}")
# Interpretation: Positive coef = increases cancer probability

# Random Forest: Somewhat interpretable
importances = rf.feature_importances_
# Interpretation: Shows which features matter, not how

# Neural Network (Project 12): Black box
# Interpretation: Hard to understand why it made predictions
```

## Exercise 5: Medical Decision Threshold (20 minutes)

**Goal**: Optimize for medical context (recall > precision)

**Task 5.1**: Test different thresholds
```python
# Default: Threshold = 0.5
# Benign if probability < 0.5
# Malignant if probability >= 0.5

# Try: Threshold = 0.3
# More false positives, fewer false negatives
# Why? Catch more cancers, even if some are false alarms

# Code: Adjust probability threshold in app.py:
predictions = model.predict_proba(X_test)[:, 1]
y_pred = (predictions >= 0.3).astype(int)  # Lower threshold
```

---

# Common Mistakes & Solutions

## CNN Mistakes (Project 11)

### Mistake 1: Not normalizing images
```python
# ❌ Wrong
X_train = keras.datasets.cifar10.load_data()[0]  # [0-255]
# Problem: Gradients explode, training fails

# ✅ Correct
X_train = X_train.astype('float32') / 255.0  # [0-1]
# Solution: Normalized gradients, stable training
```

### Mistake 2: Removing too much with dropout
```python
# ❌ Wrong
layers.Dropout(0.8)  # Disable 80% of neurons!
# Problem: Model can't learn (data destroyed)

# ✅ Correct
layers.Dropout(0.25)  # Disable 25%
# Solution: Still learn, but regularized
```

### Mistake 3: Forgetting to normalize validation data
```python
# ❌ Wrong
X_val = validation_data  # Unnormalized
model.predict(X_val)  # Wrong predictions!

# ✅ Correct
X_val = validation_data / 255.0
model.predict(X_val)  # Consistent with training
```

## PyTorch Mistakes (Project 12)

### Mistake 1: Forgetting model.eval()
```python
# ❌ Wrong
model.train()
model.eval()  # Forgot this line during prediction!
predictions = model(test_data)  # Dropout still active!
# Problem: Inconsistent predictions (dropout randomness)

# ✅ Correct
model.eval()  # Disable dropout
with torch.no_grad():  # Don't compute gradients
    predictions = model(test_data)
```

### Mistake 2: Not zeroing gradients
```python
# ❌ Wrong
for epoch in range(10):
    for batch in loader:
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()
# Problem: Gradients accumulate, wrong updates!

# ✅ Correct
optimizer.zero_grad()  # Clear old gradients
loss.backward()
optimizer.step()
```

### Mistake 3: Clamping predictions
```python
# ❌ Wrong (in train.py)
predictions = model(users, movies)  # Can be negative, >5!
loss = criterion(predictions, ratings)  # Unbounded loss

# ✅ Correct
predictions = torch.clamp(model(users, movies), 1, 5)
# Solution: Predictions in valid range [1, 5]
```

## scikit-learn Mistakes (Project 13)

### Mistake 1: Forgetting to fit scaler on train only
```python
# ❌ Wrong
scaler = StandardScaler()
scaler.fit(X)  # Fit on all data (data leakage!)
X_train_scaled = scaler.transform(X_train)

# ✅ Correct
scaler.fit(X_train)  # Fit only on training
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use train stats
```

### Mistake 2: Grid search without CV
```python
# ❌ Wrong
model = RandomForest(max_depth=10)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Single test set!
# Problem: Score varies with different test sets

# ✅ Correct
GridSearchCV(RandomForest(), {'max_depth': range(3,20)}, cv=5)
# Solution: Robust score from 5-fold CV
```

### Mistake 3: Feature selection bias
```python
# ❌ Wrong
selector = SelectKBest(f_classif, k=20)
selector.fit(X)  # Select features on all data!
X_train = selector.transform(X_train)
# Problem: Features selected using test data (leakage!)

# ✅ Correct
selector.fit(X_train)  # Select features from train only
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
```

---

# Interview Prep

## Common Interview Questions

### Question 1: "Explain CNNs to someone who doesn't know ML"

**Answer**:
```
CNN is like a scanning system that learns to recognize patterns.

Imagine you have a large image and want to find cats:
1. Scanning (Conv layer): Small 3×3 window slides over image
   - Learns to detect edges, textures
   - Multiple windows look for different patterns
   
2. Compression (MaxPool): Keep only important features
   - Reduces image size
   - Focuses on strongest signals
   
3. Learning (Dense layer): Combine features to make decision
   - Learns: "These patterns together = cat"
   
Progressive learning:
- Layer 1: Learns edges (low-level)
- Layer 2: Learns shapes (mid-level)  
- Layer 3: Learns parts (high-level)
- Output: Makes decision

That's why CNNs work for images - they understand spatial patterns!
```

### Question 2: "What's the difference between precision and recall?"

**Answer**:
```
Think of cancer detection:

Precision = "Of all cancer cases we PREDICTED, how many were actually cancer?"
- Question: When we say "cancer", how often are we right?
- Formula: TP / (TP + FP)
- Medical: False positive = unnecessary surgery (bad but survivable)

Recall = "Of all ACTUAL cancer cases, how many did we find?"
- Question: Did we catch all real cancer?
- Formula: TP / (TP + FN)
- Medical: False negative = missed cancer (worse - patient dies)

Example:
100 patients, 10 have cancer
Model predictions: Find 8 cancer, 2 false alarms

Precision = 8 / (8 + 2) = 80% (when we predict cancer, we're right 80%)
Recall = 8 / 10 = 80% (we caught 80% of actual cancers)

For cancer: RECALL IS MORE IMPORTANT!
Better to have false alarms (can do more tests) than miss real cases
```

### Question 3: "Why does hyperparameter tuning matter?"

**Answer**:
```
Wrong hyperparameters = wrong model performance

Example with Random Forest:
max_depth=2   → 70% accuracy (UNDERFITTING - too simple)
max_depth=10  → 92% accuracy (JUST RIGHT - good balance)
max_depth=50  → 88% accuracy (OVERFITTING - memorized training)

GridSearchCV tries many combinations:
- Tests max_depth ∈ [2, 5, 10, 20, 30, 50]
- Ranks each by cross-validation score
- Picks best (max_depth=10)
- Retrains on all data

Without tuning: Might pick max_depth=2 (picks first option)
With tuning: Picks max_depth=10 (optimal)

Difference: 70% vs 92% accuracy!
```

### Question 4: "When would you use scikit-learn vs TensorFlow?"

**Answer**:
```
Use scikit-learn when:
✓ Data is tabular (rows and columns)
✓ Dataset is small to medium (<1M rows)
✓ Need quick prototype
✓ Need interpretability (feature importance)
✓ Need traditional ML (not images/text)
Examples: Loan approval, customer churn, medical diagnosis

Use TensorFlow when:
✓ Data is unstructured (images, text, audio)
✓ Dataset is large (1M+ rows)
✓ Need high accuracy (willing to spend time)
✓ Computer vision or NLP task
✓ Need to deploy on mobile/edge
Examples: Image classification, object detection, chatbots

Middle ground: Both can handle similar problems, but different strengths
```

### Question 5: "Walk me through your ML pipeline"

**Answer**:
```
1. DATA PREPARATION
   - Load data
   - Handle missing values
   - Normalize/scale features
   - Split train/test (80/20)
   - Stratify if imbalanced

2. FEATURE ENGINEERING
   - Select important features (SelectKBest)
   - Create new features
   - Handle categorical data

3. MODEL SELECTION
   - Try multiple algorithms
   - Use cross-validation (5-fold)
   - Compare metrics (accuracy, F1, AUC)
   - Pick best model

4. HYPERPARAMETER TUNING
   - GridSearchCV to find best params
   - Cross-validation for robustness
   - Train on best params

5. EVALUATION
   - Final test set performance
   - Per-class metrics
   - Confusion matrix
   - Feature importance analysis

6. DEBUGGING
   - Identify misclassified examples
   - Understand model errors
   - Consider improvements

7. DEPLOYMENT
   - Save model
   - Create API
   - Monitor performance in production
```

### Question 6: "How do you prevent overfitting?"

**Answer**:
```
Overfitting = Model memorizes training data instead of learning patterns

Prevention strategies:

1. REGULARIZATION (reduce model complexity)
   - Dropout: Randomly disable neurons (25-50%)
   - Batch Norm: Normalize layer inputs
   - L1/L2: Penalize large weights
   
2. DATA (more and better data)
   - Collect more training data
   - Data augmentation (flip, rotate images)
   - Remove noisy samples
   
3. VALIDATION (monitor while training)
   - Use validation set during training
   - Early stopping: Stop when validation loss increases
   - K-fold cross-validation: Multiple train/test splits
   
4. ARCHITECTURE (simpler model)
   - Fewer layers
   - Fewer neurons
   - Fewer parameters
   
5. HYPERPARAMETERS (less powerful model)
   - Lower learning rate
   - Higher regularization
   - Fewer training epochs

Best approach: Combine multiple strategies!
Example: Dropout + Early Stopping + Cross-validation
```

## Practice Problems

### Problem 1: Model Comparison
```
You have 3 models:
Model A: 95% accuracy, 70% recall
Model B: 92% accuracy, 92% recall
Model C: 90% accuracy, 95% recall

For detecting fraud (false negatives are expensive):
Which model would you choose? Why?

Answer: Model C
Reason: Recall is most important (catch fraudsters!)
92-95% recall is much better than 70% recall
Acceptable to have more false positives for fraud detection
```

### Problem 2: Metrics Interpretation
```
A medical test has:
- Sensitivity (Recall) = 95%
- Specificity = 80%
- Precision = 70%

Question: If test shows "positive", what's probability patient has disease?

Answer: ~70% (this is Precision!)
Reason: Precision = TP / (TP + FP) = likelihood test is correct
        High sensitivity means we catch most real cases
        Low specificity means we have many false alarms
        70% precision means 70% of "positive" results are real
```

### Problem 3: Hyperparameter Selection
```
You tested 5 max_depth values in GridSearchCV:
max_depth=3:  CV Score = 0.82 ± 0.05
max_depth=5:  CV Score = 0.87 ± 0.04
max_depth=10: CV Score = 0.89 ± 0.03
max_depth=20: CV Score = 0.88 ± 0.06
max_depth=30: CV Score = 0.84 ± 0.08

Question: Which depth would you pick? Why?

Answer: max_depth=10
Reasons:
- Highest mean score (0.89)
- Lowest std dev (0.03) = most stable
- max_depth=20 close but higher variance
- max_depth=30 shows signs of overfitting (lower score, high variance)
```

---

## Quick Reference Guide

### When to Use Which Metric

| Metric | When to Use | Medical Example |
|--------|------------|-----------------|
| Accuracy | Balanced classes | Overall success rate |
| Precision | False positives expensive | Avoid unnecessary surgery |
| Recall | False negatives expensive | Don't miss cancer |
| F1 | Both matter equally | General diagnosis |
| ROC-AUC | Imbalanced data | Rare disease detection |

### Algorithm Decision Tree

```
Is your data...

Images/Audio?
  YES → Use CNN (TensorFlow/Project 11)
  NO → Go to next question

Text?
  YES → Use RNN/Transformer (PyTorch)
  NO → Go to next question

Tabular with <1M rows?
  YES → Use scikit-learn (Project 13)
  NO → Use Spark/TensorFlow

Need to interpret model?
  YES → Use Linear/Tree models
  NO → Can use Neural Network
```

---

## Testing Your Knowledge

After completing all projects, you should be able to:

### Project 11 Knowledge
- [ ] Explain how convolution layers work
- [ ] Design a CNN for a problem
- [ ] Interpret per-class accuracy results
- [ ] Identify overfitting from validation curves
- [ ] Deploy a model with FastAPI

### Project 12 Knowledge
- [ ] Explain embeddings and why they work
- [ ] Write a PyTorch training loop
- [ ] Calculate NDCG@5 by hand
- [ ] Compare ranking metrics vs classification metrics
- [ ] Debug PyTorch models with print statements

### Project 13 Knowledge
- [ ] Explain GridSearchCV and cross-validation
- [ ] Calculate precision/recall/F1 from confusion matrix
- [ ] Interpret feature importance
- [ ] Choose between algorithms
- [ ] Prevent overfitting with regularization

### General Knowledge
- [ ] Describe end-to-end ML pipeline
- [ ] Explain train/test/validation splits
- [ ] Compare TensorFlow vs PyTorch vs scikit-learn
- [ ] Choose metrics based on business context
- [ ] Handle class imbalance

---

## Next Steps After Projects

1. **Week 1-2**: Master one project deeply
2. **Week 3-4**: Master all three projects
3. **Week 5**: Do hands-on exercises
4. **Week 6**: Practice interview questions
5. **Week 7**: Build your own project combining concepts

**Your own project ideas**:
- Sentiment analysis (Project 11 + 13)
- Movie recommendation (Project 12)
- Fraud detection (Project 13)
- Image classification on custom data (Project 11)
- Build end-to-end system with API (all projects)

---

**You're ready for ML engineering interviews!** 🚀
