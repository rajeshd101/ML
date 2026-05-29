import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Generate synthetic Mall Customers dataset
print("Generating synthetic Mall Customers dataset...")
np.random.seed(42)
n_samples = 200
df = pd.DataFrame({
    'CustomerID': np.arange(1, n_samples + 1),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.randint(18, 70, n_samples),
    'Annual Income (k$)': np.random.randint(15, 140, n_samples),
    'Spending Score (1-100)': np.random.randint(1, 100, n_samples)
})
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Encode Gender: Male=1, Female=0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Features: Age, Annual Income, Spending Score (skip CustomerID, Gender)
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Standardize using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Try different k values and pick k=5
print("\nFinding optimal k (trying k=3,4,5,6):")
for k in [3, 4, 5, 6]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    print(f"  k={k}: inertia={kmeans.inertia_:.2f}")

# Fit KMeans with k=5
k = 5
print(f"\nFitting KMeans with k={k}...")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save models
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Models saved: kmeans_model.pkl, scaler.pkl")

# Print cluster centers and counts per cluster
print("\nCluster Centers (standardized space):")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  Cluster {i}: {center}")

print("\nCluster Centers (original space):")
original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(original_centers):
    print(f"  Cluster {i}: Age={center[0]:.1f}, Income=${center[1]:.1f}k, Spending={center[2]:.1f}")

print("\nCounts per cluster:")
labels = kmeans.labels_
unique, counts = np.unique(labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} customers")

print("\nTraining complete!")