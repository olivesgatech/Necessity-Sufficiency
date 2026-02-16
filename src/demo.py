"""
Quick Demo Script

Run a fast demonstration of the necessity and sufficiency framework.
This uses reduced sample sizes for quick execution (~2-3 minutes).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from counterfactual_generator import ForwardCounterfactualGenerator, GlobalScoreCalculator
from visualization import NecessitySufficiencyVisualizer
import os

print("=" * 70)
print("QUICK DEMO: Necessity and Sufficiency Analysis")
print("=" * 70)
print("\nThis demo will:")
print("1. Load Breast Cancer dataset")
print("2. Train a Logistic Regression model")
print("3. Calculate necessity & sufficiency for top 5 features")
print("4. Generate visualizations")
print("\nEstimated time: 2-3 minutes\n")
print("=" * 70)

# Create output directory
os.makedirs('../results', exist_ok=True)

# Load and prepare data
print("\n[1/4] Loading Breast Cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Classes: {np.bincount(y)}")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
print("\n[2/4] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"   Training accuracy: {train_acc:.4f}")
print(f"   Test accuracy: {test_acc:.4f}")

# Calculate scores (limited features for speed)
print("\n[3/4] Calculating necessity & sufficiency scores...")
print("   (Using top 5 features only for demo speed)")

# Get top 5 features by coefficient magnitude
coef = np.abs(model.coef_[0])
top_indices = np.argsort(coef)[-5:][::-1]

print(f"\n   Top 5 features:")
for rank, idx in enumerate(top_indices, 1):
    print(f"   {rank}. {feature_names[idx]}")

# Initialize CF generator
cf_generator = ForwardCounterfactualGenerator(
    model=model,
    feature_names=feature_names,
    n_perturbations=30  # Reduced for speed
)

# Calculate scores only for top features
calculator = GlobalScoreCalculator(cf_generator)

necessity_scores = np.zeros(len(feature_names))
sufficiency_scores = np.zeros(len(feature_names))

for idx in top_indices:
    print(f"\n   Processing: {feature_names[idx]}")
    necessity_scores[idx] = calculator.calculate_global_necessity(
        X_test, y_test, idx, n_samples=30  # Reduced for speed
    )
    sufficiency_scores[idx] = calculator.calculate_global_sufficiency(
        X_test, y_test, idx, n_samples=30  # Reduced for speed
    )
    print(f"     Necessity: {necessity_scores[idx]:.4f}")
    print(f"     Sufficiency: {sufficiency_scores[idx]:.4f}")

# Visualize
print("\n[4/4] Generating visualizations...")
visualizer = NecessitySufficiencyVisualizer()

output_path = '../results/demo_global_scores.png'
visualizer.plot_global_scores(
    feature_names=feature_names,
    necessity_scores=necessity_scores,
    sufficiency_scores=sufficiency_scores,
    save_path=output_path,
    top_k=5
)

print("\n" + "=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print(f"\nResults saved to: {output_path}")
print("\nKey Findings:")
print("-" * 70)

# Show top feature
top_feat_idx = top_indices[0]
print(f"\nMost Important Feature: {feature_names[top_feat_idx]}")
print(f"  Necessity: {necessity_scores[top_feat_idx]:.4f}")
print(f"  Sufficiency: {sufficiency_scores[top_feat_idx]:.4f}")

if necessity_scores[top_feat_idx] > 0.5 and sufficiency_scores[top_feat_idx] > 0.3:
    print("  ✓ This feature is both necessary AND sufficient")
elif necessity_scores[top_feat_idx] > 0.5:
    print("  ✓ This feature is necessary but not sufficient alone")
elif sufficiency_scores[top_feat_idx] > 0.3:
    print("  ✓ This feature is sufficient but not always necessary")
else:
    print("  ⚠ This feature has low necessity and sufficiency")

print("\n" + "=" * 70)
print("Next Steps:")
print("-" * 70)
print("1. Run full analysis: python main.py --dataset breast_cancer")
print("2. Try different models: python main.py --model random_forest")
print("3. Explore notebooks: jupyter notebook ../notebooks/")
print("=" * 70)
