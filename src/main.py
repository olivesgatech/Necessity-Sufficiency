"""
Main Analysis Script

Run complete necessity and sufficiency analysis on a dataset.

Usage:
    python main.py --dataset breast_cancer --model logistic
    python main.py --dataset adult --model random_forest
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer, load_iris
import warnings
warnings.filterwarnings('ignore')

from counterfactual_generator import ForwardCounterfactualGenerator, GlobalScoreCalculator
from xai_evaluator import XAIRobustnessEvaluator
from visualization import NecessitySufficiencyVisualizer
import os


def load_data(dataset_name: str):
    """
    Load dataset.
    
    Args:
        dataset_name: Name of dataset ('breast_cancer', 'iris', or 'custom')
        
    Returns:
        X, y, feature_names
    """
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        
    elif dataset_name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        # Convert to binary classification
        y = (y == 2).astype(int)
        feature_names = list(data.feature_names)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, feature_names


def train_model(X_train, y_train, model_name: str):
    """
    Train classification model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Model type ('logistic', 'random_forest', 'gaussian_nb')
        
    Returns:
        Trained model
    """
    if model_name == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'gaussian_nb':
        model = GaussianNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser(description='Necessity and Sufficiency Analysis')
    parser.add_argument('--dataset', type=str, default='breast_cancer',
                       choices=['breast_cancer', 'iris'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='logistic',
                       choices=['logistic', 'random_forest', 'gaussian_nb'],
                       help='Model to train')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples for global score calculation')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top features for robustness analysis')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("NECESSITY AND SUFFICIENCY ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Top-k features: {args.top_k}")
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    X, y, feature_names = load_data(args.dataset)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    model = train_model(X_train, y_train, args.model)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Initialize counterfactual generator
    cf_generator = ForwardCounterfactualGenerator(
        model=model,
        feature_names=feature_names,
        n_perturbations=50
    )
    
    # Calculate global scores
    print("\n" + "=" * 80)
    print("CALCULATING GLOBAL NECESSITY AND SUFFICIENCY SCORES")
    print("=" * 80)
    global_calculator = GlobalScoreCalculator(cf_generator)
    
    necessity_scores, sufficiency_scores = global_calculator.calculate_all_global_scores(
        X_test, y_test, n_samples=args.n_samples
    )
    
    # Save scores
    scores_df = pd.DataFrame({
        'Feature': feature_names,
        'Necessity': necessity_scores,
        'Sufficiency': sufficiency_scores
    })
    scores_df = scores_df.sort_values('Necessity', ascending=False)
    scores_path = os.path.join(args.output_dir, f'{args.dataset}_{args.model}_scores.csv')
    scores_df.to_csv(scores_path, index=False)
    print(f"\nSaved scores to {scores_path}")
    
    # Visualize global scores
    print("\n" + "=" * 80)
    print("VISUALIZING GLOBAL SCORES")
    print("=" * 80)
    visualizer = NecessitySufficiencyVisualizer()
    
    global_plot_path = os.path.join(args.output_dir, 
                                    f'{args.dataset}_{args.model}_global_scores.png')
    visualizer.plot_global_scores(
        feature_names=feature_names,
        necessity_scores=necessity_scores,
        sufficiency_scores=sufficiency_scores,
        save_path=global_plot_path,
        top_k=7
    )
    
    # XAI Robustness Evaluation
    print("\n" + "=" * 80)
    print("XAI ROBUSTNESS EVALUATION")
    print("=" * 80)
    
    xai_evaluator = XAIRobustnessEvaluator(
        model=model,
        X_train=X_train,
        feature_names=feature_names
    )
    
    robustness_results = xai_evaluator.full_robustness_evaluation(
        X_test=X_test,
        necessity_scores=necessity_scores,
        sufficiency_scores=sufficiency_scores,
        top_k=args.top_k,
        n_samples=min(50, len(X_test))
    )
    
    # Save robustness results
    robustness_df = pd.DataFrame(robustness_results)
    robustness_path = os.path.join(args.output_dir, 
                                   f'{args.dataset}_{args.model}_robustness.csv')
    robustness_df.to_csv(robustness_path, index=False)
    print(f"\nSaved robustness results to {robustness_path}")
    
    # Visualize robustness analysis
    print("\n" + "=" * 80)
    print("VISUALIZING ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    robustness_plot_path = os.path.join(args.output_dir,
                                       f'{args.dataset}_{args.model}_robustness.png')
    visualizer.plot_robustness_analysis(
        robustness_results=robustness_results,
        save_path=robustness_plot_path
    )
    
    comparison_plot_path = os.path.join(args.output_dir,
                                       f'{args.dataset}_{args.model}_comparison.png')
    visualizer.plot_comparison(
        robustness_results=robustness_results,
        save_path=comparison_plot_path
    )
    
    table_path = os.path.join(args.output_dir,
                             f'{args.dataset}_{args.model}_table.png')
    visualizer.create_summary_table(
        robustness_results=robustness_results,
        save_path=table_path
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
