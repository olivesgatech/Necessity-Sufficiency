"""
XAI Robustness Evaluator

This module evaluates the robustness of LIME and SHAP explanations
using necessity and sufficiency scores.

Based on:
- Chowdhury et al. (2023) "Explaining Explainers: Necessity and Sufficiency in Tabular Data"
"""

import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import lime
import lime.lime_tabular
import shap


class XAIRobustnessEvaluator:
    """
    Evaluate robustness of LIME and SHAP explanations using
    necessity and sufficiency scores.
    """
    
    def __init__(self, 
                 model,
                 X_train: np.ndarray,
                 feature_names: List[str],
                 class_names: List[str] = None):
        """
        Initialize XAI robustness evaluator.
        
        Args:
            model: Trained classifier
            X_train: Training data for LIME and SHAP
            feature_names: List of feature names
            class_names: List of class names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names or ['Class 0', 'Class 1']
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=self.class_names,
            mode='classification'
        )
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.KernelExplainer(
            self.model.predict_proba,
            shap.sample(X_train, 100)
        )
    
    def get_lime_feature_ranking(self, 
                                instance: np.ndarray,
                                top_k: int = None) -> List[Tuple[int, float]]:
        """
        Get LIME feature importance ranking for an instance.
        
        Args:
            instance: Single data instance
            top_k: Number of top features to return (None for all)
            
        Returns:
            List of (feature_index, importance) tuples sorted by importance
        """
        # Get LIME explanation
        exp = self.lime_explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        # Extract feature importances
        feature_importance = exp.as_list()
        
        # Parse feature indices and importances
        rankings = []
        for feat_str, importance in feature_importance:
            # Extract feature name from string
            feat_name = feat_str.split('<=')[0].split('>')[0].strip()
            if feat_name in self.feature_names:
                feat_idx = self.feature_names.index(feat_name)
                rankings.append((feat_idx, abs(importance)))
        
        # Sort by importance (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            rankings = rankings[:top_k]
        
        return rankings
    
    def get_shap_feature_ranking(self,
                                instance: np.ndarray,
                                top_k: int = None) -> List[Tuple[int, float]]:
        """
        Get SHAP feature importance ranking for an instance.
        
        Args:
            instance: Single data instance
            top_k: Number of top features to return (None for all)
            
        Returns:
            List of (feature_index, importance) tuples sorted by importance
        """
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(instance.reshape(1, -1))
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        shap_values = shap_values.flatten()
        
        # Create rankings
        rankings = [(i, abs(shap_values[i])) for i in range(len(shap_values))]
        
        # Sort by importance (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            rankings = rankings[:top_k]
        
        return rankings
    
    def evaluate_necessity_robustness(self,
                                     X_test: np.ndarray,
                                     necessity_scores: np.ndarray,
                                     method: str = 'lime',
                                     top_k: int = 5,
                                     n_samples: int = 50) -> Dict[str, np.ndarray]:
        """
        Evaluate necessity robustness of LIME or SHAP explanations.
        
        Args:
            X_test: Test instances
            necessity_scores: Global necessity scores for all features
            method: 'lime' or 'shap'
            top_k: Evaluate top-k features
            n_samples: Number of test samples to evaluate
            
        Returns:
            Dictionary with rank-wise necessity scores
        """
        n_samples = min(n_samples, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        # Store necessity scores for each rank
        rank_necessity = {k: [] for k in range(1, top_k + 1)}
        
        print(f"\nEvaluating {method.upper()} necessity robustness...")
        
        for idx in tqdm(sample_indices):
            instance = X_test[idx]
            
            # Get feature ranking
            if method.lower() == 'lime':
                rankings = self.get_lime_feature_ranking(instance, top_k)
            else:
                rankings = self.get_shap_feature_ranking(instance, top_k)
            
            # Record necessity scores for each rank
            for rank, (feat_idx, _) in enumerate(rankings, 1):
                if rank <= top_k:
                    rank_necessity[rank].append(necessity_scores[feat_idx])
        
        # Calculate mean necessity for each rank
        mean_necessity = {}
        for rank in range(1, top_k + 1):
            if rank_necessity[rank]:
                mean_necessity[f'rank_{rank}'] = np.mean(rank_necessity[rank])
            else:
                mean_necessity[f'rank_{rank}'] = 0.0
        
        return mean_necessity
    
    def evaluate_sufficiency_robustness(self,
                                       X_test: np.ndarray,
                                       sufficiency_scores: np.ndarray,
                                       method: str = 'lime',
                                       top_k: int = 5,
                                       n_samples: int = 50) -> Dict[str, np.ndarray]:
        """
        Evaluate sufficiency robustness of LIME or SHAP explanations.
        
        Args:
            X_test: Test instances
            sufficiency_scores: Global sufficiency scores for all features
            method: 'lime' or 'shap'
            top_k: Evaluate top-k features
            n_samples: Number of test samples to evaluate
            
        Returns:
            Dictionary with rank-wise sufficiency scores
        """
        n_samples = min(n_samples, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        # Store sufficiency scores for each rank
        rank_sufficiency = {k: [] for k in range(1, top_k + 1)}
        
        print(f"\nEvaluating {method.upper()} sufficiency robustness...")
        
        for idx in tqdm(sample_indices):
            instance = X_test[idx]
            
            # Get feature ranking
            if method.lower() == 'lime':
                rankings = self.get_lime_feature_ranking(instance, top_k)
            else:
                rankings = self.get_shap_feature_ranking(instance, top_k)
            
            # Record sufficiency scores for each rank
            for rank, (feat_idx, _) in enumerate(rankings, 1):
                if rank <= top_k:
                    rank_sufficiency[rank].append(sufficiency_scores[feat_idx])
        
        # Calculate mean sufficiency for each rank
        mean_sufficiency = {}
        for rank in range(1, top_k + 1):
            if rank_sufficiency[rank]:
                mean_sufficiency[f'rank_{rank}'] = np.mean(rank_sufficiency[rank])
            else:
                mean_sufficiency[f'rank_{rank}'] = 0.0
        
        return mean_sufficiency
    
    def full_robustness_evaluation(self,
                                  X_test: np.ndarray,
                                  necessity_scores: np.ndarray,
                                  sufficiency_scores: np.ndarray,
                                  top_k: int = 5,
                                  n_samples: int = 50) -> Dict:
        """
        Perform complete robustness evaluation for both LIME and SHAP.
        
        Args:
            X_test: Test instances
            necessity_scores: Global necessity scores
            sufficiency_scores: Global sufficiency scores
            top_k: Evaluate top-k features
            n_samples: Number of test samples
            
        Returns:
            Dictionary with all robustness evaluation results
        """
        results = {}
        
        # Evaluate LIME
        results['lime_necessity'] = self.evaluate_necessity_robustness(
            X_test, necessity_scores, 'lime', top_k, n_samples
        )
        results['lime_sufficiency'] = self.evaluate_sufficiency_robustness(
            X_test, sufficiency_scores, 'lime', top_k, n_samples
        )
        
        # Evaluate SHAP
        results['shap_necessity'] = self.evaluate_necessity_robustness(
            X_test, necessity_scores, 'shap', top_k, n_samples
        )
        results['shap_sufficiency'] = self.evaluate_sufficiency_robustness(
            X_test, sufficiency_scores, 'shap', top_k, n_samples
        )
        
        return results
