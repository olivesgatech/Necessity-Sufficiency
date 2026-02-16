"""
Forward Counterfactual Generation Module

This module implements forward counterfactual generation for calculating
necessity and sufficiency scores without requiring a target prediction.

Based on:
- Chowdhury et al. (2023) "Explaining Explainers: Necessity and Sufficiency in Tabular Data"
- Chowdhury et al. (2025) "A unified framework for evaluating the robustness of 
  machine-learning interpretability for prospect risking"
"""

import numpy as np
from typing import Tuple, List, Optional
from tqdm import tqdm


class ForwardCounterfactualGenerator:
    """
    Generate forward counterfactuals by perturbing individual features
    without targeting a specific prediction outcome.
    """
    
    def __init__(self, model, feature_names: List[str], n_perturbations: int = 50):
        """
        Initialize the counterfactual generator.
        
        Args:
            model: Trained classifier with predict method
            feature_names: List of feature names
            n_perturbations: Number of perturbations per feature
        """
        self.model = model
        self.feature_names = feature_names
        self.n_perturbations = n_perturbations
        
    def generate_counterfactuals(self, 
                                 instance: np.ndarray, 
                                 feature_idx: int,
                                 X_context: np.ndarray) -> np.ndarray:
        """
        Generate counterfactuals by perturbing a single feature.
        
        Args:
            instance: Single data instance (1D array)
            feature_idx: Index of feature to perturb
            X_context: Context data for sampling perturbation values
            
        Returns:
            Array of counterfactual instances
        """
        counterfactuals = []
        
        # Get the range of values for this feature from context
        feature_values = X_context[:, feature_idx]
        
        # Sample perturbation values different from the original
        original_value = instance[feature_idx]
        
        # Create perturbations
        for _ in range(self.n_perturbations):
            cf = instance.copy()
            # Sample a different value from the context distribution
            new_value = np.random.choice(feature_values)
            # Ensure it's different from original
            attempts = 0
            while new_value == original_value and attempts < 10:
                new_value = np.random.choice(feature_values)
                attempts += 1
            cf[feature_idx] = new_value
            counterfactuals.append(cf)
            
        return np.array(counterfactuals)
    
    def calculate_necessity_score(self,
                                  instance: np.ndarray,
                                  feature_idx: int,
                                  X_context: np.ndarray,
                                  original_prediction: int) -> float:
        """
        Calculate necessity score for a feature.
        
        Necessity: Probability that changing the feature changes the prediction.
        
        Args:
            instance: Single data instance
            feature_idx: Index of feature to evaluate
            X_context: Context data for sampling
            original_prediction: Original model prediction
            
        Returns:
            Necessity score (0 to 1)
        """
        counterfactuals = self.generate_counterfactuals(instance, feature_idx, X_context)
        
        if len(counterfactuals) == 0:
            return 0.0
        
        # Predict on counterfactuals
        cf_predictions = self.model.predict(counterfactuals)
        
        # Count how many predictions changed
        changed = np.sum(cf_predictions != original_prediction)
        
        necessity_score = changed / len(counterfactuals)
        
        return necessity_score
    
    def calculate_sufficiency_score(self,
                                   reference_instance: np.ndarray,
                                   feature_idx: int,
                                   X_opposite_class: np.ndarray,
                                   target_prediction: int) -> float:
        """
        Calculate sufficiency score for a feature.
        
        Sufficiency: Probability that setting the feature to a reference value
        produces the target prediction.
        
        Args:
            reference_instance: Reference instance with desired feature value
            feature_idx: Index of feature to evaluate
            X_opposite_class: Instances with opposite prediction
            target_prediction: Target prediction to achieve
            
        Returns:
            Sufficiency score (0 to 1)
        """
        if len(X_opposite_class) == 0:
            return 0.0
        
        reference_value = reference_instance[feature_idx]
        
        # Intervene on instances from opposite class
        counterfactuals = []
        for instance in X_opposite_class:
            cf = instance.copy()
            cf[feature_idx] = reference_value
            counterfactuals.append(cf)
        
        counterfactuals = np.array(counterfactuals)
        
        # Predict on counterfactuals
        cf_predictions = self.model.predict(counterfactuals)
        
        # Count how many achieved target prediction
        achieved = np.sum(cf_predictions == target_prediction)
        
        sufficiency_score = achieved / len(counterfactuals)
        
        return sufficiency_score


class GlobalScoreCalculator:
    """
    Calculate global necessity and sufficiency scores by aggregating
    local scores across multiple instances.
    """
    
    def __init__(self, cf_generator: ForwardCounterfactualGenerator):
        """
        Initialize global score calculator.
        
        Args:
            cf_generator: ForwardCounterfactualGenerator instance
        """
        self.cf_generator = cf_generator
        
    def calculate_global_necessity(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_idx: int,
                                   n_samples: int = 100) -> float:
        """
        Calculate global necessity score for a feature.
        
        Args:
            X: Feature matrix
            y: True labels (for context)
            feature_idx: Index of feature to evaluate
            n_samples: Number of instances to sample
            
        Returns:
            Global necessity score
        """
        # Sample instances
        n_samples = min(n_samples, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        necessity_scores = []
        
        for idx in tqdm(sample_indices, desc=f"Calculating necessity for feature {feature_idx}"):
            instance = X[idx]
            original_pred = self.cf_generator.model.predict([instance])[0]
            
            score = self.cf_generator.calculate_necessity_score(
                instance, feature_idx, X, original_pred
            )
            necessity_scores.append(score)
        
        return np.mean(necessity_scores)
    
    def calculate_global_sufficiency(self,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    feature_idx: int,
                                    n_samples: int = 100) -> float:
        """
        Calculate global sufficiency score for a feature.
        
        Args:
            X: Feature matrix
            y: True labels (for context)
            feature_idx: Index of feature to evaluate
            n_samples: Number of reference instances to sample
            
        Returns:
            Global sufficiency score
        """
        # Get predictions
        predictions = self.cf_generator.model.predict(X)
        
        # Sample reference instances from each class
        unique_classes = np.unique(predictions)
        
        if len(unique_classes) < 2:
            return 0.0
        
        sufficiency_scores = []
        
        for target_class in unique_classes:
            # Get instances with target class
            target_indices = np.where(predictions == target_class)[0]
            # Get instances with opposite class
            opposite_indices = np.where(predictions != target_class)[0]
            
            if len(target_indices) == 0 or len(opposite_indices) == 0:
                continue
            
            # Sample reference instances
            n_ref = min(n_samples // len(unique_classes), len(target_indices))
            ref_indices = np.random.choice(target_indices, n_ref, replace=False)
            
            for ref_idx in ref_indices:
                reference_instance = X[ref_idx]
                
                # Sample opposite class instances
                n_opp = min(50, len(opposite_indices))
                opp_sample = np.random.choice(opposite_indices, n_opp, replace=False)
                X_opposite = X[opp_sample]
                
                score = self.cf_generator.calculate_sufficiency_score(
                    reference_instance, feature_idx, X_opposite, target_class
                )
                sufficiency_scores.append(score)
        
        return np.mean(sufficiency_scores) if sufficiency_scores else 0.0
    
    def calculate_all_global_scores(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate global necessity and sufficiency scores for all features.
        
        Args:
            X: Feature matrix
            y: True labels
            n_samples: Number of instances to sample per feature
            
        Returns:
            Tuple of (necessity_scores, sufficiency_scores) arrays
        """
        n_features = X.shape[1]
        
        necessity_scores = np.zeros(n_features)
        sufficiency_scores = np.zeros(n_features)
        
        print("Calculating global necessity and sufficiency scores...")
        
        for feature_idx in range(n_features):
            print(f"\nFeature {feature_idx + 1}/{n_features}: {self.cf_generator.feature_names[feature_idx]}")
            
            # Calculate necessity
            necessity_scores[feature_idx] = self.calculate_global_necessity(
                X, y, feature_idx, n_samples
            )
            
            # Calculate sufficiency
            sufficiency_scores[feature_idx] = self.calculate_global_sufficiency(
                X, y, feature_idx, n_samples
            )
            
            print(f"  Necessity: {necessity_scores[feature_idx]:.4f}")
            print(f"  Sufficiency: {sufficiency_scores[feature_idx]:.4f}")
        
        return necessity_scores, sufficiency_scores
