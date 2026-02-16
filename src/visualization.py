"""
Visualization Module

Create publication-quality plots for necessity and sufficiency analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os


class NecessitySufficiencyVisualizer:
    """
    Create visualizations for necessity and sufficiency analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-paper'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = {
            'necessity': '#FF6B6B',
            'sufficiency': '#4ECDC4',
            'lime': '#95E1D3',
            'shap': '#F38181'
        }
    
    def plot_global_scores(self,
                          feature_names: List[str],
                          necessity_scores: np.ndarray,
                          sufficiency_scores: np.ndarray,
                          save_path: str = None,
                          top_k: int = 7):
        """
        Plot global necessity and sufficiency scores.
        
        Args:
            feature_names: List of feature names
            necessity_scores: Global necessity scores
            sufficiency_scores: Global sufficiency scores
            save_path: Path to save figure
            top_k: Number of top features to display
        """
        # Get top features by necessity
        top_indices = np.argsort(necessity_scores)[-top_k:][::-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot necessity scores
        y_pos = np.arange(len(top_indices))
        ax1.barh(y_pos, necessity_scores[top_indices], 
                color=self.colors['necessity'], alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([feature_names[i] for i in top_indices])
        ax1.set_xlabel('Necessity Score', fontsize=12)
        ax1.set_title('Global Necessity Scores', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(necessity_scores[top_indices]):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
        
        # Plot sufficiency scores
        ax2.barh(y_pos, sufficiency_scores[top_indices],
                color=self.colors['sufficiency'], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([feature_names[i] for i in top_indices])
        ax2.set_xlabel('Sufficiency Score', fontsize=12)
        ax2.set_title('Global Sufficiency Scores', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(sufficiency_scores[top_indices]):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved global scores plot to {save_path}")
        
        plt.show()
    
    def plot_robustness_analysis(self,
                                robustness_results: Dict,
                                save_path: str = None):
        """
        Plot robustness analysis for LIME and SHAP.
        
        Args:
            robustness_results: Dictionary with robustness evaluation results
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        lime_nec = robustness_results['lime_necessity']
        lime_suf = robustness_results['lime_sufficiency']
        shap_nec = robustness_results['shap_necessity']
        shap_suf = robustness_results['shap_sufficiency']
        
        ranks = list(range(1, len(lime_nec) + 1))
        
        # LIME Necessity
        ax = axes[0, 0]
        values = [lime_nec[f'rank_{r}'] for r in ranks]
        ax.plot(ranks, values, 'o-', linewidth=2, markersize=8, 
               color=self.colors['lime'], label='LIME')
        ax.set_xlabel('Feature Rank', fontsize=12)
        ax.set_ylabel('Necessity Score', fontsize=12)
        ax.set_title('LIME - Necessity Robustness', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ranks)
        
        # Add value labels
        for r, v in zip(ranks, values):
            ax.text(r, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        
        # LIME Sufficiency
        ax = axes[0, 1]
        values = [lime_suf[f'rank_{r}'] for r in ranks]
        ax.plot(ranks, values, 'o-', linewidth=2, markersize=8,
               color=self.colors['lime'], label='LIME')
        ax.set_xlabel('Feature Rank', fontsize=12)
        ax.set_ylabel('Sufficiency Score', fontsize=12)
        ax.set_title('LIME - Sufficiency Robustness', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ranks)
        
        # Add value labels
        for r, v in zip(ranks, values):
            ax.text(r, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        
        # SHAP Necessity
        ax = axes[1, 0]
        values = [shap_nec[f'rank_{r}'] for r in ranks]
        ax.plot(ranks, values, 'o-', linewidth=2, markersize=8,
               color=self.colors['shap'], label='SHAP')
        ax.set_xlabel('Feature Rank', fontsize=12)
        ax.set_ylabel('Necessity Score', fontsize=12)
        ax.set_title('SHAP - Necessity Robustness', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ranks)
        
        # Add value labels
        for r, v in zip(ranks, values):
            ax.text(r, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        
        # SHAP Sufficiency
        ax = axes[1, 1]
        values = [shap_suf[f'rank_{r}'] for r in ranks]
        ax.plot(ranks, values, 'o-', linewidth=2, markersize=8,
               color=self.colors['shap'], label='SHAP')
        ax.set_xlabel('Feature Rank', fontsize=12)
        ax.set_ylabel('Sufficiency Score', fontsize=12)
        ax.set_title('SHAP - Sufficiency Robustness', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ranks)
        
        # Add value labels
        for r, v in zip(ranks, values):
            ax.text(r, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved robustness analysis plot to {save_path}")
        
        plt.show()
    
    def plot_comparison(self,
                       robustness_results: Dict,
                       save_path: str = None):
        """
        Create comparison plot between LIME and SHAP.
        
        Args:
            robustness_results: Dictionary with robustness evaluation results
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        lime_nec = robustness_results['lime_necessity']
        shap_nec = robustness_results['shap_necessity']
        lime_suf = robustness_results['lime_sufficiency']
        shap_suf = robustness_results['shap_sufficiency']
        
        ranks = list(range(1, len(lime_nec) + 1))
        
        # Necessity comparison
        lime_nec_values = [lime_nec[f'rank_{r}'] for r in ranks]
        shap_nec_values = [shap_nec[f'rank_{r}'] for r in ranks]
        
        ax1.plot(ranks, lime_nec_values, 'o-', linewidth=2, markersize=8,
                color=self.colors['lime'], label='LIME')
        ax1.plot(ranks, shap_nec_values, 's-', linewidth=2, markersize=8,
                color=self.colors['shap'], label='SHAP')
        ax1.set_xlabel('Feature Rank', fontsize=12)
        ax1.set_ylabel('Necessity Score', fontsize=12)
        ax1.set_title('Necessity Comparison: LIME vs SHAP', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(ranks)
        
        # Sufficiency comparison
        lime_suf_values = [lime_suf[f'rank_{r}'] for r in ranks]
        shap_suf_values = [shap_suf[f'rank_{r}'] for r in ranks]
        
        ax2.plot(ranks, lime_suf_values, 'o-', linewidth=2, markersize=8,
                color=self.colors['lime'], label='LIME')
        ax2.plot(ranks, shap_suf_values, 's-', linewidth=2, markersize=8,
                color=self.colors['shap'], label='SHAP')
        ax2.set_xlabel('Feature Rank', fontsize=12)
        ax2.set_ylabel('Sufficiency Score', fontsize=12)
        ax2.set_title('Sufficiency Comparison: LIME vs SHAP', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(ranks)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.show()
    
    def create_summary_table(self,
                            robustness_results: Dict,
                            save_path: str = None):
        """
        Create summary table of robustness results.
        
        Args:
            robustness_results: Dictionary with robustness evaluation results
            save_path: Path to save table
        """
        from matplotlib.patches import Rectangle
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        ranks = list(range(1, len(robustness_results['lime_necessity']) + 1))
        
        table_data = []
        table_data.append(['Rank', 'LIME\nNecessity', 'LIME\nSufficiency', 
                          'SHAP\nNecessity', 'SHAP\nSufficiency'])
        
        for rank in ranks:
            row = [
                f'{rank}',
                f"{robustness_results['lime_necessity'][f'rank_{rank}']:.3f}",
                f"{robustness_results['lime_sufficiency'][f'rank_{rank}']:.3f}",
                f"{robustness_results['shap_necessity'][f'rank_{rank}']:.3f}",
                f"{robustness_results['shap_sufficiency'][f'rank_{rank}']:.3f}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
        
        plt.title('Robustness Analysis Summary', fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved summary table to {save_path}")
        
        plt.show()
