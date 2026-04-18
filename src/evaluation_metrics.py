"""
Evaluation Metrics for Job-Resume Matching Models
Implements metrics: R2, RMSE, Precision@K, NDCG, MAE, Mean Percentile Rank
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for ranking and regression models.
    Suitable for job-resume matching prediction tasks.
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: True match scores (0-1)
            y_pred: Predicted match scores (0-1)
            
        Returns:
            RMSE value
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared (coefficient of determination)."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5, threshold: float = 0.6) -> float:
        """
        Calculate Precision@K for top-k recommendations.
        
        A prediction is considered correct if:
        1. It's in the top-k highest predictions
        2. Its true value >= threshold (indicating good match)
        
        Args:
            y_true: True match scores (binary or continuous)
            y_pred: Predicted match scores
            k: Number of top predictions to consider
            threshold: Score threshold for considering a match as positive
            
        Returns:
            Precision@K value (0-1)
        """
        if len(y_pred) < k:
            k = len(y_pred)
        
        # Get indices of top-k predictions
        top_k_indices = np.argsort(y_pred)[-k:]
        
        # Count how many of top-k predictions are correct
        correct = np.sum(y_true[top_k_indices] >= threshold)
        
        return correct / k
    
    @staticmethod
    def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG measures ranking quality: higher-ranked items should have higher relevance scores.
        Formula: NDCG = DCG@K / IDCG@K
        
        Args:
            y_true: True match scores (relevance labels)
            y_pred: Predicted match scores
            k: Number of top predictions to consider
            
        Returns:
            NDCG@K value (0-1)
        """
        if len(y_pred) < k:
            k = len(y_pred)
        
        # Get indices of top-k predictions
        top_k_indices = np.argsort(y_pred)[-k:][::-1]  # Sort descending
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            # Discount factor starts at 1 for position 0
            discount = np.log2(i + 2)  # log2(i+2) to avoid log(1)=0
            dcg += y_true[idx] / discount
        
        # Calculate IDCG (Ideal DCG: sorted ground truth)
        ideal_order = np.argsort(y_true)[-k:][::-1]
        idcg = 0.0
        for i, idx in enumerate(ideal_order):
            discount = np.log2(i + 2)
            idcg += y_true[idx] / discount
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return min(1.0, max(0.0, ndcg))  # Clamp to [0, 1]
    
    @staticmethod
    def mean_percentile_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Percentile Rank.
        
        For each sample, calculates the percentile rank of its true value
        in the distribution of predictions.
        
        Args:
            y_true: True match scores
            y_pred: Predicted match scores
            
        Returns:
            Mean percentile rank (0-100)
        """
        percentile_ranks = []
        
        for true_val in y_true:
            # Percentile rank = (number of values <= true_val) / total values * 100
            rank = (np.sum(y_pred <= true_val) / len(y_pred)) * 100
            percentile_ranks.append(rank)
        
        return np.mean(percentile_ranks)
    
    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5, threshold: float = 0.6) -> float:
        """
        Calculate Recall@K.
        
        Recall@K = (# of relevant items in top-k) / (total # of relevant items)
        
        Args:
            y_true: True match scores (binary or continuous)
            y_pred: Predicted match scores
            k: Number of top predictions
            threshold: Score threshold for considering a match as positive
            
        Returns:
            Recall@K value (0-1)
        """
        if len(y_pred) < k:
            k = len(y_pred)
        
        # Total number of relevant items
        total_relevant = np.sum(y_true >= threshold)
        
        if total_relevant == 0:
            return 0.0
        
        # Get indices of top-k predictions
        top_k_indices = np.argsort(y_pred)[-k:]
        
        # Count relevant items in top-k
        relevant_in_top_k = np.sum(y_true[top_k_indices] >= threshold)
        
        return relevant_in_top_k / total_relevant
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           k_values: List[int] = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True match scores
            y_pred: Predicted match scores
            k_values: List of k values for Precision@K, NDCG@K, Recall@K
            
        Returns:
            Dictionary of all metrics
        """
        if k_values is None:
            k_values = [5, 10]
        
        metrics = {
            'r2': EvaluationMetrics.r2(y_true, y_pred),
            'rmse': EvaluationMetrics.rmse(y_true, y_pred),
            'mae': EvaluationMetrics.mae(y_true, y_pred),
            'mean_percentile_rank': EvaluationMetrics.mean_percentile_rank(y_true, y_pred),
        }
        
        for k in k_values:
            if k <= len(y_pred):
                metrics[f'precision@{k}'] = EvaluationMetrics.precision_at_k(y_true, y_pred, k)
                metrics[f'recall@{k}'] = EvaluationMetrics.recall_at_k(y_true, y_pred, k)
                metrics[f'ndcg@{k}'] = EvaluationMetrics.ndcg_at_k(y_true, y_pred, k)
        
        return metrics


class MetricsReporter:
    """Generate formatted reports of evaluation metrics."""
    
    @staticmethod
    def format_metrics_table(metrics_dict: Dict[str, float], model_name: str = "Model") -> str:
        """Format metrics dictionary as a readable table."""
        report = f"\n{'='*70}\n"
        report += f"{model_name.upper()} - EVALUATION METRICS\n"
        report += f"{'='*70}\n"
        report += f"{'Metric':<30} {'Value':>15}\n"
        report += f"{'-'*70}\n"
        
        for metric_name, value in metrics_dict.items():
            if isinstance(value, float):
                report += f"{metric_name:<30} {value:>15.4f}\n"
            else:
                report += f"{metric_name:<30} {str(value):>15}\n"
        
        report += f"{'='*70}\n"
        return report
    
    @staticmethod
    def compare_models(models_metrics: Dict[str, Dict[str, float]]) -> str:
        """Compare metrics across multiple models."""
        df = pd.DataFrame(models_metrics).T
        report = f"\n{'='*80}\n"
        report += f"MODEL COMPARISON\n"
        report += f"{'='*80}\n"
        report += df.to_string()
        report += f"\n{'='*80}\n"
        return report


def log_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Log formatted metrics to logger."""
    logger.info(MetricsReporter.format_metrics_table(metrics, model_name))
