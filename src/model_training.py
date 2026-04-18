"""
Model Training Module for Job-Resume Matching
Trains and evaluates Random Forest and XGBoost models
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.evaluation_metrics import EvaluationMetrics, MetricsReporter, log_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_jobs': -1,  # Use all processors
}

# Random Forest Hyperparameters
RF_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# XGBoost Hyperparameters
XGB_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Grid Search Configuration
GRID_SEARCH_CONFIG = {
    'cv': MODEL_CONFIG['cv_folds'],
    'scoring': 'r2',
    'n_jobs': MODEL_CONFIG['n_jobs'],
    'verbose': 1,
}


class ModelTrainer:
    """Train and evaluate job-resume matching models."""
    
    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        """Initialize model trainer."""
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}
        self.predictions = {}
        self.metrics = {}
        logger.info("ModelTrainer initialized")
    
    def prepare_features(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for training.
        
        Args:
            features_df: DataFrame with engineered features and match_score
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("=" * 80)
        logger.info("PREPARING FEATURES FOR TRAINING")
        logger.info("=" * 80)
        
        # Extract features and target
        feature_cols = [col for col in features_df.columns if col != 'match_score']
        X = features_df[feature_cols].values
        y = features_df['match_score'].values
        
        logger.info(f"Feature columns: {feature_cols}")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Feature dimensionality: {X.shape[1]}")
        logger.info(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=self.random_state
        )
        
        logger.info(f"\nTrain set size: {len(X_train)} ({100*(1-MODEL_CONFIG['test_size']):.1f}%)")
        logger.info(f"Test set size: {len(X_test)} ({100*MODEL_CONFIG['test_size']:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           hyperparameter_tuning: bool = True) -> RandomForestRegressor:
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_tuning: Whether to perform GridSearchCV
            
        Returns:
            Trained Random Forest model
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RANDOM FOREST MODEL")
        logger.info("=" * 80)
        
        if hyperparameter_tuning:
            logger.info("Performing Hyperparameter Tuning with GridSearchCV...")
            logger.info(f"Hyperparameter space:")
            for param, values in RF_PARAMS.items():
                logger.info(f"  {param}: {values}")
            
            base_rf = RandomForestRegressor(random_state=self.random_state, n_jobs=MODEL_CONFIG['n_jobs'])
            grid_search = GridSearchCV(
                base_rf,
                RF_PARAMS,
                **GRID_SEARCH_CONFIG
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV R² score: {grid_search.best_score_:.4f}")
            
            self.best_models['random_forest'] = {
                'params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
            }
            
            rf_model = grid_search.best_estimator_
        else:
            logger.info("Training Random Forest with default parameters...")
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=MODEL_CONFIG['n_jobs']
            )
            rf_model.fit(X_train, y_train)
            logger.info("Random Forest model trained")
        
        self.models['random_forest'] = rf_model
        return rf_model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      hyperparameter_tuning: bool = True) -> xgb.XGBRegressor:
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_tuning: Whether to perform GridSearchCV
            
        Returns:
            Trained XGBoost model
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING XGBOOST MODEL")
        logger.info("=" * 80)
        
        if hyperparameter_tuning:
            logger.info("Performing Hyperparameter Tuning with GridSearchCV...")
            logger.info(f"Hyperparameter space:")
            for param, values in XGB_PARAMS.items():
                logger.info(f"  {param}: {values}")
            
            base_xgb = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=MODEL_CONFIG['n_jobs'],
                verbosity=0
            )
            grid_search = GridSearchCV(
                base_xgb,
                XGB_PARAMS,
                **GRID_SEARCH_CONFIG
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV R² score: {grid_search.best_score_:.4f}")
            
            self.best_models['xgboost'] = {
                'params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
            }
            
            xgb_model = grid_search.best_estimator_
        else:
            logger.info("Training XGBoost with default parameters...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=MODEL_CONFIG['n_jobs'],
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            logger.info("XGBoost model trained")
        
        self.models['xgboost'] = xgb_model
        return xgb_model
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING {model_name.upper()}")
        logger.info(f"{'='*80}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        self.predictions[model_name] = {'y_test': y_test, 'y_pred': y_pred}
        
        # Compute metrics
        metrics = EvaluationMetrics.compute_all_metrics(y_test, y_pred)
        self.metrics[model_name] = metrics
        
        log_metrics(metrics, model_name)
        
        return metrics
    
    def cross_validate_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                            model_name: str) -> Tuple[np.ndarray, float]:
        """
        Perform cross-validation on training set.
        
        Args:
            model: Model to validate
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            CV scores and mean CV score
        """
        logger.info(f"\nPerforming {self.cv_folds}-Fold Cross-Validation on {model_name}...")
        
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=self.cv_folds,
            scoring='r2',
            n_jobs=MODEL_CONFIG['n_jobs']
        )
        
        self.cv_scores[model_name] = cv_scores
        
        logger.info(f"Cross-validation R² scores: {cv_scores}")
        logger.info(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return cv_scores, cv_scores.mean()
    
    def get_feature_importance(self, model: Any, feature_names: list, 
                              model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Extract and display feature importance.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of model
            top_n: Number of top features to display
            
        Returns:
            DataFrame of feature importances
        """
        logger.info(f"\n{model_name.upper()} - TOP {top_n} FEATURE IMPORTANCES")
        logger.info("-" * 60)
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(feature_importance_df.head(top_n).to_string(index=False))
        
        return feature_importance_df
    
    def save_model(self, model: Any, model_name: str, output_dir: str = "models"):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name of model
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / f"{model_name}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Any:
        """Load trained model from disk."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from: {model_path}")
        return model
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive training summary report."""
        report = f"\n{'='*80}\n"
        report += "PHASE 3: MODEL TRAINING SUMMARY\n"
        report += f"{'='*80}\n\n"
        
        # Model comparison
        if len(self.metrics) > 1:
            report += MetricsReporter.compare_models(self.metrics)
        
        # Best model determination
        best_model_name = max(self.metrics.keys(), key=lambda x: self.metrics[x].get('r2', -1))
        report += f"\nBest Model: {best_model_name.upper()}\n"
        report += f"R² Score: {self.metrics[best_model_name].get('r2', 'N/A'):.4f}\n"
        
        report += f"\n{'='*80}\n"
        return report


def train_all_models(features_df: pd.DataFrame, hyperparameter_tuning: bool = True) -> Dict:
    """
    Train all models end-to-end.
    
    Args:
        features_df: DataFrame with engineered features
        hyperparameter_tuning: Whether to perform GridSearchCV
        
    Returns:
        Dictionary with all training results
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    trainer = ModelTrainer(random_state=MODEL_CONFIG['random_state'], 
                          cv_folds=MODEL_CONFIG['cv_folds'])
    
    # Prepare features
    X_train, X_test, y_train, y_test = trainer.prepare_features(features_df)
    
    # Get feature names
    feature_cols = [col for col in features_df.columns if col != 'match_score']
    
    # Train Random Forest
    rf_model = trainer.train_random_forest(X_train, y_train, hyperparameter_tuning)
    trainer.cross_validate_model(rf_model, X_train, y_train, 'random_forest')
    trainer.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    trainer.get_feature_importance(rf_model, feature_cols, 'random_forest')
    
    # Train XGBoost
    xgb_model = trainer.train_xgboost(X_train, y_train, hyperparameter_tuning)
    trainer.cross_validate_model(xgb_model, X_train, y_train, 'xgboost')
    trainer.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
    trainer.get_feature_importance(xgb_model, feature_cols, 'xgboost')
    
    # Display summary
    logger.info(trainer.generate_summary_report())
    
    # Save best model
    best_model_name = max(trainer.metrics.keys(), key=lambda x: trainer.metrics[x].get('r2', -1))
    best_model = trainer.models[best_model_name]
    trainer.save_model(best_model, f"best_{best_model_name}")
    
    return {
        'trainer': trainer,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'best_model_name': best_model_name,
    }
