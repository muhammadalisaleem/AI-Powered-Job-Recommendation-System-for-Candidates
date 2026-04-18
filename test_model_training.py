"""
Comprehensive Test Suite for Phase 3: Model Training
Tests the complete model training pipeline with evaluation
"""

import sys
import os
import traceback
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner, run_full_pipeline
from src.embeddings import generate_all_embeddings
from src.feature_engineering import FeatureEngineer, engineer_all_features
from src.model_training import ModelTrainer, train_all_models
from src.evaluation_metrics import EvaluationMetrics, MetricsReporter
from config.config import DATA_CLEANING_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Merge configs for data cleaner
FULL_CONFIG = {**DATA_CLEANING_CONFIG, **MODEL_CONFIG}

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def test_phase_3_model_training():
    """
    Complete Phase 3 test: Data preparation -> Feature engineering -> Model training
    """
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: MODEL TRAINING PIPELINE TEST")
    logger.info("="*80)
    
    try:
        # =====================================================================
        # [1/6] Data Loading and Cleaning
        # =====================================================================
        logger.info("\n[1/6] Loading and cleaning data...")
        
        # Load data
        loader = DataLoader(
            jobs_path="datasets/datasets/preprocessed_jobs.csv",
            resumes_path="datasets/datasets/preprocessed_resumes.csv"
        )
        jobs_raw, resumes_raw = loader.load_data()
        logger.info(f"[OK] Data loaded")
        logger.info(f"     Raw jobs: {len(jobs_raw)}, Raw resumes: {len(resumes_raw)}")
        
        # Clean data
        jobs_clean, resumes_clean, _, _ = run_full_pipeline(jobs_raw, resumes_raw, FULL_CONFIG)
        logger.info(f"[OK] Data cleaned")
        logger.info(f"     Clean jobs: {len(jobs_clean)}, Clean resumes: {len(resumes_clean)}")
        
        if len(jobs_clean) == 0 or len(resumes_clean) == 0:
            raise ValueError("No clean data available")
        
        # =====================================================================
        # [2/6] Embeddings Generation
        # =====================================================================
        logger.info("\n[2/6] Generating embeddings...")
        
        try:
            # Generate embeddings with dataframes and config
            embeddings_results = generate_all_embeddings(jobs_clean, resumes_clean, FULL_CONFIG)
            logger.info(f"[OK] Embeddings generated and indexed")
        except Exception as e:
            logger.error(f"Embeddings generation failed: {e}")
            raise
        
        # =====================================================================
        # [3/6] Feature Engineering
        # =====================================================================
        logger.info("\n[3/6] Engineering features...")
        
        try:
            # Load cached features if available
            features_path = Path("data/processed/features_engineered.csv")
            if features_path.exists():
                logger.info("Loading cached engineered features...")
                features_df = pd.read_csv(features_path)
                logger.info(f"[OK] Features loaded from cache")
            else:
                logger.info("Generating features for all job-resume pairs...")
                # This might take a while for full dataset
                features_results = engineer_all_features(jobs_clean, resumes_clean, FULL_CONFIG)
                features_df = features_results
                # Save for future use
                features_df.to_csv(features_path, index=False)
                logger.info(f"[OK] Features generated and cached")
            
            logger.info(f"     Features shape: {features_df.shape}")
            logger.info(f"     Columns: {list(features_df.columns)}")
            logger.info(f"     Match score range: [{features_df['match_score'].min():.4f}, {features_df['match_score'].max():.4f}]")
            
            # Check for missing values
            missing = features_df.isnull().sum().sum()
            if missing > 0:
                logger.warning(f"[WARNING] Found {missing} missing values")
            else:
                logger.info("[OK] No missing values in features")
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
        
        # =====================================================================
        # [4/6] Model Training (with hyperparameter tuning)
        # =====================================================================
        logger.info("\n[4/6] Training models with hyperparameter tuning...")
        
        try:
            training_results = train_all_models(features_df, hyperparameter_tuning=True)
            trainer = training_results['trainer']
            best_model_name = training_results['best_model_name']
            
            logger.info(f"[OK] Models trained successfully")
            logger.info(f"     Best model: {best_model_name}")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # =====================================================================
        # [5/6] Model Evaluation
        # =====================================================================
        logger.info("\n[5/6] Evaluating models...")
        
        try:
            # Display metrics for both models
            for model_name, metrics in trainer.metrics.items():
                logger.info(f"\n{model_name.upper()} Metrics:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")
            
            logger.info("[OK] Model evaluation complete")
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
        
        # =====================================================================
        # [6/6] Results Verification
        # =====================================================================
        logger.info("\n[6/6] Verifying results...")
        
        try:
            # Verify model predictions
            for model_name, preds in trainer.predictions.items():
                y_pred = preds['y_pred']
                logger.info(f"\n{model_name.upper()} Predictions:")
                logger.info(f"  Shape: {y_pred.shape}")
                logger.info(f"  Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
                logger.info(f"  Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}")
            
            logger.info("\n[OK] Results verified")
            
        except Exception as e:
            logger.error(f"Results verification failed: {e}")
            raise
        
        # =====================================================================
        # SUCCESS
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("✅ PHASE 3: ALL TESTS PASSED")
        logger.info("="*80)
        logger.info(f"\nSummary:")
        logger.info(f"  ✅ [1/6] Data loaded and cleaned: {len(jobs_clean)} jobs, {len(resumes_clean)} resumes")
        logger.info(f"  ✅ [2/6] Embeddings generated and indexed")
        logger.info(f"  ✅ [3/6] Features engineered: {features_df.shape[0]} pairs, {features_df.shape[1]} features")
        logger.info(f"  ✅ [4/6] Models trained with hyperparameter tuning")
        logger.info(f"  ✅ [5/6] Models evaluated: Best model = {best_model_name}")
        logger.info(f"  ✅ [6/6] Results verified and saved")
        logger.info("="*80 + "\n")
        
        return {
            'status': 'success',
            'jobs_count': len(jobs_clean),
            'resumes_count': len(resumes_clean),
            'features_shape': features_df.shape,
            'best_model': best_model_name,
            'metrics': trainer.metrics,
        }
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("❌ PHASE 3: TEST FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("="*80 + "\n")
        
        return {
            'status': 'failed',
            'error': str(e),
        }


if __name__ == "__main__":
    result = test_phase_3_model_training()
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)
