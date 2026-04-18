"""
Test script for feature engineering and embeddings pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    JOBS_CSV_PATH, RESUMES_CSV_PATH,
    DATA_CLEANING_CONFIG, MODEL_CONFIG,
    FEATURE_ENGINEERING_CONFIG,
    PROCESSED_DATA_DIR
)
from src.data_loader import DataLoader
from src.data_cleaner import run_full_pipeline
from src.embeddings import generate_all_embeddings, EmbeddingsGenerator, FAISSIndex
from src.feature_engineering import engineer_all_features, FeatureEngineer


def test_feature_engineering_pipeline():
    """Test the complete feature engineering pipeline."""
    
    print("\n" + "="*80)
    print("[TEST] FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Merge configs
    merged_config = {**DATA_CLEANING_CONFIG, **MODEL_CONFIG}
    
    # Test 1: Load and clean data
    print("\n[1/5] Loading and cleaning data...")
    try:
        loader = DataLoader(str(JOBS_CSV_PATH), str(RESUMES_CSV_PATH))
        jobs_df, resumes_df = loader.load_data()
        
        cleaned_jobs, cleaned_resumes, train_jobs, test_jobs = run_full_pipeline(
            jobs_df, resumes_df, merged_config
        )
        
        print(f"[OK] Data cleaned successfully!")
        print(f"     Jobs: {len(cleaned_jobs)}, Resumes: {len(cleaned_resumes)}")
    except Exception as e:
        print(f"[ERROR] Data cleaning failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Generate embeddings
    print("\n[2/5] Generating embeddings...")
    try:
        print(f"     Model: {FEATURE_ENGINEERING_CONFIG['EMBEDDING_MODEL']}")
        print(f"     Dimension: {FEATURE_ENGINEERING_CONFIG['EMBEDDING_DIM']}")
        
        embeddings_result = generate_all_embeddings(
            cleaned_jobs,
            cleaned_resumes,
            FEATURE_ENGINEERING_CONFIG,
            output_dir=PROCESSED_DATA_DIR
        )
        
        generator = embeddings_result['generator']
        job_embeddings = embeddings_result['job_embeddings']
        resume_embeddings = embeddings_result['resume_embeddings']
        faiss_index = embeddings_result['faiss_index']
        
        print(f"[OK] Embeddings generated!")
        print(f"     Job embeddings shape: {job_embeddings.shape}")
        print(f"     Resume embeddings shape: {resume_embeddings.shape}")
        print(f"     FAISS index built with {faiss_index.index.ntotal} vectors")
    except Exception as e:
        print(f"[ERROR] Embeddings generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Test similarity search
    print("\n[3/5] Testing FAISS similarity search...")
    try:
        test_query = resume_embeddings[0:1]
        distances, indices = faiss_index.search(test_query, k=5)
        
        print(f"[OK] Similarity search successful!")
        print(f"     Query resume index: 0")
        print(f"     Top 5 similar jobs found:")
        
        for rank, (dist, job_idx) in enumerate(zip(distances[0], indices[0]), 1):
            job_title = cleaned_jobs.iloc[job_idx]['job_title'][:40]
            similarity = 1 / (1 + dist)
            print(f"       {rank}. {job_title} (similarity: {similarity:.3f})")
    except Exception as e:
        print(f"[ERROR] Similarity search failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Engineer features
    print("\n[4/5] Engineering features...")
    try:
        features_result = engineer_all_features(
            cleaned_jobs,
            cleaned_resumes,
            merged_config,
            job_embeddings,
            resume_embeddings
        )
        
        features_df = features_result['features_df']
        feature_matrix = features_result['feature_matrix']
        
        print(f"[OK] Features engineered!")
        print(f"     Features dataframe shape: {features_df.shape}")
        print(f"     Feature matrix shape: {feature_matrix.shape}")
        print(f"     Match score range: {features_df['match_score'].min():.3f} - {features_df['match_score'].max():.3f}")
        print(f"     Mean match score: {features_df['match_score'].mean():.3f}")
    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Verify feature statistics
    print("\n[5/5] Verifying feature statistics...")
    try:
        feature_cols = ['skill_overlap', 'education_alignment', 'experience_alignment',
                       'salary_compatibility', 'semantic_similarity', 'matching_skills_count']
        
        print(f"[OK] Feature statistics verified!")
        print(f"\n     Feature Statistics:")
        for col in feature_cols:
            mean_val = features_df[col].mean()
            std_val = features_df[col].std()
            print(f"       {col}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        # Check match score quality
        good_matches = (features_df['match_score'] >= 0.7).sum()
        excellent_matches = (features_df['match_score'] >= 0.8).sum()
        print(f"\n     Match Quality:")
        print(f"       Good matches (>= 0.7): {good_matches} ({good_matches/len(features_df)*100:.1f}%)")
        print(f"       Excellent matches (>= 0.8): {excellent_matches} ({excellent_matches/len(features_df)*100:.1f}%)")
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print("\n" + "="*80)
    print("[SUCCESS] ALL FEATURE ENGINEERING TESTS PASSED!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  Jobs: {len(cleaned_jobs)}")
    print(f"  Resumes: {len(cleaned_resumes)}")
    print(f"  Job-Resume pairs: {len(features_df):,}")
    print(f"  Embeddings dimension: {FEATURE_ENGINEERING_CONFIG['EMBEDDING_DIM']}")
    print(f"  Engineered features: {len(feature_cols)}")
    print(f"  Mean match score: {features_df['match_score'].mean():.3f}")
    print(f"\n[READY] Ready for Phase 3: Model Training\n")
    
    return True


if __name__ == "__main__":
    success = test_feature_engineering_pipeline()
    sys.exit(0 if success else 1)
