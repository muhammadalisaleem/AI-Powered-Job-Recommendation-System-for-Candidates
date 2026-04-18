"""
Standalone test script for data loading and cleaning.
Run this to quickly verify the data pipeline works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    JOBS_CSV_PATH, RESUMES_CSV_PATH,
    DATA_CLEANING_CONFIG, MODEL_CONFIG
)
from src.data_loader import DataLoader, load_and_explore_data
from src.data_cleaner import DataCleaner, run_full_pipeline


def test_data_pipeline():
    """Test the complete data loading and cleaning pipeline."""
    
    print("\n" + "="*80)
    print("[TEST] TESTING DATA PIPELINE")
    print("="*80)
    
    # Merge configs for data cleaner
    merged_config = {**DATA_CLEANING_CONFIG, **MODEL_CONFIG}
    
    # Test 1: Load data
    print("\n[1/4] Testing data loading...")
    try:
        jobs_df, resumes_df, jobs_summary, resumes_summary = load_and_explore_data(
            str(JOBS_CSV_PATH),
            str(RESUMES_CSV_PATH)
        )
        print(f"[OK] Data loaded successfully!")
        print(f"   Jobs: {len(jobs_df)} records")
        print(f"   Resumes: {len(resumes_df)} records")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return False
    
    # Test 2: Clean data
    print("\n[2/4] Testing data cleaning pipeline...")
    try:
        cleaned_jobs, cleaned_resumes, train_jobs, test_jobs = run_full_pipeline(
            jobs_df,
            resumes_df,
            merged_config
        )
        print(f"[OK] Data cleaning completed!")
        print(f"   Cleaned jobs: {len(cleaned_jobs)} records ({len(cleaned_jobs)/len(jobs_df)*100:.1f}%)")
        print(f"   Cleaned resumes: {len(cleaned_resumes)} records ({len(cleaned_resumes)/len(resumes_df)*100:.1f}%)")
        print(f"   Train set: {len(train_jobs)} records")
        print(f"   Test set: {len(test_jobs)} records")
    except Exception as e:
        print(f"[ERROR] Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Verify data quality
    print("\n[3/4] Testing data quality checks...")
    try:
        # Check for required columns
        required_job_cols = ['job_id', 'job_title', 'job_description', 'skills_list']
        for col in required_job_cols:
            assert col in cleaned_jobs.columns, f"Missing column: {col}"
        
        required_resume_cols = ['Resume_ID', 'Skills', 'Experience (Years)', 'skills_list']
        for col in required_resume_cols:
            assert col in cleaned_resumes.columns, f"Missing column: {col}"
        
        # Check for no missing values in critical columns
        assert cleaned_jobs['job_id'].notna().all(), "Missing values in job_id"
        assert cleaned_resumes['Resume_ID'].notna().all(), "Missing values in Resume_ID"
        
        print(f"[OK] Data quality checks passed!")
        print(f"   [CHECK] All required columns present")
        print(f"   [CHECK] No missing values in critical columns")
        print(f"   [CHECK] Skills extracted for jobs: {cleaned_jobs['skills_list'].apply(len).mean():.1f} avg skills/job")
        print(f"   [CHECK] Skills extracted for resumes: {cleaned_resumes['skills_list'].apply(len).mean():.1f} avg skills/resume")
    except AssertionError as e:
        print(f"[ERROR] Quality check failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Display sample records
    print("\n[4/4] Displaying sample records...")
    try:
        print("\n[SAMPLE] Job (first record):")
        job = cleaned_jobs.iloc[0]
        print(f"   ID: {job['job_id']}")
        print(f"   Title: {job['job_title']}")
        print(f"   Skills: {job['skills_list'][:3]}... ({len(job['skills_list'])} total)")
        
        print("\n[SAMPLE] Resume (first record):")
        resume = cleaned_resumes.iloc[0]
        print(f"   ID: {resume['Resume_ID']}")
        print(f"   Name: {resume['Name']}")
        print(f"   Experience: {resume['Experience (Years)']} years")
        print(f"   Skills: {resume['skills_list'][:3]}... ({len(resume['skills_list'])} total)")
        
        print("\n[OK] Sample records displayed successfully!")
    except Exception as e:
        print(f"[ERROR] Error displaying samples: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print("\n" + "="*80)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*80)
    print(f"\n[SUMMARY]:")
    print(f"   • Raw jobs: {len(jobs_df)} → Cleaned: {len(cleaned_jobs)} ({len(cleaned_jobs)/len(jobs_df)*100:.1f}%)")
    print(f"   • Raw resumes: {len(resumes_df)} → Cleaned: {len(cleaned_resumes)} ({len(cleaned_resumes)/len(resumes_df)*100:.1f}%)")
    print(f"   • Train/test split: {len(train_jobs)}/{len(test_jobs)}")
    print(f"\n[READY] Ready for feature engineering and model training!")
    
    return True


if __name__ == "__main__":
    success = test_data_pipeline()
    sys.exit(0 if success else 1)
