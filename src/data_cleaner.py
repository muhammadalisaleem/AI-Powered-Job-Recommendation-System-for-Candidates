"""
Data Cleaning and Preprocessing Module
Handles all data cleaning, validation, and preprocessing operations.
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing for jobs and resumes datasets.
    """
    
    def __init__(self, jobs_df: pd.DataFrame, resumes_df: pd.DataFrame, config: Dict):
        """
        Initialize DataCleaner with dataframes and configuration.
        
        Args:
            jobs_df (pd.DataFrame): Jobs dataset
            resumes_df (pd.DataFrame): Resumes dataset
            config (Dict): Configuration parameters from config.py
        """
        self.jobs_df = jobs_df.copy()
        self.resumes_df = resumes_df.copy()
        self.config = config
        self.cleaning_log = []
        
    def log_cleaning_step(self, step: str, records_before: int, records_after: int):
        """Log each cleaning step."""
        removed = records_before - records_after
        logger.info(f"  [STEP] {step}")
        logger.info(f"    Records: {records_before} → {records_after} (removed: {removed})")
        self.cleaning_log.append({
            'step': step,
            'before': records_before,
            'after': records_after,
            'removed': removed
        })
    
    # ==================== JOBS CLEANING ====================
    
    def clean_jobs_data(self) -> pd.DataFrame:
        """
        Main pipeline for cleaning jobs data.
        
        Returns:
            pd.DataFrame: Cleaned jobs dataframe
        """
        logger.info("\n" + "=" * 80)
        logger.info("🧹 JOBS DATA CLEANING PIPELINE")
        logger.info("=" * 80)
        
        initial_count = len(self.jobs_df)
        logger.info(f"Starting records: {initial_count}")
        
        # Step 1: Remove duplicates
        before = len(self.jobs_df)
        self.jobs_df = self.jobs_df.drop_duplicates(subset=['job_id'], keep='first')
        self.log_cleaning_step("Remove duplicate job IDs", before, len(self.jobs_df))
        
        # Step 2: Filter for tech jobs (INFORMATION-TECHNOLOGY category)
        before = len(self.jobs_df)
        self.jobs_df = self.jobs_df[
            self.jobs_df['category'].isin(self.config['RELEVANT_CATEGORIES'])
        ]
        self.log_cleaning_step(
            f"Filter for tech jobs ({self.config['RELEVANT_CATEGORIES']})",
            before, len(self.jobs_df)
        )
        
        # Step 3: Filter for relevant job titles
        before = len(self.jobs_df)
        self.jobs_df['job_title_lower'] = self.jobs_df['job_title'].str.lower()
        keywords_pattern = '|'.join(self.config['RELEVANT_JOB_KEYWORDS'])
        self.jobs_df = self.jobs_df[
            self.jobs_df['job_title_lower'].str.contains(keywords_pattern, na=False, regex=True)
        ]
        self.jobs_df = self.jobs_df.drop('job_title_lower', axis=1)
        self.log_cleaning_step("Filter for relevant job titles", before, len(self.jobs_df))
        
        # Step 4: Remove rows with critical missing values
        before = len(self.jobs_df)
        required_cols = ['job_id', 'job_title', 'job_description', 'job_skill_set']
        for col in required_cols:
            if col in self.jobs_df.columns:
                self.jobs_df = self.jobs_df[self.jobs_df[col].notna()]
        self.log_cleaning_step("Remove rows with missing critical columns", before, len(self.jobs_df))
        
        # Step 5: Clean and standardize text fields
        self.jobs_df = self._clean_text_fields(self.jobs_df)
        logger.info("  ✓ Clean and standardize text fields")
        
        # Step 6: Extract skills from job_skill_set
        self.jobs_df = self._extract_and_validate_skills(self.jobs_df)
        logger.info("  ✓ Extract and validate skills")
        
        # Step 7: Remove rows with no skills
        before = len(self.jobs_df)
        self.jobs_df = self.jobs_df[self.jobs_df['skills_list'].apply(lambda x: len(x) > 0)]
        self.log_cleaning_step("Remove jobs with no extracted skills", before, len(self.jobs_df))
        
        # Step 8: Final validation
        self.jobs_df = self._validate_jobs(self.jobs_df)
        
        logger.info(f"\n✅ Jobs cleaning complete!")
        logger.info(f"Final record count: {len(self.jobs_df)} (removed: {initial_count - len(self.jobs_df)})")
        
        return self.jobs_df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields."""
        text_cols = ['job_title', 'job_description']
        for col in text_cols:
            if col in df.columns:
                # Remove extra whitespace
                df[col] = df[col].str.strip()
                # Convert to lowercase for processing (keep original for display)
                # Remove special characters but keep alphanumeric and spaces
                df[col] = df[col].str.replace(r'[^a-zA-Z0-9\s\-\.\,\(\)]', '', regex=True)
        return df
    
    def _extract_and_validate_skills(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract skills from job_skill_set column (usually formatted as a list/string)."""
        def extract_skills(skill_str):
            """Extract individual skills from string representation."""
            if pd.isna(skill_str):
                return []
            
            # If it's a string representation of a list, parse it
            if isinstance(skill_str, str):
                # Remove brackets and quotes
                skill_str = skill_str.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                # Split by comma
                skills = [s.strip().lower() for s in skill_str.split(',')]
                # Filter empty strings
                skills = [s for s in skills if s and len(s) > 0]
                return skills
            return []
        
        df['skills_list'] = df['job_skill_set'].apply(extract_skills)
        return df
    
    def _validate_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate jobs and remove problematic records."""
        before = len(df)
        
        # Validate that all records have required fields
        df = df.dropna(subset=['job_id', 'job_title', 'job_description', 'skills_list'])
        
        # Validate descriptions have reasonable length (> 100 chars)
        df = df[df['job_description'].str.len() > 100]
        
        after = len(df)
        if before > after:
            logger.info(f"  ✓ Validation removed {before - after} invalid records")
        
        return df
    
    # ==================== RESUMES CLEANING ====================
    
    def clean_resumes_data(self) -> pd.DataFrame:
        """
        Main pipeline for cleaning resumes data.
        
        Returns:
            pd.DataFrame: Cleaned resumes dataframe
        """
        logger.info("\n" + "=" * 80)
        logger.info("🧹 RESUMES DATA CLEANING PIPELINE")
        logger.info("=" * 80)
        
        initial_count = len(self.resumes_df)
        logger.info(f"Starting records: {initial_count}")
        
        # Step 1: Remove duplicates
        before = len(self.resumes_df)
        self.resumes_df = self.resumes_df.drop_duplicates(subset=['Resume_ID'], keep='first')
        self.log_cleaning_step("Remove duplicate Resume IDs", before, len(self.resumes_df))
        
        # Step 2: Remove rows with missing critical columns
        before = len(self.resumes_df)
        required_cols = ['Resume_ID', 'Skills', 'Experience (Years)']
        for col in required_cols:
            if col in self.resumes_df.columns:
                self.resumes_df = self.resumes_df[self.resumes_df[col].notna()]
        self.log_cleaning_step("Remove rows with missing critical columns", before, len(self.resumes_df))
        
        # Step 3: Clean and validate numeric fields
        self.resumes_df = self._clean_numeric_fields(self.resumes_df)
        logger.info("  ✓ Clean and validate numeric fields")
        
        # Step 4: Extract and validate skills
        self.resumes_df = self._extract_resume_skills(self.resumes_df)
        logger.info("  ✓ Extract and validate resume skills")
        
        # Step 5: Remove records with no skills
        before = len(self.resumes_df)
        self.resumes_df = self.resumes_df[self.resumes_df['skills_list'].apply(lambda x: len(x) > 0)]
        self.log_cleaning_step("Remove resumes with no extracted skills", before, len(self.resumes_df))
        
        # Step 6: Standardize education field
        self.resumes_df = self._standardize_education(self.resumes_df)
        logger.info("  ✓ Standardize education field")
        
        # Step 7: Final validation
        self.resumes_df = self._validate_resumes(self.resumes_df)
        
        logger.info(f"\n✅ Resumes cleaning complete!")
        logger.info(f"Final record count: {len(self.resumes_df)} (removed: {initial_count - len(self.resumes_df)})")
        
        return self.resumes_df
    
    def _clean_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric fields (experience, salary)."""
        # Experience years validation
        if 'Experience (Years)' in df.columns:
            df['Experience (Years)'] = pd.to_numeric(df['Experience (Years)'], errors='coerce')
            # Clamp to reasonable ranges
            df = df[
                (df['Experience (Years)'] >= self.config['MIN_EXPERIENCE_YEARS']) &
                (df['Experience (Years)'] <= self.config['MAX_EXPERIENCE_YEARS'])
            ]
        
        # Salary validation
        if 'Salary Expectation ($)' in df.columns:
            df['Salary Expectation ($)'] = pd.to_numeric(df['Salary Expectation ($)'], errors='coerce')
            df = df[
                (df['Salary Expectation ($)'] >= self.config['MIN_SALARY']) &
                (df['Salary Expectation ($)'] <= self.config['MAX_SALARY'])
            ]
        
        return df
    
    def _extract_resume_skills(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract skills from resume Skills column."""
        def extract_skills(skill_str):
            """Extract individual skills from string."""
            if pd.isna(skill_str):
                return []
            
            if isinstance(skill_str, str):
                # Split by comma or common delimiters
                skills = [s.strip().lower() for s in re.split(r'[,;]', skill_str)]
                # Filter empty strings and very short skills
                skills = [s for s in skills if s and len(s) > 1]
                return skills
            return []
        
        df['skills_list'] = df['Skills'].apply(extract_skills)
        return df
    
    def _standardize_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize education field values."""
        if 'Education' not in df.columns:
            return df
        
        education_mapping = {
            'b.sc': 'B.Sc',
            'b.tech': 'B.Tech',
            'm.tech': 'M.Tech',
            'mba': 'MBA',
            'phd': 'PhD',
        }
        
        def standardize_edu(edu):
            if pd.isna(edu):
                return 'Unknown'
            edu_lower = str(edu).lower().strip()
            for key, value in education_mapping.items():
                if key in edu_lower:
                    return value
            return edu
        
        df['Education'] = df['Education'].apply(standardize_edu)
        return df
    
    def _validate_resumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation for resumes."""
        before = len(df)
        
        # Ensure all required columns exist and have valid data
        df = df.dropna(subset=['Resume_ID', 'Experience (Years)'])
        
        # Remove records with experience years < 0 or unrealistic
        df = df[df['Experience (Years)'] >= 0]
        
        after = len(df)
        if before > after:
            logger.info(f"  ✓ Validation removed {before - after} invalid records")
        
        return df
    
    # ==================== DATASET SPLITTING ====================
    
    def create_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split for the jobs dataset for model evaluation.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        logger.info("\n" + "=" * 80)
        logger.info("CREATING TRAIN/TEST SPLIT")
        logger.info("=" * 80)
        
        train_df, test_df = train_test_split(
            self.jobs_df,
            test_size=self.config['TEST_SIZE'],
            random_state=self.config['RANDOM_SEED']
        )
        
        logger.info(f"Train set: {len(train_df)} records ({len(train_df)/len(self.jobs_df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} records ({len(test_df)/len(self.jobs_df)*100:.1f}%)")
        
        return train_df, test_df
    
    def get_cleaning_summary(self) -> pd.DataFrame:
        """Get summary of all cleaning steps."""
        return pd.DataFrame(self.cleaning_log)


def run_full_pipeline(jobs_df: pd.DataFrame, resumes_df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete cleaning pipeline for both datasets.
    
    Args:
        jobs_df (pd.DataFrame): Raw jobs data
        resumes_df (pd.DataFrame): Raw resumes data
        config (Dict): Configuration dictionary
        
    Returns:
        Tuple containing (cleaned_jobs, cleaned_resumes, train_jobs, test_jobs)
    """
    cleaner = DataCleaner(jobs_df, resumes_df, config)
    
    # Clean both datasets
    cleaned_jobs = cleaner.clean_jobs_data()
    cleaned_resumes = cleaner.clean_resumes_data()
    
    # Create train/test split
    train_jobs, test_jobs = cleaner.create_train_test_split()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 80)
    logger.info(cleaner.get_cleaning_summary().to_string())
    
    return cleaned_jobs, cleaned_resumes, train_jobs, test_jobs


if __name__ == "__main__":
    # Example usage (will be called from main notebook)
    pass
