"""
Data Loader Module
Handles loading and initial exploration of raw datasets (jobs and resumes).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading, exploring, and basic validation of job and resume datasets.
    """
    
    def __init__(self, jobs_path: str, resumes_path: str):
        """
        Initialize DataLoader with paths to CSV files.
        
        Args:
            jobs_path (str): Path to jobs CSV file
            resumes_path (str): Path to resumes CSV file
        """
        self.jobs_path = Path(jobs_path)
        self.resumes_path = Path(resumes_path)
        self.jobs_df = None
        self.resumes_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load jobs and resumes CSV files.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (jobs_df, resumes_df)
        """
        logger.info(f"Loading jobs data from: {self.jobs_path}")
        self.jobs_df = pd.read_csv(self.jobs_path)
        logger.info(f"✓ Loaded {len(self.jobs_df)} job records")
        
        logger.info(f"Loading resumes data from: {self.resumes_path}")
        self.resumes_df = pd.read_csv(self.resumes_path)
        logger.info(f"✓ Loaded {len(self.resumes_df)} resume records")
        
        return self.jobs_df, self.resumes_df
    
    def explore_jobs_data(self) -> Dict:
        """
        Perform exploratory analysis on jobs dataset.
        
        Returns:
            Dict: Summary statistics and insights
        """
        logger.info("=" * 80)
        logger.info("JOBS DATASET EXPLORATION")
        logger.info("=" * 80)
        
        summary = {}
        
        # Basic info
        logger.info(f"\n📊 Dataset Shape: {self.jobs_df.shape}")
        summary['shape'] = self.jobs_df.shape
        
        # Column info
        logger.info(f"\n📋 Columns: {list(self.jobs_df.columns)}")
        summary['columns'] = list(self.jobs_df.columns)
        
        # Data types
        logger.info(f"\n📝 Data Types:\n{self.jobs_df.dtypes}")
        summary['dtypes'] = self.jobs_df.dtypes.to_dict()
        
        # Missing values
        missing_counts = self.jobs_df.isnull().sum()
        missing_pct = (missing_counts / len(self.jobs_df)) * 100
        missing_df = pd.DataFrame(missing_values)
        logger.info(f"\n❓ Missing Values:\n{missing_df}"){
            'Column': missing_counts.index,
            'Count': missing_counts.values,
            'Percentage': missing_pct.values
        }
        logger.info(f"\n❓ Missing Values:\n{missing_df}")
        summary['missing'] = missing_counts.to_dict()
        
        # Category distribution
        logger.info(f"\n📂 Job Category Distribution:\n{self.jobs_df['category'].value_counts()}")
        summary['categories'] = self.jobs_df['category'].value_counts().to_dict()
        
        # Sample data
        logger.info(f"\n🔍 First 2 records:")
        logger.info(self.jobs_df.head(2).to_string())
        
        return summary
    
    def explore_resumes_data(self) -> Dict:
        """
        Perform exploratory analysis on resumes dataset.
        
        Returns:
            Dict: Summary statistics and insights
        """
        logger.info("=" * 80)
        logger.info("RESUMES DATASET EXPLORATION")
        logger.info("=" * 80)
        
        summary = {}
        
        # Basic info
        logger.info(f"\n📊 Dataset Shape: {self.resumes_df.shape}")
        summary['shape'] = self.resumes_df.shape
        
        # Column info
        logger.info(f"\n📋 Columns: {list(self.resumes_df.columns)}")
        summary['columns'] = list(self.resumes_df.columns)
        
        # Data types
        logger.info(f"\n📝 Data Types:\n{self.resumes_df.dtypes}")
        summary['dtypes'] = self.resumes_df.dtypes.to_dict()
        
        # Missing values
        missing_counts = self.resumes_df.isnull().sum()
        missing_pct = (missing_counts / len(self.resumes_df)) * 100
        logger.info(f"\n❓ Missing Values:\n{pd.DataFrame({
            'Column': missing_counts.index,
            'Count': missing_counts.values,
            'Percentage': missing_pct.values
        })}")
        summary['missing'] = missing_counts.to_dict()
        
        # Job role distribution
        logger.info(f"\n💼 Job Role Distribution:\n{self.resumes_df['Job Role'].value_counts()}")
        summary['job_roles'] = self.resumes_df['Job Role'].value_counts().to_dict()
        
        # Experience distribution
        logger.info(f"\n📈 Experience Statistics:\n{self.resumes_df['Experience (Years)'].describe()}")
        summary['experience_stats'] = self.resumes_df['Experience (Years)'].describe().to_dict()
        
        # Education distribution
        logger.info(f"\n🎓 Education Distribution:\n{self.resumes_df['Education'].value_counts()}")
        summary['education'] = self.resumes_df['Education'].value_counts().to_dict()
        
        # Salary distribution
        logger.info(f"\n💰 Salary Statistics:\n{self.resumes_df['Salary Expectation ($)'].describe()}")
        summary['salary_stats'] = self.resumes_df['Salary Expectation ($)'].describe().to_dict()
        
        # Sample data
        logger.info(f"\n🔍 First 2 records:")
        logger.info(self.resumes_df.head(2).to_string())
        
        return summary
    
    def get_data_quality_report(self) -> str:
        """
        Generate a comprehensive data quality report.
        
        Returns:
            str: Data quality assessment
        """
        logger.info("=" * 80)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 80)
        
        report = []
        report.append("\n🔍 JOBS DATASET QUALITY CHECKS:")
        report.append(f"  • Total records: {len(self.jobs_df)}")
        report.append(f"  • Duplicate records: {self.jobs_df.duplicated().sum()}")
        report.append(f"  • Columns with missing values: {self.jobs_df.isnull().any().sum()}")
        
        report.append("\n🔍 RESUMES DATASET QUALITY CHECKS:")
        report.append(f"  • Total records: {len(self.resumes_df)}")
        report.append(f"  • Duplicate records: {self.resumes_df.duplicated().sum()}")
        report.append(f"  • Columns with missing values: {self.resumes_df.isnull().any().sum()}")
        
        report_text = "\n".join(report)
        logger.info(report_text)
        return report_text


def load_and_explore_data(jobs_path: str, resumes_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Convenience function to load and explore both datasets.
    
    Args:
        jobs_path (str): Path to jobs CSV
        resumes_path (str): Path to resumes CSV
        
    Returns:
        Tuple containing (jobs_df, resumes_df, jobs_summary, resumes_summary)
    """
    loader = DataLoader(jobs_path, resumes_path)
    jobs_df, resumes_df = loader.load_data()
    
    jobs_summary = loader.explore_jobs_data()
    resumes_summary = loader.explore_resumes_data()
    quality_report = loader.get_data_quality_report()
    
    return jobs_df, resumes_df, jobs_summary, resumes_summary


if __name__ == "__main__":
    # Example usage
    from config.config import JOBS_CSV_PATH, RESUMES_CSV_PATH
    
    # Load and explore data
    jobs_df, resumes_df, jobs_summary, resumes_summary = load_and_explore_data(
        str(JOBS_CSV_PATH),
        str(RESUMES_CSV_PATH)
    )
