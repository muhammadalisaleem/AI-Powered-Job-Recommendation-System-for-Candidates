"""
Feature Engineering Module
Extracts and engineers features for job-resume matching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature extraction and engineering for job-resume matching.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer.
        
        Args:
            config (Dict): Configuration dictionary with feature parameters
        """
        self.config = config
        self.scaler = StandardScaler()
        logger.info("FeatureEngineer initialized")
    
    def extract_skill_overlap(self, job_skills: List[str], 
                            resume_skills: List[str]) -> float:
        """
        Calculate skill overlap percentage between job and resume.
        
        Args:
            job_skills (List[str]): Skills required for job
            resume_skills (List[str]): Skills in resume
            
        Returns:
            float: Overlap percentage (0.0 to 1.0)
        """
        if not job_skills or not resume_skills:
            return 0.0
        
        job_skills_set = set(str(s).lower().strip() for s in job_skills)
        resume_skills_set = set(str(s).lower().strip() for s in resume_skills)
        
        overlap = len(job_skills_set & resume_skills_set)
        return overlap / len(job_skills_set) if job_skills_set else 0.0
    
    def calculate_education_score(self, required_level: str, 
                                 resume_level: str) -> float:
        """
        Calculate education alignment score.
        
        Args:
            required_level (str): Required education level (B.Sc, M.Tech, MBA, PhD)
            resume_level (str): Resume education level
            
        Returns:
            float: Score (0.0 to 1.0)
        """
        education_hierarchy = {
            'High School': 1,
            'Bachelor': 2,
            'B.Sc': 2,
            'Master': 3,
            'M.Tech': 3,
            'MBA': 3,
            'PhD': 4,
            'Doctor': 4
        }
        
        # Default to Bachelor if not found
        required_score = education_hierarchy.get(
            str(required_level).strip(), 2
        )
        resume_score = education_hierarchy.get(
            str(resume_level).strip(), 2
        )
        
        # Score: 1.0 if equal or better, otherwise proportional
        if resume_score >= required_score:
            return 1.0
        else:
            return resume_score / required_score if required_score > 0 else 0.5
    
    def calculate_experience_alignment(self, required_years: float,
                                      resume_years: float) -> float:
        """
        Calculate experience alignment score.
        
        Args:
            required_years (float): Years of experience required
            resume_years (float): Years of experience in resume
            
        Returns:
            float: Score (0.0 to 1.0)
        """
        if required_years <= 0:
            return 0.5
        
        # Ideal experience is the required years
        # More flexible: accept 0.5x to 2x required
        ratio = resume_years / required_years if required_years > 0 else 1.0
        
        if 0.5 <= ratio <= 2.0:
            # Linear score between 0.7 (at 0.5x or 2x) and 1.0 (exact match)
            if ratio < 1.0:
                return 0.7 + (1.0 - ratio) * 0.3
            else:
                return 1.0 - (ratio - 1.0) * 0.15
        elif ratio < 0.5:
            return 0.4  # Too inexperienced
        else:  # ratio > 2.0
            return 0.8  # Overqualified but acceptable
    
    def calculate_salary_compatibility(self, job_salary: float,
                                      resume_salary: float) -> float:
        """
        Calculate salary compatibility score.
        
        Args:
            job_salary (float): Job salary offered
            resume_salary (float): Resume salary expectation
            
        Returns:
            float: Score (0.0 to 1.0)
        """
        if job_salary <= 0 or resume_salary <= 0:
            return 0.5  # Unknown compatibility
        
        ratio = resume_salary / job_salary
        
        if ratio <= 1.0:
            return 1.0  # Job pays more than expected
        elif ratio <= 1.2:
            return 0.9  # Job pays 10-20% less
        elif ratio <= 1.5:
            return 0.7  # Job pays 20-50% less
        else:
            return 0.5  # Job pays significantly less
    
    def engineer_features_for_pair(self, job_row: pd.Series,
                                  resume_row: pd.Series,
                                  job_embedding: np.ndarray = None,
                                  resume_embedding: np.ndarray = None) -> Dict:
        """
        Engineer features for a single job-resume pair.
        
        Args:
            job_row (pd.Series): Job data row
            resume_row (pd.Series): Resume data row
            job_embedding (np.ndarray): Job embedding vector
            resume_embedding (np.ndarray): Resume embedding vector
            
        Returns:
            Dict: Feature dictionary
        """
        features = {}
        
        # Skill overlap
        job_skills = job_row.get('skills_list', [])
        resume_skills = resume_row.get('skills_list', [])
        features['skill_overlap'] = self.extract_skill_overlap(job_skills, resume_skills)
        
        # Education alignment
        required_education = job_row.get('education_required', 'Bachelor')
        resume_education = resume_row.get('Education', 'Bachelor')
        features['education_alignment'] = self.calculate_education_score(
            required_education, resume_education
        )
        
        # Experience alignment
        required_exp = float(job_row.get('experience_required', 0))
        resume_exp = float(resume_row.get('Experience (Years)', 0))
        features['experience_alignment'] = self.calculate_experience_alignment(
            required_exp, resume_exp
        )
        
        # Salary compatibility
        job_salary = float(job_row.get('salary', 0))
        resume_salary = float(resume_row.get('Salary Expectation ($)', 0))
        features['salary_compatibility'] = self.calculate_salary_compatibility(
            job_salary, resume_salary
        )
        
        # Semantic similarity from embeddings
        if job_embedding is not None and resume_embedding is not None:
            similarity = cosine_similarity(
                job_embedding.reshape(1, -1),
                resume_embedding.reshape(1, -1)
            )[0][0]
            features['semantic_similarity'] = float(similarity)
        else:
            features['semantic_similarity'] = 0.5
        
        # Count of matching skills
        features['matching_skills_count'] = len(
            set(str(s).lower().strip() for s in job_skills) &
            set(str(s).lower().strip() for s in resume_skills)
        )
        
        # Job title relevance (placeholder)
        features['job_title_relevance'] = 0.5
        
        return features
    
    def engineer_features_for_all_pairs(self, 
                                       jobs_df: pd.DataFrame,
                                       resumes_df: pd.DataFrame,
                                       job_embeddings: np.ndarray = None,
                                       resume_embeddings: np.ndarray = None) -> pd.DataFrame:
        """
        Engineer features for all job-resume pairs.
        
        Args:
            jobs_df (pd.DataFrame): Cleaned jobs data
            resumes_df (pd.DataFrame): Cleaned resumes data
            job_embeddings (np.ndarray): Optional job embeddings
            resume_embeddings (np.ndarray): Optional resume embeddings
            
        Returns:
            pd.DataFrame: Feature matrix with shape (n_resumes * n_jobs, n_features)
        """
        logger.info("Engineering features for all job-resume pairs...")
        
        features_list = []
        
        for resume_idx, (_, resume_row) in enumerate(resumes_df.iterrows()):
            for job_idx, (_, job_row) in enumerate(jobs_df.iterrows()):
                
                # Get embeddings if available
                job_emb = job_embeddings[job_idx] if job_embeddings is not None else None
                resume_emb = resume_embeddings[resume_idx] if resume_embeddings is not None else None
                
                # Engineer features
                features = self.engineer_features_for_pair(
                    job_row, resume_row, job_emb, resume_emb
                )
                
                # Add identifiers
                features['job_id'] = job_row.get('job_id', job_idx)
                features['resume_id'] = resume_row.get('Resume_ID', resume_idx)
                features['job_idx'] = job_idx
                features['resume_idx'] = resume_idx
                
                features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Engineered {len(features_df)} feature vectors")
        return features_df
    
    def create_feature_matrix(self, features_df: pd.DataFrame,
                            feature_columns: List[str] = None) -> np.ndarray:
        """
        Create normalized feature matrix from features dataframe.
        
        Args:
            features_df (pd.DataFrame): DataFrame with engineered features
            feature_columns (List[str]): Columns to use as features
            
        Returns:
            np.ndarray: Normalized feature matrix
        """
        if feature_columns is None:
            feature_columns = [
                'skill_overlap', 'education_alignment', 'experience_alignment',
                'salary_compatibility', 'semantic_similarity', 'matching_skills_count'
            ]
        
        X = features_df[feature_columns].fillna(0).values
        
        # Normalize matching_skills_count (last column)
        if 'matching_skills_count' in feature_columns:
            idx = feature_columns.index('matching_skills_count')
            max_count = X[:, idx].max()
            if max_count > 0:
                X[:, idx] = X[:, idx] / max_count
        
        logger.info(f"Created feature matrix: {X.shape}")
        return X
    
    def compute_match_score(self, features: Dict, weights: Dict = None) -> float:
        """
        Compute overall match score from features.
        
        Args:
            features (Dict): Feature dictionary from engineer_features_for_pair
            weights (Dict): Weights for each feature (optional)
            
        Returns:
            float: Match score (0.0 to 1.0)
        """
        if weights is None:
            weights = {
                'skill_overlap': 0.25,
                'education_alignment': 0.15,
                'experience_alignment': 0.20,
                'salary_compatibility': 0.15,
                'semantic_similarity': 0.15,
                'matching_skills_count': 0.10
            }
        
        score = sum(
            features.get(key, 0.0) * weight
            for key, weight in weights.items()
        )
        
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]


def engineer_all_features(jobs_df: pd.DataFrame, resumes_df: pd.DataFrame,
                         config: Dict, job_embeddings: np.ndarray = None,
                         resume_embeddings: np.ndarray = None) -> Dict:
    """
    Complete feature engineering pipeline.
    
    Args:
        jobs_df (pd.DataFrame): Cleaned jobs data
        resumes_df (pd.DataFrame): Cleaned resumes data
        config (Dict): Configuration dictionary
        job_embeddings (np.ndarray): Optional job embeddings
        resume_embeddings (np.ndarray): Optional resume embeddings
        
    Returns:
        Dict: Contains features_df, feature_matrix, and engineer
    """
    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)
    
    engineer = FeatureEngineer(config)
    
    # Engineer features for all pairs
    features_df = engineer.engineer_features_for_all_pairs(
        jobs_df, resumes_df, job_embeddings, resume_embeddings
    )
    
    # Create normalized feature matrix
    feature_matrix = engineer.create_feature_matrix(features_df)
    
    # Compute match scores
    logger.info("Computing match scores...")
    features_df['match_score'] = features_df.apply(
        lambda row: engineer.compute_match_score(row.to_dict()),
        axis=1
    )
    
    logger.info(f"Match score statistics:")
    logger.info(f"  Mean: {features_df['match_score'].mean():.3f}")
    logger.info(f"  Median: {features_df['match_score'].median():.3f}")
    logger.info(f"  Std: {features_df['match_score'].std():.3f}")
    logger.info(f"  Min: {features_df['match_score'].min():.3f}")
    logger.info(f"  Max: {features_df['match_score'].max():.3f}")
    
    return {
        'features_df': features_df,
        'feature_matrix': feature_matrix,
        'engineer': engineer
    }


if __name__ == "__main__":
    logger.info("Feature engineering module loaded successfully")
