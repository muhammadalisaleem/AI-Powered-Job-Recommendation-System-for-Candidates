"""
Embeddings Module
Handles Sentence-Transformer embeddings generation and FAISS indexing for semantic similarity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer
import pickle

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """
    Handles generation and management of Sentence-Transformer embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """
        Initialize embeddings generator.
        
        Args:
            model_name (str): Sentence-Transformer model name
            embedding_dim (int): Dimension of embeddings (should match model)
        """
        logger.info(f"Loading Sentence-Transformer model: {model_name}")
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded. Embedding dimension: {embedding_dim}")
        
        self.job_embeddings = None
        self.resume_embeddings = None
        self.job_texts = None
        self.resume_texts = None
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, 
                          show_progress_bar: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            batch_size (int): Batch size for processing
            show_progress_bar (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Array of embeddings (n_texts, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def generate_job_embeddings(self, jobs_df: pd.DataFrame, 
                               text_column: str = "job_description") -> np.ndarray:
        """
        Generate embeddings for job descriptions.
        
        Args:
            jobs_df (pd.DataFrame): DataFrame containing job data
            text_column (str): Column name containing job descriptions
            
        Returns:
            np.ndarray: Job embeddings array
        """
        logger.info(f"Generating embeddings for {len(jobs_df)} job descriptions...")
        texts = jobs_df[text_column].fillna("").tolist()
        self.job_texts = texts
        self.job_embeddings = self.generate_embeddings(texts)
        return self.job_embeddings
    
    def generate_resume_embeddings(self, resumes_df: pd.DataFrame,
                                  text_column: str = "Skills") -> np.ndarray:
        """
        Generate embeddings for resume skills/descriptions.
        
        Args:
            resumes_df (pd.DataFrame): DataFrame containing resume data
            text_column (str): Column name containing resume text
            
        Returns:
            np.ndarray: Resume embeddings array
        """
        logger.info(f"Generating embeddings for {len(resumes_df)} resumes...")
        texts = resumes_df[text_column].fillna("").tolist()
        self.resume_texts = texts
        self.resume_embeddings = self.generate_embeddings(texts)
        return self.resume_embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, path: Path):
        """
        Save embeddings to disk.
        
        Args:
            embeddings (np.ndarray): Embeddings array to save
            path (Path): Path to save embeddings
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to: {path}")
    
    def load_embeddings(self, path: Path) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            path (Path): Path to embeddings file
            
        Returns:
            np.ndarray: Loaded embeddings
        """
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings from: {path} (shape: {embeddings.shape})")
        return embeddings


class FAISSIndex:
    """
    Handles FAISS indexing for fast similarity search.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim (int): Dimension of embeddings
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_mapping = None
        logger.info(f"Initialized FAISS index (dimension: {embedding_dim})")
    
    def build_index(self, embeddings: np.ndarray, use_gpu: bool = False) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings (np.ndarray): Embeddings array (n_samples, embedding_dim)
            use_gpu (bool): Whether to use GPU (requires faiss-gpu)
        """
        logger.info(f"Building FAISS index from {len(embeddings)} embeddings...")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Create index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"GPU not available: {e}. Using CPU.")
        
        # Add embeddings
        self.index.add(embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings.
        
        Args:
            query_embeddings (np.ndarray): Query embeddings (n_queries, embedding_dim)
            k (int): Number of results to return
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, indices) arrays
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embeddings = query_embeddings.astype('float32')
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices
    
    def save_index(self, path: Path):
        """
        Save FAISS index to disk.
        
        Args:
            path (Path): Path to save index
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to move from GPU to CPU if needed (only if faiss-gpu is installed)
        try:
            if hasattr(faiss, 'GpuIndex') and isinstance(self.index, faiss.GpuIndex):
                self.index = faiss.index_gpu_to_cpu(self.index)
        except (AttributeError, TypeError):
            pass  # CPU version doesn't have GPU support
        
        faiss.write_index(self.index, str(path))
        logger.info(f"Saved FAISS index to: {path}")
    
    def load_index(self, path: Path, use_gpu: bool = False):
        """
        Load FAISS index from disk.
        
        Args:
            path (Path): Path to index file
            use_gpu (bool): Whether to use GPU
        """
        self.index = faiss.read_index(str(path))
        
        if use_gpu:
            try:
                if hasattr(faiss, 'StandardGpuResources'):
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info("Loaded FAISS index on GPU")
            except Exception as e:
                logger.warning(f"GPU not available: {e}")
        
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")


def generate_all_embeddings(jobs_df: pd.DataFrame, resumes_df: pd.DataFrame,
                           config: Dict, output_dir: Optional[Path] = None) -> Dict:
    """
    Generate all embeddings for jobs and resumes.
    
    Args:
        jobs_df (pd.DataFrame): Cleaned jobs data
        resumes_df (pd.DataFrame): Cleaned resumes data
        config (Dict): Configuration with EMBEDDING_MODEL and EMBEDDING_DIM
        output_dir (Optional[Path]): Directory to save embeddings
        
    Returns:
        Dict: Contains generator, job_embeddings, resume_embeddings, faiss_index
    """
    logger.info("=" * 80)
    logger.info("GENERATING EMBEDDINGS")
    logger.info("=" * 80)
    
    # Initialize embeddings generator
    generator = EmbeddingsGenerator(
        model_name=config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dim=config.get("EMBEDDING_DIM", 384)
    )
    
    # Generate job embeddings
    job_embeddings = generator.generate_job_embeddings(jobs_df)
    
    # Generate resume embeddings
    resume_embeddings = generator.generate_resume_embeddings(resumes_df)
    
    # Build FAISS index
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING FAISS INDEX")
    logger.info("=" * 80)
    
    faiss_index = FAISSIndex(embedding_dim=config.get("EMBEDDING_DIM", 384))
    faiss_index.build_index(job_embeddings)
    
    # Save embeddings if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        generator.save_embeddings(job_embeddings, output_dir / "jobs_embeddings.npy")
        generator.save_embeddings(resume_embeddings, output_dir / "resumes_embeddings.npy")
        faiss_index.save_index(output_dir / "faiss_index.bin")
        logger.info(f"Saved all embeddings to: {output_dir}")
    
    return {
        'generator': generator,
        'job_embeddings': job_embeddings,
        'resume_embeddings': resume_embeddings,
        'faiss_index': faiss_index
    }


if __name__ == "__main__":
    from config.config import EMBEDDING_MODEL, EMBEDDING_DIM
    
    # Example usage
    logger.info("Embeddings module loaded successfully")
