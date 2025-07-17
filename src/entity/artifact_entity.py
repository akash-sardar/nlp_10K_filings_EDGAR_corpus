# --------------------------------------------------
# artifact_entity.py
# --------------------------------------------------
# This module defines data structures using Python's dataclass for 
# handling artifacts generated during the data ingestion and request ingestion stages.
# --------------------------------------------------

from dataclasses import dataclass
from pyspark.sql import DataFrame
import pandas as pd

# --------------------------------------------------
# Data Ingestion Artifact Class
# --------------------------------------------------
@dataclass
class DataIngestionArtifact:
    """
    DataIngestionArtifact stores information about the artifacts 
    generated during the data ingestion phase.
    
    Attributes:
        corpus_file_path (str): Path to the system file generated during ingestion.
    """
    corpus_file_path: list #  Path or list of paths to the ingested file

# --------------------------------------------------
# Chunking Artifact Class
# --------------------------------------------------
@dataclass
class ChunkingArtifact:
    """
    Chunks the dataset into pieces to be able to be ingested by Sentence Transformers
    
    Attributes:
        chunked_dataframe (DataFrame): Path to chunked data
    """
    chunked_dataframe: DataFrame  #  chunked dataframe   

# --------------------------------------------------
# Embedding Artifact Class
# --------------------------------------------------
@dataclass
class EmbeddingArtifact:
    """
    Converts the chunks into Embeddings
    
    Attributes:
        embedded_dataframe (DataFrame)
    """
    df: pd.DataFrame # pandas dataframe with embeddings included
    spark_df: DataFrame # Sparkdataframe with embeddings included

# --------------------------------------------------
# Retriever Artifact Class
# --------------------------------------------------
@dataclass
class RetrieverArtifact:
    """
    Retrieves relevant text from chunks and creates context for llm
    
    Attributes:
        top_k_context: list
        queries: List of input queries and ground truth
    """
    all_contexts: list # List of rows with top_k context
    queries: list # List of queries requested


# --------------------------------------------------
# Generated Artifact Class
# --------------------------------------------------
@dataclass
class GeneratedArtifact:
    """
    Generated result out of LLM process
    
    Attributes:
        responses: List of dicitonary with responses, chunks and chunk_id
        queries: List of input queries and ground truth
    """
    responses: list[dict] # List
    queries: list # List of queries requested

