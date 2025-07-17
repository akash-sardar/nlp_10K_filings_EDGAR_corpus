"""
Main Application Pipeline:
1. Data ingestion from HuggingFace datasets
2. Text chunking for embedding model
3. Embedding generation using sentence transformers
4. Clustering analysis with outlier detection and visualization

Output:
- Three visualization plots (clusters, outliers, sections)
- Summary statistics CSV file
- Results saved in 'Results/' directory
"""
import sys
import random
import numpy as np
import torch

# Set all random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)



from src.components.data_ingestion import DataIngestion
from src.components.chunking import Chunking
from src.components.embeddings import Embedding
from src.components.clustering import Clustering
from src.components.retreiver import Retriever
from src.components.llm_processor import LLM_processer
from src.components.evaluation import Evaluator


def clustering_pipeline():
    try:
        # Step 1: Data Ingestion
        # Download SEC filing data from HuggingFace hub
        print("Starting data ingestion...")
        data_ingestion = DataIngestion()
        data_ingestion_artifact = data_ingestion.import_data_from_hub()
        print("Data ingestion completed.")

        # Step 2: Text Chunking
        # Split long documents into smaller chunks for embedding processing
        print("Starting text chunking...")
        chunking = Chunking(data_ingestion_artifact=data_ingestion_artifact)
        chunking_artifact = chunking.custom_text_splitter()
        print("Text chunking completed.")

        # Step 3: Embedding Generation
        # Convert text chunks to numerical embeddings using sentence transformers
        print("Starting embedding generation...")
        embedding = Embedding(chunking_artifact=chunking_artifact)
        embedding_artifact = embedding.embed_chunks()
        print("Embedding generation completed.")

        # Step 4: Clustering Analysis
        # Perform dimensionality reduction, clustering, and outlier detection
        print("Starting clustering analysis...")
        clustering = Clustering(embedding_artifact=embedding_artifact)
        clustering.perform_clustering()
        print("Clustering analysis completed.")
        
        print("\nPipeline completed successfully!")
        print("Check 'Clustering_Results/' directory for visualization plots and summary statistics.")
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        raise(e)


def rag_pipeline(rag = False):
    try:
        rag = True

        # Step 1: Data Ingestion
        # Download SEC filing data from HuggingFace hub
        print("Starting data ingestion...")
        data_ingestion = DataIngestion(rag)
        data_ingestion_artifact = data_ingestion.import_data_from_hub()
        print("Data ingestion completed.")

        # Step 2: Text Chunking
        # Split long documents into smaller chunks for embedding processing
        print("Starting text chunking...")
        chunking = Chunking(data_ingestion_artifact=data_ingestion_artifact)
        chunking_artifact = chunking.custom_text_splitter()
        print("Text chunking completed.")

        # Step 3: Embedding Generation
        # Convert text chunks to numerical embeddings using sentence transformers
        print("Starting embedding generation...")
        embedding = Embedding(chunking_artifact=chunking_artifact, rag=rag)
        embedding_artifact = embedding.embed_chunks()
        print("Embedding generation completed.")

        # Step 4: Retrieve top_k context
        # Retrieve Chunks of relevant text to build context
        print("Starting retrieval...")
        retriever = Retriever(embedding_artifact=embedding_artifact)
        retriever_artifact = retriever.retrieve()
        print("Retrieval completed.")

        # Step 5: Generate answers using LLM
        # Use selected context to extract data attributes out of LLM
        print("Starting generation...")
        generator = LLM_processer(retriever_artifact)
        generated_artifact = generator.generate()
        print("Answer Generation completed.")

        # Step 6: Evaluate Answers
        # Evaluate answers using golden datasets
        print("Starting generation...")
        evaluator = Evaluator(generated_artifact)
        evaluator.evaluate()
        print("Evaluation completed.")                 

    except Exception as e:
        raise(e)    


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rag = True
        print("Running RAG PIPELINE for TASK #2")
        rag_pipeline(rag)
    else:
        print("Running Clustering PIPELINE for TASK #1")
        clustering_pipeline()

