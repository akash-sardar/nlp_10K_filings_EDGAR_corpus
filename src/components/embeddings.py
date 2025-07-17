from src.entity.artifact_entity import ChunkingArtifact, EmbeddingArtifact
from src.utils.main_utils import encode_using_sbert


import os
import torch
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


class Embedding:
    def __init__(self, chunking_artifact: ChunkingArtifact, rag = False):
        """
        Performs embedding of chunked dataset using sentence transformer.
        
        Args:
            chunking_artifact: ChunkingArtifact containing PySpark DataFrame with 
                             exploded chunks as rows
        """
        try:
            self.chunking_artifact = chunking_artifact
            self.rag = rag
            # Initialize Spark session for data processing
            self.spark = SparkSession.builder \
                .appName("AIG_SEC_Filing_Analysis") \
                .getOrCreate()            
        except Exception as e:
            raise(e)
        

    def convert_to_pandas(self, df):
        """
        Convert PySpark DataFrame to Pandas DataFrame for faster local processing.
        
        Args:
            df: PySpark DataFrame containing chunked data
            
        Returns:
            Pandas DataFrame with selected columns for embedding
            
        Note:
            Stops Spark session after conversion to free resources
        """
        try:
            # Select only required columns for embedding
            print("Starting Pandas conversion")
            df_pandas = df.select("cik", "filename", "year", "chunk", "section_name", "chunk_id").toPandas()
            
            # Stop Spark session to free resources
            # self.spark.stop()
            
            return df_pandas
        except Exception as e:
            raise(e)


    def embed_chunks(self)->EmbeddingArtifact:
        """
        Main method to embed chunked data using sentence transformers.
        
        Uses all-MiniLM-L6-v2 model for generating 384-dimensional embeddings.
        Processes chunks in batches for memory efficiency.
        
        Returns:
            EmbeddingArtifact containing pandas DataFrame with embeddings column
        """
        try:
            
            if self.rag:
                
                chunk_file_path = "Knowledge/df_chunked_pandas.csv"

                if os.path.exists(chunk_file_path):
                    print("Loading existing chunked DataFrame from Knowledge/ folder...")
                    df_chunked_pandas = pd.read_csv(chunk_file_path)
                else:
                    print("No existing chunk file found. Update Vaidation Dataset. Generating New chunks...")
                    # Get chunked dataframe from artifact
                    df_exploded = self.chunking_artifact.chunked_dataframe

                    # Convert to pandas for faster local processing
                    df_chunked_pandas = self.convert_to_pandas(df_exploded)

                    # Save to CSV
                    os.makedirs("Data", exist_ok=True)
                    df_chunked_pandas.to_csv(chunk_file_path, index=False)
                    print("Chunked DataFrame saved to Knowledge/df_chunked_pandas.csv")
            else:
                    # Get chunked dataframe from artifact
                    df_exploded = self.chunking_artifact.chunked_dataframe

                    # Convert to pandas for faster local processing
                    df_chunked_pandas = self.convert_to_pandas(df_exploded)                

            # Extract chunks as list for batch processing
            chunks_list = df_chunked_pandas["chunk"].tolist()

            embeddings = encode_using_sbert(chunks_list)          
            
            # Add embeddings as new column to dataframe
            df_chunked_pandas["embeddings"] = embeddings.tolist()

            # create Pyspark Dataframe for future usage
            df_cleaned = self.spark.createDataFrame(df_chunked_pandas)

            # Create and return embedding artifact
            embedding_artifact = EmbeddingArtifact(df=df_chunked_pandas, spark_df = df_cleaned)

            return embedding_artifact

        except Exception as e:
            raise(e)