from src.entity.artifact_entity import EmbeddingArtifact, RetrieverArtifact
from src.utils.main_utils import encode_using_sbert

from pathlib import Path
import ast
import numpy as np
import regex as re
import spacy
NER = spacy.load("en_core_web_sm")

from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql import DataFrame
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.sql.types import ArrayType, StringType, FloatType,  StructType, StructField, DoubleType

class Retriever:
   """
   This class implements a retrieval that combines semantic search
   with year-based filtering to find the most relevant document chunks for user queries.
  
   It caches year-specific data and provides top-k retrieval with similarity scoring.
   
   Attributes:
       embedding_artifact (EmbeddingArtifact): Contains preprocessed document embeddings
       spark (SparkSession): Distributed computing session for large-scale data processing
   """
   
   def __init__(self, embedding_artifact: EmbeddingArtifact):
       try:
           self.embedding_artifact = embedding_artifact           
           self.spark = SparkSession.builder \
               .appName("AIG_SEC_Filing_Analysis") \
               .getOrCreate()               
       except Exception as e:
           raise(e)
       
   def return_top_k_context(self, query:str, query_embedding: np.array, df_embedding: DataFrame, k = 3)-> list:
       """
       Retrieve the top-k most relevant document chunks for a given query.
      
       Args:
           query (str): The user's natural language query
           query_embedding (np.array): Numerical representation of the query
           df_embedding (DataFrame): PySpark DataFrame containing document embeddings
           k (int, optional): Number of top chunks to retrieve. Defaults to 3.
       
       Returns:
           list: List of Row objects containing the top-k most relevant chunks with metadata
                 including company CIK, year, section name, chunk content, chunk ID, and similarity score
       """
       # Extract YEAR from query using NER
       year = self.extract_year(query)

    #    if year:
    #        # Create year-specific knowledge directory for efficient caching
    #        knowledge_dir = Path("Knowledge") / str(year)
    #        knowledge_dir.mkdir(parents=True, exist_ok=True)

    #        knowledge_path = knowledge_dir / f"{year}.parquet"

    #        if knowledge_path.exists():
    #            # Utilize cached year-specific data for faster retrieval
    #            filtered_embeddings_df = self.spark.read.parquet(str(knowledge_path))
    #        else:
    #            # Filter original dataset and cache for future use
    #            filtered_embeddings_df = df_embedding.filter(col("year") == year)
    #            # Persist filtered data in Parquet format
    #            filtered_embeddings_df.write.mode("overwrite").parquet(str(knowledge_path))
    #    else:
    #        # Use complete dataset when no YEAR identified
    #        filtered_embeddings_df = df_embedding

       if year:
           filtered_embeddings_df = df_embedding.filter(col("year") == year)
       else:
           filtered_embeddings_df = df_embedding

       # Convert query embedding to Spark ML Vector format for computation
       query_vector = Vectors.dense(query_embedding)

       # Define cosine  using Spark UDF
       @udf(DoubleType())
       def cosine_similarity(vec):
           """
           Compute cosine similarity between document and query vectors.
           """
           if isinstance(vec, list):
               vec = Vectors.dense(vec)
           dot = float(vec.dot(query_vector))
           norm1 = float(vec.norm(2))
           norm2 = float(query_vector.norm(2))
           return dot / (norm1 * norm2) if norm1!=0 and norm2 !=0 else 0.0
       
       # Apply similarity computation and rank results
       df_result = filtered_embeddings_df.withColumn("score", cosine_similarity(col("embeddings")))
       # Retrieve top-k chunks with comprehensive metadata
       top_k_chunks = df_result.orderBy(col("score").desc()).select("cik","year", "section_name","chunk","chunk_id", "score").take(k)

       return top_k_chunks


   def extract_year(self, text):
       """
       Extract year information from user queries using advanced NLP techniques.

       Args:
           text (str): Input text from which to extract year information
       
       Returns:
           str or None: Extracted year as string (e.g., "2019") or None if no year found
       """
       if not isinstance(text, str):
           return None
       

       entities = NER(text).ents
       dates = [ent.text for ent in entities if ent.label_ == "DATE"]
       pattern = r'(20\d{2}|19\d{2})'  # Pattern for years 1900-2099
       
       # Process NER-identified date entities first
       if dates:
           for date_text in dates:
               match = re.findall(pattern, date_text)
               if match:
                   year = match[0]
                   print(f"Extracted year from NER date entity: {year}")
                   return year
       
       # direct text pattern matching
       match = re.findall(pattern, text)
       if match:
           # Handle multiple years by selecting the most recent
           years = [int(year) for year in match]
           year = str(max(years))  # Prioritize most recent year
           print(f"Extracted year from direct text search: {year}")
           return year
       
       return None
       

   def retrieve(self) -> RetrieverArtifact:
       """
       Execute the complete document retrieval pipeline for multiple queries.
      
       Returns:
           RetrieverArtifact: Structured artifact containing all retrieved contexts
                             and original queries for further processing
       
       Raises:
           Exception: If query loading fails, embedding generation errors occur,
                     or retrieval pipeline encounters critical errors
       """
       try:
           # Extract preprocessed embeddings
           df_embedding = self.embedding_artifact.spark_df

           # Load and validate query data from external source
           try:
               queries_path = "Query/query_list.txt"
               with open(queries_path, 'r', encoding='utf-8') as file:
                   content = file.read()
                   
               if content.strip():
                   # Parse structured query data
                   query_list = ast.literal_eval(content)
               else:
                   raise Exception("Query file is empty")
                   
           except FileNotFoundError:
               print(f"Query List not found")
               raise
           except (ValueError, SyntaxError) as e:
               print(f"Invalid format in query file: {e}")
               raise
           except Exception as e:
               print(f"Error reading queries: {e}")
               raise
           
           # Extract query texts for embedding generation
           queries = [query["query"] for query in query_list if query["query"]]

           # Generate query embeddings
           query_embeddings = encode_using_sbert(queries)
           
           # Validate embedding generation results
           if query_embeddings.shape[0] != len(queries):
               print(f"Query embedding mismatch: Query embedding returned {query_embeddings.shape[0]} rows")
               print(f"Number of queries {len(queries)}")
               raise
           print(f"Query embeddings shape: {query_embeddings.shape}")

           # Execute retrieval for each query with context
           context = []
           for index, query in enumerate(query_list):
               # Retrieve top-k relevant chunks for current query
               context.append(self.return_top_k_context(query["query"], 
                                                        query_embeddings[index], 
                                                        df_embedding))

           # artifact for downstream processing
           retrieved_artifact = RetrieverArtifact(all_contexts=context, queries=query_list)

           # Clean up
           self.spark.stop()

           return retrieved_artifact
       
       except Exception as e:
           raise(e)