from src.entity.artifact_entity import DataIngestionArtifact, ChunkingArtifact

import regex as re
import os
import sys
os.environ['JAVA_HOME'] = r'D:\Softwares\Microsoft\jdk-17.0.15.6-hotspot'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, explode, pandas_udf, lit, length
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import expr
from functools import reduce


def create_custom_chunks(text:str, overlap: int = 200, chunk_size = 2000)-> list:
    if not isinstance(text, str):
        return []
    text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F-\x9F]','', text)
    chunks = []

    
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
        
    prev_chunk_len = 0  
    
    while len(text) > chunk_size:
        right = chunk_size
        
        last_newline_index = text[:chunk_size].rfind("\n")
        
        if last_newline_index > 0:
            right = last_newline_index

        # to ensure chunks have complete words instead of cut words
        # if right > 0 and text[right-1].isalnum() and text[right].isalnum():
        #     last_space = text[:right].rfind(' ')
        #     if last_space > right-50 and last_space > 0:
        #         right = last_space
            
        chunk = text[:right].strip()
        if chunk:
            chunks.append(chunk)
        
        if len(chunks) > 1 and prev_chunk_len < 1000 and len(chunks[-1]) < 1000 :
            chunks[-2]+=" " + chunks[-1]
            chunks = chunks[:-1]
        
        prev_chunk_len = len(chunks[-1])
        
        if right > overlap:
            start = right - overlap
            # Prevent index error
            while start < len(text):
                if not (text[start].isalnum() and text[start-1].isalnum()):
                    break
                start += 1
            text = text[start:]
        else:
            text = text[right:]
            
    if text:
        rem = text.strip()
        if rem:
            chunks.append(rem)        

    return chunks


@pandas_udf(returnType=ArrayType(StringType()))
def chunk_pandas_udf(series):
    def safe_apply(text):
        try:
            return create_custom_chunks(text)
        except Exception:
            return []
    return series.apply(safe_apply)



class Chunking:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact):
        """
        Performs chunking of SEC filing dataset into smaller chunks to fit into embedding model context size.
        
        Args:
            data_ingestion_artifact: DataIngestionArtifact containing path to ingested parquet data
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            # Start Spark session with optimized configurations
            self.spark = SparkSession.builder \
                .appName("AIG_SEC_Filing_Analysis") \
                .master("local[8]") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.python.worker.memory", "2g") \
                .config("spark.sql.shuffle.partitions", "50") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
                .config("spark.python.worker.faulthandler.enabled", "true") \
                .getOrCreate()

            self.spark.sparkContext.setLogLevel("WARN")

        except Exception as e:
            raise(e)
        

    def read_saved_data_into_dataframe(self):
        """
        Reads saved parquet data and limits to 10 companies as per requirements.
        
        Returns:
            PySpark DataFrame with limited companies
        """
        try:
            if len(self.data_ingestion_artifact.corpus_file_path)==1:
                df_all_companies = self.spark.read.parquet(self.data_ingestion_artifact.corpus_file_path[0])
                df = df_all_companies.limit(10)
            else:
                df = self.spark.read.parquet(*self.data_ingestion_artifact.corpus_file_path)
            return df
        except Exception as e:
            raise(e)


    def custom_text_splitter(self) -> ChunkingArtifact:
        """
        Main method to perform text chunking on all SEC filing sections.
        
        Returns:
            ChunkingArtifact containing exploded DataFrame with chunks
        """
        try:
            df = self.read_saved_data_into_dataframe()

            # Get all columns except metadata columns
            columns = [column for column in df.columns if column not in ["filename", "cik", "year"]]

            # Create chunks for each section column
            for column in columns:
                df = df.withColumn(f"{column}_chunked", chunk_pandas_udf(col(column)))

            # Explode and collect all sections into unified format
            all_chunks = []
            for section_name in columns:
                df_section = df.select(
                    "cik", 
                    "filename",
                    "year",
                    explode(col(f"{section_name}_chunked")).alias("chunk")
                ).filter(length(col("chunk")) > 15) \
                .withColumn("section_name", lit(section_name)) \
                .withColumn("chunk_id", expr("uuid()"))
                df_section = df_section.select("cik", "filename", "year", "chunk", "section_name","chunk_id")
                
                all_chunks.append(df_section)

            # Union all sections into single DataFrame
            df_exploded = reduce(DataFrame.unionByName, all_chunks)              
            chunking_artifact = ChunkingArtifact(chunked_dataframe=df_exploded)
            return chunking_artifact
        
        except Exception as e:
            raise(e)