from src.entity.artifact_entity import DataIngestionArtifact

from pathlib import Path
import os

import datasets

class DataIngestion:
    def __init__(self, rag = False):
        """
        Initializes the DataIngestion instance to copy data from HuggingFace Hub and save in Directory data in parquet format

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
            nested_json_input (dict): Input JSON data for ingestion.
        """
        self.rag = rag

    def import_data_from_hub(self)->DataIngestionArtifact:
        try:

            if self.rag:
                data_dir = Path("Data")
                data_path_2018 = data_dir / "data_2018.parquet"
                data_path_2019 = data_dir / "data_2019.parquet"
                data_path_2020 = data_dir / "data_2020.parquet"
                
                
                #cik = "718413"

                # Loading 2018 10-k Fillings
                # Check if data already exists
                if data_path_2018.exists():
                    print(f"Data already exists for 2018 at {data_path_2018}")
                else:
                    edgar_corpus_2018_raw = datasets.load_dataset("eloukas/edgar-corpus", "year_2018")
                    cik_dataset = edgar_corpus_2018_raw.filter(lambda x: x["cik"] == "718413")
                    os.makedirs(data_dir, exist_ok=True)
                    cik_dataset["train"].to_parquet(data_path_2018)
                    print(f"2018 Data saved to {data_path_2018}")

                # Loading 2019 10-k Fillings
                # Check if data already exists
                if data_path_2019.exists():
                    print(f"Data already exists for 2019 at {data_path_2019}")
                else:
                    edgar_corpus_2019_raw = datasets.load_dataset("eloukas/edgar-corpus", "year_2019")
                    cik_dataset = edgar_corpus_2019_raw.filter(lambda x: x["cik"] == "718413")
                    cik_dataset["train"].to_parquet(data_path_2019)
                    print(f"2019 Data saved to {data_path_2019}")

                # Loading 2020 10-k Fillings
                # Check if data already exists
                if data_path_2020.exists():
                    print(f"Data already exists for 2020 at {data_path_2020}")
                else:
                    edgar_corpus_2020_raw = datasets.load_dataset("eloukas/edgar-corpus", "year_2020")
                    cik_dataset = edgar_corpus_2020_raw.filter(lambda x: x["cik"] == "718413")
                    cik_dataset["train"].to_parquet(data_path_2020)
                    print(f"2020 Data saved to {data_path_2020}")
                # Create and return the data ingestion artifact
                data_ingestion_artifact = DataIngestionArtifact(corpus_file_path=[str(data_path_2018), str(data_path_2019), str(data_path_2020)])
                return data_ingestion_artifact                                                                              
            else:
                data_dir = Path("Data")
                data_path = data_dir / "data.parquet"
                
                # Check if data already exists
                if data_path.exists():
                    print(f"Data already exists at {data_path}")
                else:
                    edgar_corpus_2020_raw = datasets.load_dataset("eloukas/edgar-corpus", "year_2020")
                    os.makedirs(data_dir, exist_ok=True)
                    edgar_corpus_2020_raw["train"].to_parquet(data_path)
                    print(f"Data saved to {data_path}")

                # Create and return the data ingestion artifact
                data_ingestion_artifact = DataIngestionArtifact(corpus_file_path=[str(data_path)])
                return data_ingestion_artifact
            
        except Exception as e:
            raise(e)
        
                    


