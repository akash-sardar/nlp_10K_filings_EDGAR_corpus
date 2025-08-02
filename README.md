# SEC Filing Analysis - AIG Assignment

## Repository Structure
```
├── app.py                          # Main pipeline execution (clustering vs RAG)
├── setup.py                        # Package configuration
├── requirements.txt                # Dependencies
├── METHODOLOGY.md                  # Technical methodology and results
├── src/
│   ├── components/                 # Core processing modules
│   │   ├── data_ingestion.py       # HuggingFace data loading
│   │   ├── chunking.py             # PySpark text chunking with UDFs
│   │   ├── embeddings.py           # Sentence transformer integration
│   │   ├── clustering.py           # PCA, t-SNE, K-means implementation
│   │   ├── retreiver.py            # Semantic search and retrieval
│   │   ├── llm_processor.py        # GPT-4 query processing
│   │   └── evaluation.py           # BERT F1 validation
│   ├── entity/
│   │   └── artifact_entity.py      # Data flow artifacts (OOP containers)
│   └── utils/
│       └── main_utils.py           # Shared embedding functions
├── Knowledge/
│   └── df_chunked_pandas.csv       # Preprocessed chunks for RAG
├── Notebooks/
│   ├── Notebook_RAG_task_02.ipynb                          # Notebook Experiments for RAG
│   ├── Notebook_RAG_task_02.html                           # HTML Presentation for RAG
│   ├── Notebook_clustering_task_01_refined.ipynb           # Notebook Experiments for Clustering task
│   ├── Notebook_clustering_task_01_refined.html            # HTML Presentation for Clustering task
│   ├── Notebook_clustering_task_01.ipynb                   # Notebook Experiments for Clustering task
│   └── Notebook_clustering_task_01.html                    # HTML Presentation for Clustering task
├── Clustering_Results/             # Task 1 outputs
│   ├── Fig_01_*.jpeg               # Section-colored t-SNE plot
│   ├── Fig_02_*.jpeg               # Cluster-colored t-SNE plot
│   ├── Fig_03_*.jpeg               # Outlier visualization
│   └── summary_statistics.csv      # Clustering metrics
├── RAG_Validation_Results/         # Task 2 outputs
│   ├── VALIDATION_DATA.py          # Ground truth queries
│   └── evaluation_results.csv      # RAG performance metrics
└── .gitignore                      # Version control exclusions
```

## Overview
PySpark-based solution for analyzing SEC 10-K filings using clustering and RAG (Retrieval-Augmented Generation) techniques. Implements two distinct pipelines for document understanding and data extraction.

## Architecture
- **Language**: Python
- **Framework**: PySpark for distributed processing
- **Design Pattern**: Object-oriented pipeline with reusable components
- **Data Source**: EDGAR-CORPUS from HuggingFace

## Pipeline Structure
```
app.py
├── clustering_pipeline() # Task 1: Document clustering and outlier detection
└── rag_pipeline()        # Task 2: RAG-based data extraction
```

## Core Components

### Data Processing Classes
- **DataIngestion**: Downloads and filters SEC filing data
- **Chunking**: Splits documents using custom PySpark UDFs
- **Embedding**: Generates 768-dimensional embeddings via sentence-transformers
- **Clustering**: Performs PCA, t-SNE, and K-means clustering
- **Retriever**: Implements semantic search with cosine similarity
- **LLM_processer**: Handles GPT-4 integration with query classification
- **Evaluator**: Validates system performance using BERT F1 scores

### Pipeline Execution
```bash
# Task 1: Clustering Analysis
python app.py

# Task 2: RAG Pipeline
python app.py rag
```

## Task 1: Document Clustering
**Objective**: Visualize SEC filings in 2D space and identify outliers

**Process**:
1. Custom text chunking (2,000 chars, 200 overlap)
2. Embedding generation (all-mpnet-base-v2)
3. Standard scaling and PCA (75 components)
4. K-means clustering (k=20)
5. Statistical outlier detection (mean + 2σ)
6. t-SNE visualization

**Results**:
- 2,217 chunks from 10 companies
- 48 outliers (2.16%)
- 3 visualization plots generated

## Task 2: RAG System
**Objective**: Extract specific data attributes from multi-year filings

**Process**:
1. Multi-year data processing (2018-2020)
2. Semantic retrieval (top-k=3)
3. Query classification (financial/risk/operational/regulatory/general)
4. GPT-4 response generation
5. BERT F1 evaluation

**Results**:
- 100% answer accuracy (7/7 queries)
- 85.7% chunk recall
- 0.846 average BERT F1 score

## Technical Implementation

### PySpark Optimizations
- Local distributed processing with 4GB memory allocation
- 100 shuffle partitions for parallel processing
- Pandas UDFs for vectorized text operations
- Adaptive query execution enabled

### OOP Design Benefits
- Modular components with clear separation of concerns
- Reusable classes across both pipelines
- Artifact-based data flow between components
- Consistent error handling and logging

### Shared Infrastructure
Both pipelines leverage the same:
- Data ingestion and chunking logic
- Embedding generation methods
- PySpark session management
- Artifact entity definitions

## Installation
```bash
pip install -r requirements.txt
```

## Dependencies
- PySpark 3.x
- sentence-transformers
- scikit-learn
- matplotlib, seaborn
- openai
- bert-score

## Output Structure
```
├── Clustering_Results/
│   ├── Fig_01_2D_Embeddings_with_Section_name_colors.jpeg
│   ├── Fig_02_2D_Embeddings_clusters.jpeg
│   ├── Fig_03_2D_Embeddings_outliers.jpeg
│   └── summary_statistics.csv
└── RAG_Validation_Results/
    └── evaluation_results.csv
```

## Key Features
- **Scalable Processing**: PySpark handles large SEC filing datasets
- **Flexible Pipeline**: Single codebase supports both clustering and RAG tasks
- **Robust Evaluation**: BERT F1 scoring for semantic accuracy
- **Production Ready**: Modular design with proper error handling

## Performance Metrics
- **Clustering**: 1.85% outlier rate with clear section-based patterns
- **RAG**: Perfect accuracy on validation queries with strong semantic retrieval
- **Processing**: Efficient memory usage with distributed computing benefits