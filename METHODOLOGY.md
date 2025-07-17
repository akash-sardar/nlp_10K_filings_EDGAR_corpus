# METHODOLOGY

## Table of Contents
1. [Data Processing Pipeline](#1-data-processing-pipeline)
2. [Text Chunking Strategy](#2-text-chunking-strategy)
3. [Embedding Generation](#3-embedding-generation)
4. [Dimensionality Reduction](#4-dimensionality-reduction)
5. [Clustering Analysis](#5-clustering-analysis)
6. [Outlier Detection](#6-outlier-detection)
7. [Visualization Strategy](#7-visualization-strategy)
8. [Clustering Performance (Task 1)](#8-clustering-performance-task-1)
9. [RAG Pipeline (Task 2)](#9-rag-pipeline-task-2)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Technical Optimizations](#11-technical-optimizations)
12. [Results and Analysis](#12-results-and-analysis)

---

## 1. Data Processing Pipeline

### 1.1 Dataset Selection
- **Source**: EDGAR-CORPUS from HuggingFace (eloukas/edgar-corpus)
- **Filter**: Year 2020, Filing type 10-K, All sections
- **Companies**: Limited to 10 companies from train split
- **Data Size**: 5,480 rows × 23 columns in original dataset

### 1.2 Data Exploration Results
- **Section Variance**: High variance across sections (section_1A: 201-313,147 chars)
- **Boilerplate Content**: Sections 1B, 4, 6, 9 contain mostly standardized text
- **Data Quality**: No missing values, all string data types
- **Section Distribution**: Large sections (1, 1A, 7, 8, 15) require extensive chunking
- **PySpark Processing**: Data distributed across 100 partitions for parallel processing

## 2. Text Chunking Strategy

### 2.1 Chunking Parameters
- **Chunk Size**: 2,000 characters (≈512 tokens for BERT models)
- **Overlap**: 200 characters to preserve context
- **Splitting Logic**: Newline-based splitting to maintain semantic boundaries

### 2.2 Custom Chunking Algorithm
- **Implementation**: PySpark pandas UDF for distributed processing
- **Logic**: Newline-based splitting with overlap preservation
- **Optimization**: Merge small consecutive chunks, clean control characters
- **Distribution**: Chunking operations parallelized across Spark executors

## 3. Embedding Generation

### 3.1 Model Selection
- **Model**: all-mpnet-base-v2 (sentence-transformers)
- **Dimensions**: 768-dimensional embeddings
- **Rationale**: Superior performance for semantic similarity tasks, well-suited for financial document analysis

### 3.2 Processing Approach
- **Distributed Processing**: Initial data handling via PySpark DataFrames
- **Conversion Strategy**: PySpark to Pandas conversion for embedding generation
- **Batch Processing**: Embeddings generated in batches for memory efficiency
- **Storage**: Results cached as parquet files for pipeline efficiency

## 4. Dimensionality Reduction

### 4.1 Principal Component Analysis (PCA)
- **Components**: 75 (explains 85% variance)
- **Rationale**: No clear elbow point, gradual variance decay
- **Variance Distribution**: Information spread across many dimensions

### 4.2 t-SNE Visualization
- **Parameters**: default settings with random_state=42
- **Purpose**: 2D visualization of high-dimensional embeddings
- **Output**: Clear cluster separation compared to PCA

## 5. Clustering Analysis

### 5.1 K-Means Configuration
- **Clusters**: 20 (optimal trade-off between compression and accuracy)
- **Input**: PCA-reduced embeddings (75 dimensions)
- **Evaluation**: Silhouette score = 0.0898 (highest among tested values)

### 5.2 Cluster Selection Rationale
- Tested k=6 to k=24 using silhouette analysis
- Low silhouette scores indicate tightly grouped data points
- k=20 provides meaningful separation for 10 companies × 20 sections

## 6. Outlier Detection

### 6.1 Statistical Threshold Method
- **Metric**: Euclidean distance to nearest cluster center
- **Threshold**: mean + 2 × standard deviation
- **Results**: 41 outliers (1.85% of 2,217 total chunks)

### 6.2 Outlier Distribution
- **Primary Sections**: section_1, section_1A, section_8, section_15
- **Companies**: Concentrated in 2-3 companies with diverse content
- **Interpretation**: Outliers represent unique or non-standard content

## 7. Visualization Strategy

### 7.1 Three-Plot Approach
1. **Cluster Visualization**: t-SNE plot colored by cluster assignments
2. **Outlier Analysis**: Normal points (blue) vs outliers (red) with annotations
3. **Section Analysis**: t-SNE plot colored by section names

### 7.2 Technical Implementation
- **Library**: matplotlib, seaborn, colorcet
- **Color Schemes**: tab20 for clusters, glasbey for sections
- **Annotations**: CIK-section-chunk labels for outliers

## 8. Clustering Performance (Task 1)

### 8.1 Results Summary
| Metric | Value |
|--------|-------|
| Total Chunks | 2,217 |
| Total Outliers | 41 (1.85%) |
| Unique Companies | 10 |
| Unique Sections | 20 |
| Average Chunk Length | 1,506 characters |
| K-means Clusters | 20 |

### 8.2 Visualizations Generated
1. **Fig_01_2D_Embeddings_with_Section_name_colors.jpeg** - t-SNE plot colored by section names
2. **Fig_02_2D_Embeddings_clusters.jpeg** - t-SNE plot colored by cluster assignments  
3. **Fig_03_2D_Embeddings_outliers.jpeg** - t-SNE plot highlighting outliers with CIK-section-chunk labels

### 8.3 Clustering Analysis Results
- **Section Distribution**: Sections 1, 1A, 7, 8, 15 show high dispersion across multiple clusters (diverse content)
- **Compact Clusters**: Sections 9A, 9B form tight clusters (standardized legal language)
- **Outlier Concentration**: Primary outliers in sections 1, 1A, 8, 15 indicating unique/non-standard content
- **Silhouette Score**: 0.0898 (k=20) representing optimal trade-off between compression and accuracy
- **Semantic Separation**: t-SNE reveals clear section-based clustering with some cross-section overlap due to shared legal terminology

**Analysis**: Low outlier rate (1.85%) indicates consistent semantic patterns across SEC filings. Visualizations reveal section-based clustering with outliers concentrated in diverse content areas (sections 1, 1A, 8, 15).

## 9. RAG Pipeline (Task 2)

### 9.1 Multi-Year Processing
- **Dataset**: 2018-2020 filings, single company
- **Embedding**: Same sentence-transformer model
- **Storage**: Year-filtered embedding cache

### 9.2 Retrieval System
- **Method**: Cosine similarity search
- **Context**: Top-k relevant chunks (k=3)
- **Filtering**: Year-based context selection

### 9.3 LLM Integration
- **Model**: OpenAI GPT-4
- **Prompt Engineering**: Query-type specific instructions
- **Categories**: Financial, risk, operational, regulatory, general

## 10. Evaluation Framework

### 10.1 Validation Dataset
- **Size**: 5 manually verified ground truth examples
- **Format**: Question-answer pairs with source verification
- **Purpose**: Accuracy assessment for retrieval and generation

### 10.2 Performance Metrics
- **Retrieval**: Semantic similarity scores
- **Generation**: Manual evaluation against ground truth
- **Error Analysis**: "not found" responses for missing information

## 11. Technical Optimizations

### 11.1 PySpark Configuration
- **Architecture**: Local distributed processing with 4GB memory allocation per executor
- **Partitioning**: 100 shuffle partitions distributed across available CPU cores
- **Adaptive Execution**: Dynamic partition coalescing enabled
- **UDF Processing**: Pandas UDFs for vectorized chunking operations
- **Memory Management**: Fault handler enabled for worker stability

### 11.2 Processing Efficiency
- **Caching**: Intermediate results stored as parquet files for pipeline efficiency
- **Batch Processing**: Embeddings generated in batches for memory efficiency
- **Resource Management**: Spark session lifecycle management

## 12. Results and Analysis

### 12.1 Clustering Performance (Task 1)
| Metric | Value |
|--------|-------|
| Total Chunks | 2,217 |
| Total Outliers | 41 (1.85%) |
| Unique Companies | 10 |
| Unique Sections | 20 |
| Average Chunk Length | 1,506 characters |
| K-means Clusters | 20 |

**Visualizations Generated:**
1. **Fig_01_2D_Embeddings_with_Section_name_colors.jpeg** - t-SNE plot colored by section names
2. **Fig_02_2D_Embeddings_clusters.jpeg** - t-SNE plot colored by cluster assignments  
3. **Fig_03_2D_Embeddings_outliers.jpeg** - t-SNE plot highlighting outliers with CIK-section-chunk labels

**Clustering Analysis Results:**
- **Section Distribution**: Sections 1, 1A, 7, 8, 15 show high dispersion across multiple clusters (diverse content)
- **Compact Clusters**: Sections 9A, 9B form tight clusters (standardized legal language)
- **Outlier Concentration**: Primary outliers in sections 1, 1A, 8, 15 indicating unique/non-standard content
- **Silhouette Score**: 0.0898 (k=20) representing optimal trade-off between compression and accuracy
- **Semantic Separation**: t-SNE reveals clear section-based clustering with some cross-section overlap due to shared legal terminology

**Analysis**: Low outlier rate (1.85%) indicates consistent semantic patterns across SEC filings. Visualizations reveal section-based clustering with outliers concentrated in diverse content areas (sections 1, 1A, 8, 15).

### 12.2 RAG Validation Results (Task 2)
| Metric | Performance |
|--------|-------------|
| Answer Accuracy | 100.0% (7/7 correct) |
| Chunk Recall | 85.7% (6/7 retrieved) |
| Top-1 Chunk Accuracy | 42.9% (3/7 exact match) |
| Average BERT F1 Score | 0.846 |
| BERT Pass Rate (>0.8) | 100.0% |
| Average Retrieval Score | 0.494 |

**Analysis**: 
- **Perfect Accuracy**: All 7 validation queries answered correctly including "not found" responses
- **Strong Retrieval**: 6/7 expected chunks retrieved; system correctly handles missing information
- **Semantic Quality**: BERT F1 >0.8 for complex queries demonstrates accurate content understanding
- **Ranking Gap**: Low top-1 accuracy (42.9%) indicates retrieval ranking needs improvement

### 12.3 Key Findings
- **Semantic Clustering**: t-SNE visualization reveals distinct section-based clusters
- **Outlier Distribution**: Primary outliers in sections 1, 1A, 8, 15 (diverse content areas)
- **RAG Effectiveness**: System successfully handles both factual and complex queries
- **Query Types**: Financial queries perform better than descriptive queries