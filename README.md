# VectorDB with HNSW

VectorDB is a lightweight, fully local solution for vector embeddings and similarity search. This implementation extends the original [VectorDB](https://github.com/kagisearch/vectordb) with a focus on **HNSW** (Hierarchical Navigable Small World) — an efficient algorithm for approximate nearest neighbor search.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [HNSW Algorithm Overview](#hnsw-algorithm-overview)
- [Using HNSW Directly](#using-hnsw-directly)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
  - [Running Performance Evaluations](#running-performance-evaluations)
  - [Understanding Results Structure](#understanding-results-structure)
  - [Generating Performance Plots](#generating-performance-plots)
  - [Fine-tuning HNSW Parameters](#fine-tuning-hnsw-parameters)
- [HNSW Parameter Impact](#hnsw-parameter-impact)
- [Visualizing HNSW Graphs](#visualizing-hnsw-graphs)
- [Testing HNSW Implementation](#testing-hnsw-implementation)
- [VectorDB Options](#vectordb-options)
  - [Memory Options](#memory-options)
  - [Memory Methods](#memory-methods)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

To install the required dependencies, use:

```bash
pip install -r requirements.txt
```

## Quick Start

Starter code is provided in the `driver.py` file. This demonstrates how the search works on the vector database by adding a few sentences and running a query to retrieve the most similar vector. The vectors are stored with their metadata.

To run the example:

```bash
python3 driver.py
```

## Datasets

The benchmark datasets used in this project are from [TEXMEX collection](http://corpus-texmex.irisa.fr/), which provides vector datasets specifically designed for nearest neighbor search evaluation:

- **siftsmall**: A small SIFT dataset (10K vectors) ideal for quick tests and experimentation
- **sift**: A larger SIFT dataset (1M vectors) for more comprehensive benchmarks
- **gist**: GIST dataset (1M vectors) for evaluating performance with higher-dimensional vectors

Each dataset includes:
- Base vectors (`*_base.fvecs`): The vectors to build the index from
- Query vectors (`*_query.fvecs`): The vectors to search for
- Ground truth (`*_groundtruth.ivecs`): True nearest neighbors for accuracy evaluation

For best results when getting started, use the `siftsmall` dataset as it provides a good balance between meaningful evaluation and quick processing time.

## HNSW Algorithm Overview

HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It builds a multi-layered graph structure for efficient traversal during search operations.

### Key Features:

- **Multi-layered Graph:** A hierarchy of layers for efficient navigation.
- **Customizable Parameters:** Tune `M` (connections per node) 
- **Distance Metrics:** Supports cosine and Euclidean distance metrics.
- **Bidirectional Connections:** Ensures robust graph traversal.

## Using HNSW Directly

To use HNSW for creating and searching an index, follow the example below:

```python
from hnsw import HNSW, Node

# Create an HNSW index with customizable parameters
hnsw_index = HNSW(space="cosine", M=16, ef_construction=200)

# Insert vectors with optional metadata
vectors = [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
]

for i, vector in enumerate(vectors):
    hnsw_index.insert(vector, metadata={"id": i})

# Search for nearest neighbors
query_vector = [0.8, 0.2]
query_node = Node(query_vector, -1)  # -1 indicates a node not in HNSW index
top_n = 5
results = hnsw_index.knn_search(query_node, top_n)

# Display results
for distance, node in results:
    print(f"Distance: {distance}, Vector: {node.vector}, Metadata: {node.metadata}")
```

## Evaluation and Benchmarking

### Running Performance Evaluations

You can evaluate the performance of the HNSW algorithm on different datasets using the following command:

```bash
python evaluate.py --dataset_path datasets/siftsmall --algorithm hnsw --ground_truth True
```

### Command-line Parameters:

- `--dataset_path`: Path to the dataset (e.g., sift, siftsmall, gist).
- `--algorithm`: The algorithm to evaluate (`hnsw`, `mrpt`, `faiss`, `finetune-hnsw`).
- `--ground_truth`: Whether to compare results against ground truth data (`True/False`).
- `--chunking`: Use chunking strategy for large datasets (`True/False`).
- `--chunk_size`: Size of each chunk when chunking is used (default: 1000).

> **IMPORTANT NOTE:** When using chunking, accuracy will initially be low because ground truth is based on the full dataset, while chunking evaluates on partial data. For consistent results, it's recommended to use the siftsmall dataset, which is small and easily manageable for evaluations.

### Understanding Results Structure

The results of the evaluation are organized in the `output/` directory with the following structure:

```
output/
├── hnsw/
│   └── siftsmall_size_scaling_metrics.csv  # Summary metrics for HNSW
├── faiss/
│   └── siftsmall_size_scaling_metrics.csv  # Summary metrics for Faiss
└── mrpt/
    └── siftsmall_size_scaling_metrics.csv  # Summary metrics for MRPT
```

For each algorithm, the evaluation process generates:

1. **Individual performance files**: For each vector count (when using chunking), a CSV file named `dataset_X_vectors_performance.csv` is created (e.g., `siftsmall_10000_vectors_performance.csv`). These files contain detailed per-query metrics including search time and accuracy.

2. **Size scaling metrics**: A summary file named `dataset_size_scaling_metrics.csv` (e.g., `siftsmall_size_scaling_metrics.csv`) that aggregates the performance metrics across different vector counts. This file contains three columns:
   - `vector_count`: Number of vectors used in the index
   - `accuracy`: Average accuracy across all queries
   - `query_time_ms`: Average query time in milliseconds

When fine-tuning HNSW parameters, results are saved in the `output/finetune/` directory with files named `hnsw_performance_M{m}_ef{ef}.csv` for each parameter combination.

### Generating Performance Plots

To generate performance comparison plots from the evaluation results, run:

```bash
python evaluate.py --plot-only True
```

This command reads the `_size_scaling_metrics.csv` files from each algorithm directory and generates comparative plots in the `plots/` directory. The plots show:

- Query time vs. vector count for each algorithm
- Performance scaling characteristics as dataset size increases

The generated plots are saved with timestamps (e.g., `siftsmall_query_time_20250407_210341.png`) to avoid overwriting previous results.

### Fine-tuning HNSW Parameters

To fine-tune HNSW parameters for optimal performance, run:

```bash
python evaluate.py --dataset_path datasets/siftsmall --algorithm finetune-hnsw
```

This will explore different values for `M` and `ef_construction` and save the results to the `final_output/finetune/` directory.

## HNSW Parameter Impact

- **M (Connections per node):** Controls the maximum number of connections. 
  - Higher values: Better recall, more memory usage.
  - Lower values: Less memory usage, potentially lower recall.
  - Recommended range: 5–40.

- **ef_construction:** Controls search accuracy during index construction.
  - Higher values: Better quality index, slower build times.
  - Lower values: Faster build times, potentially lower quality.
  - Recommended range: 100–450.

## Visualizing HNSW Graphs

You can visualize the layers and verify the integrity of the HNSW graph:

```python
from hnsw import HNSW
from utils.helper import plot_hnsw_layers, check_degree_limits, check_bidirectional_links

# Create and populate the HNSW index
hnsw_index = HNSW(M=16, ef_construction=200)

# Plot the graph layers
plot_hnsw_layers(hnsw_index.layered_graph, output_dir="images")

# Verify graph integrity
check_degree_limits(hnsw_index)
check_bidirectional_links(hnsw_index)
```

## Testing HNSW Implementation

Run unit tests to verify correct implementation:

```bash
python -m unittest discover hnsw/test
```

## VectorDB Options

### Memory Options

- **`memory_file`**: Optional path to persist memory to disk.
- **`chunking_strategy`**: Choose the chunking mode (default is `sliding_window`).
  
  Example configurations:
  - `{'mode': 'sliding_window', 'window_size': 240, 'overlap': 8}`
  - `{'mode': 'paragraph'}`

- **`embeddings`**: The model to use for embeddings. Options include:
  - `fast`: Uses Universal Sentence Encoder v4.
  - `normal`: Uses `BAAI/bge-small-en-v1.5` (default).
  - `best`: Uses `BAAI/bge-base-en-v1.5`.
  - `multilingual`: Uses Universal Sentence Encoder Multilingual Large v3.

### Memory Methods

- **`Memory.search(query, top_n=5, unique=False, batch_results="flatten")`**: Search inside memory.
  - `query`: Query text or list of queries.
  - `top_n`: Number of similar chunks to return (default: 5).
  - `unique`: Return only unique chunks (default: False).
  - `batch_results`: Options for handling results from a list of queries (`flatten` or `diverse`).

Example:

```python
from vectordb import Memory

memory = Memory()

memory.save(
    ["apples are green", "oranges are orange"],
    [{"url": "https://apples.com"}, {"url": "https://oranges.com"}],
)

query = "green"
results = memory.search(query, top_n=1)

print(results)
```

Output:

```json
[
  {
    "chunk": "apples are green",
    "metadata": {"url": "https://apples.com"},
    "distance": 0.87
  }
]
```

## Project Structure

```
vectorDB/
├── driver.py                   # Example script demonstrating vector search functionality
├── evaluate.py                 # Benchmarking script for algorithm evaluation
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
│
├── hnsw/                       # HNSW algorithm implementation
│   ├── __init__.py             # Package exports
│   ├── distance.py             # Distance metrics (cosine, euclidean) implementation
│   ├── hnsw.py                 # Main HNSW index implementation with insert and search
│   ├── node.py                 # Node class for graph vertices with vector data
│   ├── search.py               # Search algorithm implementation for HNSW graph
│   └── test/                   # Unit tests for HNSW components
│       ├── test_distance.py    # Tests for distance metrics
│       ├── test_hnsw.py        # Tests for HNSW index construction and search
│       ├── test_node.py        # Tests for Node class functionality
│       └── test_search.py      # Tests for search algorithm
│
├── datasets/                   # Benchmark datasets from TEXMEX collection
│   ├── siftsmall/              # Small SIFT descriptor dataset (~10K vectors)
│   ├── sift/                   # Full SIFT descriptor dataset (~1M vectors)
│   └── gist/                   # GIST descriptor dataset (higher dimensions)

├── utils/                      # Utility functions and helper classes
│   ├── __init__.py             # Package exports
│   ├── helper.py               # Helper functions for evaluation and plotting
│   └── reader.py               # Vector file reader for FVECS/IVECS formats
│
├── vectordb/                   # Core VectorDB implementation
│   ├── __init__.py             # Package exports
│   ├── chunking.py             # Text chunking strategies
│   ├── embedding.py            # Vector embedding generation
│   ├── memory.py               # Main Memory class for vector storage
│   ├── storage.py              # Persistence layer for vectors
│   └── vector_search.py        # Vector search implementations (MRPT, FAISS)
│
├── output/                     # Evaluation results directory
│   ├── hnsw/                   # HNSW performance metrics
│   ├── faiss/                  # FAISS performance metrics
│   └── mrpt/                   # MRPT performance metrics
│
├── plots/                      # Generated performance comparison plots
│   └── siftsmall_query_time_*.png # Time vs. vector count plots
│
└── images/                     # Documentation images and visualizations
```

## License

MIT License.