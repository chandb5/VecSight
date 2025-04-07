# VectorDB with HNSW

VectorDB is a lightweight, fully local solution for vector embeddings and similarity search. This implementation extends the original [VectorDB](https://github.com/kagisearch/vectordb) with a focus on **HNSW** (Hierarchical Navigable Small World) — an efficient algorithm for approximate nearest neighbor search.

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

## Evaluating HNSW Performance

You can evaluate the performance of the HNSW algorithm on different datasets using the following command:

```bash
python evaluate.py --dataset_path datasets/siftsmall --algorithm hnsw --ground_truth True
```

### Command-line Parameters:

- `--dataset_path`: Path to the dataset (e.g., sift, siftsmall, gist).
- `--algorithm`: The algorithm to evaluate (`hnsw`, `mrpt`, `faiss`, `finetune-hnsw`).
- `--ground_truth`: Whether to compare results against ground truth data (`True/False`).
- `--use_chunking`: Use chunking for large datasets (`True/False`).

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

## License

MIT License.