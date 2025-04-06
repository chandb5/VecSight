# VectorDB with HNSW

VectorDB is a lightweight, fully local solution for vector embeddings and similarity search. This implementation extends the original VectorDB with a focus on HNSW (Hierarchical Navigable Small World) - an efficient algorithm for approximate nearest neighbor search.

## Installation

To install VectorDB, use pip:

```bash
pip install vectordb2
```

Or install requirements for development:

```bash
pip install -r requirements.txt
```

## Quick Example

```python
from vectordb import Memory

# Memory is where all content you want to store/search goes.
memory = Memory()

memory.save(
    ["apples are green", "oranges are orange"],  # save your text content
    [{"url": "https://apples.com"}, {"url": "https://oranges.com"}], # associate metadata
)

# Search for top n relevant results, automatically using embeddings
query = "green"
results = memory.search(query, top_n = 1)

print(results)
```

Result:
```json
[
  {
    "chunk": "apples are green",
    "metadata": {"url": "https://apples.com"},
    "distance": 0.87
  }
]
```

## HNSW Algorithm Overview

Hierarchical Navigable Small World (HNSW) is a graph-based algorithm for approximate nearest neighbor search. It creates a multi-layered graph structure that enables efficient traversal during search operations.

Key features of our HNSW implementation:

- **Multi-layered graph structure:** Creates a hierarchy of layers for efficient navigation
- **Customizable parameters:** Tune M (connections per node) and ef_construction (search width during construction)
- **Distance metrics:** Supports both cosine and Euclidean distance metrics
- **Bidirectional connections:** Ensures robustness and effective graph traversal

## Using HNSW Directly

```python
from hnsw import HNSW, Node

# Create an HNSW index
# Parameters:
# - space: Distance metric ("cosine" or "euclidean")
# - M: Maximum number of connections per node (higher = better recall, more memory)
# - ef_construction: Size of dynamic candidate list during construction (higher = better quality, slower build)
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
query_node = Node(query_vector, -1)  # -1 level is used for query nodes
top_n = 5
results = hnsw_index.knn_search(query_node, top_n)

# Process results
for distance, node in results:
    print(f"Distance: {distance}, Vector: {node.vector}, Metadata: {node.metadata}")
```

## Evaluating HNSW Performance

Use the evaluation script to test HNSW performance against different datasets:

```bash
python evaluate.py --dataset_path datasets/siftsmall --algorithm hnsw --ground_truth True
```

### Command-line Parameters:

- `--dataset_path`: Path to dataset (sift, siftsmall, or gist)
- `--algorithm`: Algorithm to evaluate (hnsw, mrpt, faiss, finetune-hnsw)
- `--ground_truth`: Whether to compare against ground truth data (True/False)
- `--use_chunking`: Use chunking for large datasets (True/False)

### Fine-tuning HNSW Parameters

To find optimal HNSW parameters for your dataset:

```bash
python evaluate.py --dataset_path datasets/siftsmall --algorithm finetune-hnsw
```

This will test different combinations of M (graph connectivity) and ef_construction (build quality) parameters and save results to `final_output/finetune/`.

## HNSW Parameter Impact

- **M parameter**: Controls the maximum number of connections per node
  - Higher values: Better recall, more memory usage
  - Lower values: Less memory usage, potentially lower recall
  - Recommended range: 5-40

- **ef_construction**: Controls search accuracy during index construction
  - Higher values: Better quality index, slower build times
  - Lower values: Faster builds, potentially lower quality
  - Recommended range: 100-450

- **ef search**: Controls search accuracy during query time (dynamically set)

## Visualizing HNSW Graphs

The utils module includes tools to visualize HNSW layers:

```python
from hnsw import HNSW
from utils.helper import plot_hnsw_layers, check_degree_limits, check_bidirectional_links

# Create and populate HNSW index
hnsw_index = HNSW(M=16, ef_construction=200)
# ... insert vectors ...

# Plot the graph layers
plot_hnsw_layers(hnsw_index.layered_graph, output_dir="images")

# Verify graph integrity
check_degree_limits(hnsw_index)
check_bidirectional_links(hnsw_index)
```

## Testing HNSW Implementation

Run the unit tests to verify the correct implementation:

```bash
python -m unittest discover hnsw/test
```

## VectorDB Options

**Memory(memory_file=None, chunking_strategy={"mode":"sliding_window"}, embeddings="normal")**

- `memory_file`: *Optional.* Path to the memory file. If provided, memory will persist to disk.
- `chunking_strategy`: *Optional.* Dictionary containing the chunking mode.
  
   Options:
  `{'mode':'sliding_window', 'window_size': 240, 'overlap': 8}`   (default)
  `{'mode':'paragraph'}`

- `embeddings`: *Optional.* 
  
   Options:
   `fast` - Uses Universal Sentence Encoder 4
   `normal` - Uses "BAAI/bge-small-en-v1.5" (default)
   `best` - Uses "BAAI/bge-base-en-v1.5"
   `multilingual` - Uses Universal Sentence Encoder Multilingual Large 3

   You can also specify a custom HuggingFace model by name.

**Memory.search(query, top_n=5, unique=False, batch_results="flatten")**

Search inside memory.

- `query`: *Required.* Query text or list of queries.
- `top_n`: *Optional.* Number of most similar chunks to return (default: 5).
- `unique`: *Optional.* Return only unique original texts (default: False).
- `batch_results`: *Optional.* When input is a list, can be "flatten" or "diverse" (default: "flatten").

## Vector Search Performance Analysis

VectorDB optimizes for speed of retrieval. It automatically uses [Faiss](https://github.com/facebookresearch/faiss) for low number of chunks (<4000), [mrpt](https://github.com/vioshyvo/mrpt) for high number of chunks, and now also supports HNSW for highly efficient approximate nearest neighbor search.

Performance benchmarks for HNSW with different parameter settings are available in the `output/` directory.

## License

MIT License.
