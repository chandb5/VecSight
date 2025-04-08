import time
import threading
import copy
import math
import argparse
from typing import Dict, List, Optional, Tuple

from utils import Reader
from vectordb import Memory
from hnsw import HNSW, Node
from utils.helper import get_count_of_vectors, write_csv, summarize_performance, performance_plotter

OUTPUT_DIR = "./output"
PLOT_DIR = "./plots"

def load_data(
    base_path: str, size: Optional[int] = None
) -> Tuple[Memory, Reader, Reader]:
    """
    Load vector databases and ground truth data.

    Args:
        base_path (str): Base path for dataset files
        size (Optional[int]): Number of vectors to load (None for all)

    Returns:
        tuple: Loaded memory, query reader, and ground truth reader
    """
    memory = Memory(load_vec_files=True)
    dataset_name = base_path.split("/")[-1]
    memory.load_vector_files(
        f"{base_path}/{dataset_name}_base.fvecs", number_of_vectors=size
    )

    query_reader = Reader(f"{base_path}/{dataset_name}_query.fvecs")
    query_reader.read()

    ground_truths = Reader(f"{base_path}/{dataset_name}_groundtruth.ivecs")
    ground_truths.read()

    return memory, query_reader, ground_truths


def build_hnsw_index(
    memory: Memory,
    M: int,
    ef_construction: int,
    hnsw_obj: Optional[HNSW] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> HNSW:
    """
    Build HNSW index from memory vectors.

    Args:
        memory (Memory): Vector memory
        M (int): HNSW parameter for graph connectivity
        ef_construction (int): HNSW parameter for construction efficiency
        hnsw_obj (Optional[HNSW]): Existing HNSW object to continue building
        start (Optional[int]): Start index for vectors to include
        end (Optional[int]): End index for vectors to include

    Returns:
        HNSW: Constructed HNSW index
    """
    if hnsw_obj is None:
        hnsw_obj = HNSW(M=M, ef_construction=ef_construction)

    if start is None and end is None:
        start = 0
        end = len(memory.memory)

    memory_slice = memory.memory[start:end]
    vector_count = len(memory_slice)
    print(f"Building HNSW index with {vector_count} vectors [{start}:{end}]")

    chunk_size = vector_count // 5

    def load_vectors(vector_chunk, chunk_idx):
        chunk_start_idx = (chunk_idx - 1) * len(vector_chunk)
        print(f"Loading chunk {chunk_idx}/5: {len(vector_chunk)} vectors")

        for i, embedding in enumerate(vector_chunk):
            hnsw_obj.insert(
                embedding["embedding"], metadata={"id": i + chunk_start_idx}
            )
            if i % 5000 == 0 and i > 0:
                print(
                    f"  Progress: {i}/{len(vector_chunk)} vectors in chunk {chunk_idx}"
                )

    # Split the data into 5 chunks for parallel processing
    chunks = [
        (
            memory_slice[i * chunk_size : (i + 1) * chunk_size]
            if i < 4
            else memory_slice[4 * chunk_size :]
        )
        for i in range(5)
    ]

    threads = [
        threading.Thread(target=load_vectors, args=(chunks[i], i + 1)) for i in range(5)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print("HNSW index construction completed")

    return hnsw_obj


def evaluate_hnsw_performance(
    hnsw_obj: HNSW,
    queries: Reader,
    ground_truths: Reader,
    require_ground_truths: bool = False,
    k: int = 100,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate HNSW index performance for k-nearest neighbor search.

    Args:
        hnsw_obj (HNSW): Constructed HNSW index
        queries (Reader): Query vector reader
        ground_truths (Reader): Ground truth labels
        require_ground_truths (bool): Whether ground truths are required
        k (int): Number of nearest neighbors to search

    Returns:
        Dict of performance metrics for each query
    """
    results = {}
    query_count = queries.data.shape[0]
    print(f"Evaluating {query_count} queries with k={k}")

    for idx, query_vector in enumerate(queries.data):
        query_node = Node(query_vector, -1)

        start_time = time.time()
        search_results = hnsw_obj.knn_search(query_node, k)
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Calculate accuracy if ground truth is required
        accuracy = 0
        if require_ground_truths:
            ground_truth_set = set(ground_truths.data[idx].tolist())
            accuracy = sum(
                1 for res in search_results if res[1].metadata["id"] in ground_truth_set
            )

        results[idx] = {
            "time": search_time,
            "accuracy": accuracy,
        }

        if idx % 1000 == 0 and idx > 0:
            print(f"  Progress: {idx}/{query_count} queries evaluated")

    return results


def evaluate_existing_algorithm(
    memory: Memory,
    queries: Reader,
    ground_truths: Reader,
    k: int = 100,
    variant: str = "mrpt",
    require_ground_truth: bool = False,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate existing algorithm performance for k-nearest neighbor search.

    Args:
        memory (Memory): Vector memory
        queries (Reader): Query vector reader
        ground_truths (Reader): Ground truth labels
        k (int): Number of nearest neighbors to search
        variant (str): Algorithm variant to use (mrpt or faiss)
        require_ground_truth (bool): Whether ground truths are required

    Returns:
        Dict of performance metrics for each query
    """
    results = {}
    query_count = queries.data.shape[0]
    print(f"Evaluating {variant.upper()} algorithm on {query_count} queries with k={k}")

    for query_idx, query in enumerate(queries.data):
        start_time = time.time()
        query_results = memory.search_vector(query, top_k=k, preference=variant)
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        accuracy = 0
        if require_ground_truth:
            accuracy = sum(
                1
                for idx, res in enumerate(query_results)
                if str(ground_truths.data[query_idx][idx]) in res["chunk"]
            )

        results[query_idx] = {
            "time": search_time,
            "accuracy": accuracy,
        }

        if query_idx % 500 == 0 and query_idx > 0:
            print(f"  Progress: {query_idx}/{query_count} queries evaluated")

    return results


def finetune_hnsw_parameters(
    memory: Memory,
    query_reader: Reader,
    ground_truths: Reader,
    value_M: List[int] = [5, 10, 15, 20, 25, 30, 35, 40],
    value_ef: List[int] = [100, 200, 300, 400, 450],
    require_ground_truth: bool = False,
) -> None:
    """
    Fine-tune HNSW parameters M and ef_construction for optimal performance.

    Args:
        memory (Memory): Vector memory
        query_reader (Reader): Query vector reader
        ground_truths (Reader): Ground truth labels
        value_M (List[int]): List of M values to test
        value_ef (List[int]): List of ef_construction values to test
    """

    def evaluate_parameter_pair(m: int, ef: int) -> None:
        max_accuracy = None
        best_result = None

        print(f"Testing parameters: M={m}, ef_construction={ef}")

        for iteration in range(2):
            print(f"  Iteration {iteration+1}/2")
            hnsw_obj = build_hnsw_index(memory, M=m, ef_construction=ef)

            result = evaluate_hnsw_performance(
                hnsw_obj,
                query_reader,
                ground_truths,
                require_ground_truths=require_ground_truth,
                k=100,
            )
            avg_accuracy, avg_time = summarize_performance(result)
            print(f"  Results: Accuracy={avg_accuracy:.2f}%, Time={avg_time:.2f}ms")

            if max_accuracy is None or avg_accuracy > max_accuracy:
                print("  → New best result")
                best_result = copy.deepcopy(result)
                max_accuracy = avg_accuracy

        output_path = f"{OUTPUT_DIR}/finetune/hnsw_performance_M{m}_ef{ef}.csv"
        write_csv(best_result, output_path)
        print(f"  Best result saved to {output_path}")

    for m in value_M:
        for ef in value_ef:
            evaluate_parameter_pair(m, ef)

def main(
    dataset_base_path: str,
    algorithm: str = "hnsw",
    require_ground_truth: bool = True,
    use_chunking: bool = False,
    chunk_size: int = 1000,
) -> None:
    """
    Main function to run algorithm performance evaluation.
    Evaluates performance incrementally at each chunk size.

    Args:
        dataset_base_path (str): Path to dataset
        algorithm (str): Algorithm to evaluate (hnsw, mrpt, faiss, finetune-hnsw)
        require_ground_truth (bool): Whether ground truths are required
        use_chunking (bool): Whether to use chunking strategy
        chunk_size (int): Size of each chunk when chunking is used
    """
    dataset = dataset_base_path.split("/")[-1]
    print(f"Evaluating {algorithm.upper()} algorithm on {dataset} dataset")

    # Handle fine-tuning separately
    if algorithm == "finetune-hnsw":
        print("Running HNSW parameter fine-tuning")
        memory, query_reader, ground_truths = load_data(dataset_base_path)
        finetune_hnsw_parameters(
            memory, query_reader, ground_truths, require_ground_truth=require_ground_truth
        )
        print("Fine-tuning completed")
        return

    # If not using chunking, just evaluate once with all data
    if not use_chunking:
        print("Processing complete dataset (no chunking)")
        memory, query_reader, ground_truths = load_data(dataset_base_path)
        
        if algorithm == "hnsw":
            hnsw_obj = build_hnsw_index(memory, M=12, ef_construction=250)
            results = evaluate_hnsw_performance(
                hnsw_obj,
                query_reader,
                ground_truths,
                require_ground_truths=require_ground_truth,
                k=100,
            )
        else:  # mrpt or faiss
            results = evaluate_existing_algorithm(
                memory,
                query_reader,
                ground_truths,
                require_ground_truth=require_ground_truth,
                variant=algorithm,
            )
        
        output_path = f"{OUTPUT_DIR}/{algorithm}/{dataset}_full_performance.csv"
        write_csv(results, output_path)
        avg_accuracy, avg_time = summarize_performance(results)
        print(f"Final Summary: Accuracy={avg_accuracy:.2f}%, Time={avg_time:.2f}ms")
        return

    # Using chunking - evaluate at each incremental chunk size
    total_vectors = get_count_of_vectors(dataset_base_path)
    num_chunks = math.ceil(total_vectors / chunk_size)
    print(f"Using chunking strategy: {num_chunks} chunks of {chunk_size} vectors each")
    
    # Load all queries and ground truths once
    memory, query_reader, ground_truths = load_data(dataset_base_path, size=None)
    
    # Track chunk performance metrics
    chunk_metrics = {}
    
    # For HNSW, build incrementally but evaluate after each chunk
    if algorithm == "hnsw":
        hnsw_obj = HNSW(M=12, ef_construction=250)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_vectors)
            vectors_so_far = end_idx
            print(f"Processing chunk {chunk_idx+1}/{num_chunks} [{start_idx}:{end_idx}]")
            
            # Add new vectors from current chunk to index
            memory_slice = memory.memory[start_idx:end_idx]
            for i, embedding in enumerate(memory_slice):
                global_id = start_idx + i
                hnsw_obj.insert(embedding["embedding"], metadata={"id": global_id})
                
                if i % 5000 == 0 and i > 0:
                    print(f"  Progress: {i}/{len(memory_slice)} vectors in chunk {chunk_idx+1}")
            
            # Evaluate index with current size
            print(f"Evaluating HNSW index with {vectors_so_far} vectors")
            results = evaluate_hnsw_performance(
                hnsw_obj,
                query_reader,
                ground_truths,
                require_ground_truths=require_ground_truth,
                k=100,
            )
            
            output_path = f"{OUTPUT_DIR}/{algorithm}/{dataset}_{vectors_so_far}_vectors_performance.csv"
            write_csv(results, output_path)
            
            avg_accuracy, avg_time = summarize_performance(results)
            chunk_metrics[vectors_so_far] = {
                "vectors": vectors_so_far,
                "accuracy": avg_accuracy,
                "query_time": avg_time
            }
            print(f"Size {vectors_so_far} vectors: Accuracy={avg_accuracy:.2f}%, Time={avg_time:.2f}ms")
    
    else:  # For MRPT or FAISS
        for chunk_idx in range(num_chunks):
            end_idx = min((chunk_idx + 1) * chunk_size, total_vectors)
            vectors_so_far = end_idx
            print(f"Processing with {vectors_so_far} vectors")
            
            # Load data up to this size
            memory, _, _ = load_data(dataset_base_path, size=end_idx)
            
            # Evaluate algorithm at current size
            results = evaluate_existing_algorithm(
                memory,
                query_reader,
                ground_truths,
                require_ground_truth=require_ground_truth,
                variant=algorithm,
            )
            
            # Save results for this size
            output_path = f"{OUTPUT_DIR}/{algorithm}/{dataset}_{vectors_so_far}_vectors_performance.csv"
            write_csv(results, output_path)
            
            # Calculate and store metrics
            avg_accuracy, avg_time = summarize_performance(results)
            chunk_metrics[vectors_so_far] = {
                "vectors": vectors_so_far,
                "accuracy": avg_accuracy,
                "query_time": avg_time
            }
            print(f"Size {vectors_so_far} vectors: Accuracy={avg_accuracy:.2f}%, Time={avg_time:.2f}ms")
            
            # Clean up memory
            del memory
    
    # Save summary metrics for plotting
    metrics_output_path = f"{OUTPUT_DIR}/{algorithm}/{dataset}_size_scaling_metrics.csv"
    with open(metrics_output_path, 'w') as f:
        f.write("vector_count,accuracy,query_time_ms\n")
        for size, metrics in sorted(chunk_metrics.items()):
            f.write(f"{metrics['vectors']},{metrics['accuracy']:.4f},{metrics['query_time']:.4f}\n")
    
    print(f"Size scaling metrics saved to {metrics_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vector search algorithms.")
    parser.add_argument(
        "--plot-only",
        type=bool,
        default=False,
        help="Generate performance comparison plots without evaluation",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/siftsmall",
        help="Path to the dataset base directory",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["hnsw", "mrpt", "faiss", "finetune-hnsw"],
        default="hnsw",
        required=False,
        help="Algorithm to evaluate (hnsw, mrpt, faiss, finetune-hnsw)",
    )
    parser.add_argument(
        "--ground_truth",
        type=bool,
        default=True,
        help="Compare ground truth data for evaluation of Recall",
    )
    parser.add_argument(
        "--chunking",
        type=bool,
        default=False,
        help="Use chunking strategy for large datasets",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of each chunk when chunking is used",
    )

    args = parser.parse_args()

    if args.plot_only:
        print("Plotting performance comparison graphs")
        performance_plotter(OUTPUT_DIR, save_dir=PLOT_DIR)
    else:
        main(
            dataset_base_path=args.dataset_path,
            algorithm=args.algorithm,
            require_ground_truth=args.ground_truth,
            use_chunking=args.chunking,
            chunk_size=args.chunk_size,
        )