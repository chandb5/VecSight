import time
import threading
import copy
import argparse
from typing import Dict, List, Optional, Tuple

from utils import Reader
from vectordb import Memory
from hnsw import HNSW, Node
from utils.helper import write_csv, summarize_performance


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

        output_path = f"./final_output/finetune/hnsw_performance_M{m}_ef{ef}.csv"
        write_csv(best_result, output_path)
        print(f"  Best result saved to {output_path}")

    for m in value_M:
        for ef in value_ef:
            evaluate_parameter_pair(m, ef)


def process_chunk(
    algorithm: str,
    dataset_base_path: str,
    chunk_index: int,
    chunk_size: int,
    require_ground_truth: bool,
    hnsw_obj: Optional[HNSW] = None,
) -> Tuple[Dict[int, Dict[str, float]], Optional[HNSW]]:
    """
    Process a single chunk of data for evaluation.

    Args:
        algorithm (str): Algorithm to evaluate
        dataset_base_path (str): Path to dataset
        dataset (str): Dataset name
        chunk_index (int): Index of current chunk
        chunk_size (int): Size of each chunk
        require_ground_truth (bool): Whether ground truths are required
        hnsw_obj (Optional[HNSW]): Existing HNSW object for continued building

    Returns:
        Tuple of results dictionary and updated HNSW object (if applicable)
    """
    chunk_num = chunk_index + 1
    print(
        f"\nProcessing chunk {chunk_num} [{chunk_index * chunk_size}:{(chunk_index + 1) * chunk_size}]"
    )

    start_idx = chunk_index * chunk_size
    end_idx = (chunk_index + 1) * chunk_size

    memory, query_reader, ground_truths = load_data(dataset_base_path, size=end_idx)

    results = None

    if algorithm == "hnsw":
        hnsw_obj = build_hnsw_index(
            memory,
            M=12,
            ef_construction=250,
            hnsw_obj=hnsw_obj,
            start=start_idx,
            end=end_idx,
        )

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

    # Clean up memory
    del memory
    del query_reader
    del ground_truths

    return results, hnsw_obj


def main(
    dataset_base_path: str,
    algorithm: str = "hnsw",
    require_ground_truth: bool = True,
    use_chunking: bool = False,
) -> None:
    """
    Main function to run algorithm performance evaluation.

    Args:
        dataset_base_path (str): Path to dataset
        algorithm (str): Algorithm to evaluate (hnsw, mrpt, faiss, finetune-hnsw)
        require_ground_truth (bool): Whether ground truths are required
        use_chunking (bool): Whether to use chunking strategy
    """
    dataset = dataset_base_path.split("/")[-1]
    print(f"Evaluating {algorithm.upper()} algorithm on {dataset} dataset")

    # Handle fine-tuning separately as it doesn't use chunking
    if algorithm == "finetune-hnsw":
        print("Running HNSW parameter fine-tuning")
        memory, query_reader, ground_truths = load_data(dataset_base_path)
        finetune_hnsw_parameters(
            memory, query_reader, ground_truths, require_ground_truth=require_ground_truth
        )
        print("Fine-tuning completed")
        return

    # Determine chunking strategy
    if use_chunking:
        chunk_size = 1000
        num_chunks = 20
        print(
            f"Using chunking strategy: {num_chunks} chunks of {chunk_size} vectors each"
        )

        hnsw_obj = None
        for i in range(num_chunks):
            results, hnsw_obj = process_chunk(
                algorithm,
                dataset_base_path,
                dataset,
                i,
                chunk_size,
                require_ground_truth,
                hnsw_obj,
            )

            # Save results
            output_path = f"./final_output/{algorithm}/{dataset}_{i+1}_performance.csv"
            write_csv(results, output_path)

            # Print summary
            avg_accuracy, avg_time = summarize_performance(results)
            print(
                f"Chunk {i+1} Summary: Accuracy={avg_accuracy:.2f}%, Time={avg_time:.2f}ms"
            )
    else:
        # Process all data at once
        print("Processing complete dataset (no chunking)")

        memory, query_reader, ground_truths = load_data(dataset_base_path)

        if algorithm == "hnsw":
            hnsw_obj = build_hnsw_index(
                memory,
                M=12,
                ef_construction=250,
            )
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

        # Save results
        output_path = f"./final_output/{algorithm}/{dataset}_full_performance.csv"
        write_csv(results, output_path)
        print(f"Results saved to {output_path}")

        avg_accuracy, avg_time = summarize_performance(results)
        print(f"Final Summary: Accuracy={avg_accuracy:.2f}%, Time={avg_time:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vector search algorithms.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset base directory",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["hnsw", "mrpt", "faiss", "finetune-hnsw"],
        default="hnsw",
        required=True,
        help="Algorithm to evaluate (hnsw, mrpt, faiss, finetune-hnsw)",
    )
    parser.add_argument(
        "--ground_truth",
        type=bool,
        default=True,
        help="Compare ground truth data for evaluation of Recall",
    )
    parser.add_argument(
        "--use_chunking",
        type=bool,
        default=False,
        help="Use chunking strategy for large datasets",
    )

    args = parser.parse_args()

    main(
        dataset_base_path=args.dataset_path,
        algorithm=args.algorithm,
        require_ground_truth=args.ground_truth,
        use_chunking=args.use_chunking,
    )
