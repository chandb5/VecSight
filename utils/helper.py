from hnsw import HNSW
from typing import Dict
from collections import defaultdict

import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import string


def check_degree_limits(hnsw: HNSW):
    for node in hnsw.layered_graph[0]:  # assumes all nodes are at least in layer 0
        for layer, neighbors in node.neighbors.items():
            allowed = hnsw.m0 if layer == 0 else hnsw.mMax
            actual = len(neighbors)
            if actual > allowed:
                print(
                    f"❌ Node {node.vector} exceeds limit at layer {layer}: {actual} > {allowed}"
                )
                return False
    print("✅ All nodes respect m, m0, mMax limits.")
    return True


def check_bidirectional_links(hnsw: HNSW):
    for node in hnsw.layered_graph[0]:
        for layer, neighbors in node.neighbors.items():
            print(
                f"Node: {node.vector}, Layer: {layer}, Neighbors: {[n.vector for n in neighbors]}"
            )
            for neighbor in neighbors:
                if node not in neighbor.neighbors.get(layer, []):
                    print(
                        f"❌ Missing back link: {node.vector} -> {neighbor.vector} at layer {layer}"
                    )
                    return False
    print("✅ All bidirectional links are intact.")
    return True


def get_color(level):
    if level == 0:
        return "#fd7f6f"
    elif level == 1:
        return "#7eb0d5"
    elif level == 2:
        return "#b2e061"
    elif level == 3:
        return "#bd7ebe"
    else:
        return "#ffee65"


def plot_hnsw_layers(layered_graph, output_dir=".", xlim=25, ylim=25):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    node_ids = {}
    label_counter = 0
    alphabet = string.ascii_uppercase

    # Assign a unique label (A, B, C, ...) to each node
    all_nodes = set()
    for nodes in layered_graph.values():
        all_nodes.update(nodes)
    for node in all_nodes:
        label = alphabet[label_counter % 26]
        if label_counter >= 26:
            label += str(label_counter // 26)
        node_ids[node] = label
        label_counter += 1

    for layer in sorted(layered_graph.keys(), reverse=True):
        nodes = layered_graph[layer]
        plt.figure(figsize=(6, 6))
        plt.grid(False)
        # plt.xticks([])
        # plt.yticks([])
        plt.title(f"HNSW Graph - Layer {layer}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)

        # Plot nodes and edges to their neighbors
        for node in nodes:
            x, y = node.vector
            label = f"{node_ids[node]}"  # ({x:.2f}, {y:.2f})"
            plt.scatter(x, y, color=get_color(node.level), s=100, zorder=2)
            plt.text(x, y, label, fontsize=9, zorder=3, color="black")

            for neighbor in node.neighbors.get(layer, []):
                if neighbor in nodes:
                    nx, ny = neighbor.vector
                    plt.plot([x, nx], [y, ny], color="gray", linewidth=1, zorder=1)

        plt.grid(True)
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"layer{layer}.png")
        plt.savefig(filepath)
        plt.close()

def write_csv(data, filename, fieldnames=None):
    if isinstance(data, dict):
        data = [{"query_id": k, **v} for k, v in data.items()]

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write rows
        writer.writerows(data)

    print(f"Data written to {filename}")

def summarize_performance(performance: Dict[int, Dict[str, float]]):
    avg_accuracy = 0
    avg_time = 0
    for idx, res in performance.items():
        avg_accuracy += res["accuracy"]
        avg_time += res["time"]
    avg_accuracy /= len(performance)
    avg_time /= len(performance)

    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average Time: {avg_time} ms")
    return avg_accuracy, avg_time

def get_count_of_vectors(dataset_base_path: str) -> int:
    """
    Get the count of vectors in the dataset.
    :param dataset_base_path: Path to the dataset directory.
    :param query_file: Name of the query file.
    :return: Count of vectors in the dataset.
    """
    filename = dataset_base_path.split("/")[-1] + "_base.fvecs"
    vector_file = os.path.join(dataset_base_path, filename)
    fv = np.fromfile(vector_file, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    return fv.size // (1 + dim) 

def performance_plotter(output_dir, save_dir="plots"):
    """
    Generate simple performance comparison plots.
    
    Args:
        output_dir: Directory containing algorithm result folders
        save_dir: Directory to save generated plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data = defaultdict(dict)

    for algo in os.listdir(output_dir):
        path = os.path.join(output_dir, algo)
        if not os.path.isdir(path):
            continue
        for file in os.listdir(path):
            if not file.endswith('_size_scaling_metrics.csv'):
                continue
            match = re.match(r'(.+)_size_scaling_metrics\.csv', file)
            if not match:
                continue
            dataset = match.group(1)
            x, y = [], []
            with open(os.path.join(path, file)) as f:
                for row in csv.DictReader(f):
                    x.append(int(row['vector_count']))
                    y.append(float(row['query_time_ms']))
            data[dataset][algo] = (x, y)

    for dataset, algos in data.items():
        plt.figure(figsize=(10, 6))
        for algo, (x, y) in algos.items():
            xy = sorted(zip(x, y))
            x_sorted, y_sorted = zip(*xy)
            plt.plot(x_sorted, y_sorted, 'o-', label=algo.upper())
        plt.title(f"{dataset.upper()} - Query Time vs Vector Count")
        plt.xlabel("Vector Count")
        plt.ylabel("Query Time (ms)")
        plt.yscale("log", base=2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{dataset}_query_time_{timestamp}.png", dpi=300)
        plt.close()