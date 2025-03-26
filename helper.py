from hnsw import HNSW
from typing import List

import csv
import matplotlib.pyplot as plt
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


def get_all_distance_linear(query: List[float], hnsw_obj: HNSW, level: int):
    for node in hnsw_obj.layered_graph[level]:
        print(f"Distance between query and node {node.vector}: {node.distance(query)}")


def write_csv(data, filename, fieldnames=None):
    if isinstance(data, dict):
        data = [{"query_id": k, **v} for k, v in data.items()]

    # Determine fieldnames if not provided
    if fieldnames is None:
        # Use keys from the first dictionary
        fieldnames = list(data[0].keys())

    # Write to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write rows
        writer.writerows(data)

    print(f"Data written to {filename}")
