# deep learning
import torch
import numpy as np

# misc
import os

# ad hoc
from tsp_sc.common.io import tud_to_networkx
from tsp_sc.common.simplices import (
    build_boundaries,
    build_laplacians,
    list_triangles,
    count_triangles,
    build_simplex_from_graph,
    extract_simplices,
)

dataset_name = "PROTEINS"
max_dim = 2
DATA_FOLDER = "data/graph_classification"
DATASET_PATH = os.path.join(DATA_FOLDER, f"raw/{dataset_name}")
PREPROC_FOLDER = os.path.join(DATA_FOLDER, "preprocessed")

graphs = tud_to_networkx(DATASET_PATH, dataset_name)
num_graphs = len(graphs)
print(f"Loaded {num_graphs} graphs belonging to the {dataset_name} dataset")

print(f"Each graph has on average {count_triangles(graphs)} triangles")

simplex_trees = [
    build_simplex_from_graph(G) for G in graphs if len(list_triangles(G)) >= 2
]

simplicial_complexes = [
    extract_simplices(simplex_tree, max_dim) for simplex_tree in simplex_trees
]

num_complexes = num_graphs
num_nodes = (
    sum(len(simplicial_complex[0]) for simplicial_complex in simplicial_complexes)
    / num_complexes
)
num_edges = (
    sum(len(simplicial_complex[1]) for simplicial_complex in simplicial_complexes)
    / num_complexes
)
num_triangles = (
    sum(len(simplicial_complex[2]) for simplicial_complex in simplicial_complexes)
    / num_complexes
)
print(
    f"There are on average {round(num_nodes)} nodes, {round(num_edges)} edges and {round(num_triangles)} triangles"
)

signals = []

for G in graphs:
    node_signal = np.ones(len(G.nodes()))
    edge_signal = np.ones(len(G.edges()))
    triangle_signal = np.ones(len(list_triangles(G)))

    graph_signal = [node_signal, edge_signal, triangle_signal]
    signals.append(graph_signal)

boundaries = [
    build_boundaries(simplicial_complex) for simplicial_complex in simplicial_complexes
]

laplacians = [build_laplacians(sc_boundary) for sc_boundary in boundaries]

boundaries_path = os.path.join(PREPROC_FOLDER, f"boundaries.npy")
laplacians_path = os.path.join(PREPROC_FOLDER, f"laplacians.npy")
signals_path = os.path.join(PREPROC_FOLDER, f"signals.npy")

np.save(laplacians_path, laplacians)
np.save(boundaries_path, boundaries)
np.save(signals_path, signals)
