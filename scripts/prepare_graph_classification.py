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
from tsp_sc.common.bodnar_utils import *
from torch_geometric.utils import from_networkx

dataset_name = "PROTEINS"
max_dim = 2
DATA_FOLDER = "data/graph-classification"
DATASET_PATH = os.path.join(DATA_FOLDER, f"raw/{dataset_name}")
PREPROC_FOLDER = os.path.join(DATA_FOLDER, f"preprocessed/{dataset_name}")

graphs = tud_to_networkx(DATASET_PATH, dataset_name)

num_graphs = len(graphs)
print(f"Loaded {num_graphs} graphs belonging to the {dataset_name} dataset")

print(f"Each graph has on average {count_triangles(graphs)} triangles")

graphs = [G for G in graphs if len(list_triangles(G)) >= 2]
num_graphs = len(graphs)
print(f"{num_graphs} graphs have at least 2 triangles")

for G in graphs:
    data = from_networkx(G)
    compute_clique_complex_with_gudhi()
    # complex_dim = G.number_of_nodes()
    #
    # simplex_tree = build_simplex_from_graph(G)
    # simplex_tables, id_maps = build_tables(simplex_tree, complex_dim)
    #
    # (
    #     boundaries_tables,
    #     boundaries,
    #     coboundaries,
    # ) = extract_boundaries_and_coboundaries_from_simplex_tree(
    #     simplex_tree, id_maps, simplex_tree.dimension()
    # )
    #
    # # Computes the adjacencies between all the simplexes in the complex
    # shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(
    #     boundaries, coboundaries, id_maps, complex_dim, include_down_adj=True
    # )
    #
    # # Construct features for the higher dimensions
    # # TODO: Make this handle edge features as well and add alternative options to compute this.
    # xs = construct_features(x, simplex_tables, init_method="")
    #
    # # Initialise the node / complex labels
    # v_y, complex_y = extract_labels(y, size)
    #
    # cochains = []
    # for i in range(complex_dim + 1):
    #     y = v_y if i == 0 else None
    #     cochain = generate_cochain(
    #         i,
    #         xs[i],
    #         upper_idx,
    #         lower_idx,
    #         shared_boundaries,
    #         shared_coboundaries,
    #         simplex_tables,
    #         boundaries_tables,
    #         complex_dim=complex_dim,
    #         y=y,
    #     )
    #     cochains.append(cochain)


# num_complexes = num_graphs
# num_nodes = (
#     sum(len(simplicial_complex[0]) for simplicial_complex in simplicial_complexes)
#     / num_complexes
# )
# num_edges = (
#     sum(len(simplicial_complex[1]) for simplicial_complex in simplicial_complexes)
#     / num_complexes
# )
# num_triangles = (
#     sum(len(simplicial_complex[2]) for simplicial_complex in simplicial_complexes)
#     / num_complexes
# )
# print(
#     f"There are on average {round(num_nodes)} nodes, {round(num_edges)} edges and {round(num_triangles)} triangles"
# )
#
# signals = []
#
# for G in graphs:
#     node_signal = np.ones(len(G.nodes()))
#     edge_signal = np.ones(len(G.edges()))
#     triangle_signal = np.ones(len(list_triangles(G)))
#
#     graph_signal = [node_signal, edge_signal, triangle_signal]
#     signals.append(graph_signal)
#
# boundaries = [
#     build_boundaries(simplicial_complex) for simplicial_complex in simplicial_complexes
# ]
# laplacians = [build_laplacians(sc_boundary) for sc_boundary in boundaries]
#
# assert len(laplacians) == len(boundaries) and len(boundaries) == len(signals)
#
# boundaries_path = os.path.join(PREPROC_FOLDER, f"boundaries.npy")
# laplacians_path = os.path.join(PREPROC_FOLDER, f"laplacians.npy")
# signals_path = os.path.join(PREPROC_FOLDER, f"signals.npy")
#
# np.save(laplacians_path, laplacians)
# np.save(boundaries_path, boundaries)
# np.save(signals_path, signals)
