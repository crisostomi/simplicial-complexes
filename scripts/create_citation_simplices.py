import os

import torch
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import pandas as pd

# graphs

# TDA

# Graph Signal Processing

# plots

# ad hoc
from inter_order.utils.simplices import (
    build_laplacians,
    build_boundaries,
)

from citations.utils.citations import *

# reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

# paths

DATA_FOLDER = os.path.join("data/citations")
OUT_FOLDER = os.path.join(DATA_FOLDER, "output")
RAW_DATA_FOLD = os.path.join(DATA_FOLDER, "raw")
RAW_GRAPH_FOLD = os.path.join(DATA_FOLDER, "bipartite_graph")

adj_papers_path = os.path.join(RAW_GRAPH_FOLD, "papers_adjacency.npz")
adj_authors_path = os.path.join(RAW_GRAPH_FOLD, "authors_adjacency.npz")
biadjacency_matrix_path = os.path.join(RAW_GRAPH_FOLD, "paper_author_biadjacency.npz")

COMPLEX_FOLDER = os.path.join(DATA_FOLDER, "collaboration_complex")

# 1. Create bipartite graph from Semantic Scholar

# process papers

corpus_path = os.path.join(RAW_DATA_FOLD, "to_use/s2-corpus-*.gz")
papers, edges = process_papers(corpus_path)

# count citations

count_citations(papers, range(1994, 2024, 5))

graph_full_path = os.path.join(RAW_GRAPH_FOLD, "paper_author_full.pickle")

save(papers, edges, graph_full_path)

# 2. Clean and downsample bipartite graph

papers, edges = load_papers_authors_bipartite_graph(graph_full_path)

papers, edges = clean(papers, edges)

papers, edges = downsample(papers, edges)

print(papers.head(n=3))
print(edges.head(n=3))

# map papers and authors to node ids

papers, authors, edges = add_node_ids(papers, edges)

print(papers.head(n=3))
print(edges.head(n=3))
print(authors.head(n=3))

# build biadjacency matrix

biadjacency = sparse.coo_matrix(
    (
        np.ones(len(edges), dtype=np.bool),
        (edges["paper_node_id"], edges["author_node_id"]),
    )
)

# save
save_paper_author_biadj(papers, authors, edges, biadjacency, RAW_GRAPH_FOLD)

# 3. Project bipartite graph

bipartite = load_paper_author_biadj(biadjacency_matrix_path)

graph_papers, graph_authors = project(bipartite)

save_projected_graphs(graph_papers, graph_authors, adj_papers_path, adj_authors_path)

# 4. Bipartite to downsampled

adjacency_papers = sparse.load_npz(adj_papers_path)
adjacency = scipy.sparse.load_npz(biadjacency_matrix_path)

papers_df_path = os.path.join(RAW_GRAPH_FOLD, "papers.csv")
papers_df = pd.read_csv(papers_df_path, index_col=0)

# obtain starting node

# shape (num_papers, )
citations = np.array(papers_df["citations_2019"])

starting_node = 150250
# starting_node = starting_node_random_walk(adjacency, weights_x=citations, min_weight=5, max_dim=10)

COMPLEX_FOLDER_START_NODE = os.path.join(COMPLEX_FOLDER, str(starting_node))

print("The starting node of the random walk has ID {}".format(starting_node))

downsample = subsample_node_x(
    starting_node,
    adjacency_papers,
    adjacency,
    weights_x=citations,
    min_weight=5,
    max_dim=10,
    length_walk=80,
)
print(downsample.shape)

np.save(os.path.join(COMPLEX_FOLDER_START_NODE, "downsampled.npy"), downsample)

# 5. Bipartite to complex

adjacency = sparse.load_npz(biadjacency_matrix_path)

# starting_node = 150250 #Defferard
downsample_papers = np.load(os.path.join(COMPLEX_FOLDER_START_NODE, "downsampled.npy"))

# Bipartite to cochains

simplices, cochains, signals_top = bipart2simpcochain(
    adjacency, citations, indices_x=downsample_papers, dimension=10
)

num_nodes = len(simplices[0])
num_edges = len(simplices[1])
num_triangles = len(simplices[2])
print(f"There are {num_nodes} nodes, {num_edges} edges and {num_triangles} triangles")

# save

cochains_path = os.path.join(COMPLEX_FOLDER_START_NODE, f"cochains.npy")
simplices_path = os.path.join(COMPLEX_FOLDER_START_NODE, f"simplices.npy")

np.save(cochains_path, cochains)
np.save(simplices_path, simplices)

# 6. Complex to Laplacians

# starting_node = 150250
# simplices = np.load(f's2_3_collaboration_complex/{starting_node}_simplices.npy')

boundaries = build_boundaries(simplices)

laplacians = build_laplacians(boundaries)

# save
boundaries_path = os.path.join(COMPLEX_FOLDER_START_NODE, f"boundaries.npy")
laplacians_path = os.path.join(COMPLEX_FOLDER_START_NODE, f"laplacians.npy")

np.save(laplacians_path, laplacians)
np.save(boundaries_path, boundaries)

# 7. Cochains to missing data

# input

# starting_node = 150250
percentage_missing_values = 30

# cochains = np.load(f's2_3_collaboration_complex/{starting_node}_cochains.npy')
# simplices = np.load(f's2_3_collaboration_complex/{starting_node}_simplices.npy')

missing_values = build_missing_values(
    simplices, percentage_missing_values=30, max_dim=7
)

damaged_dataset = build_damaged_dataset(cochains, missing_values, function=np.median)

known_values = build_known_values(missing_values, simplices)

# Save
missing_values_path = os.path.join(
    COMPLEX_FOLDER_START_NODE,
    f"percentage_{percentage_missing_values}_missing_values.npy",
)
damaged_input_path = os.path.join(
    COMPLEX_FOLDER_START_NODE,
    f"percentage_{percentage_missing_values}_input_damaged.npy",
)

known_values_path = os.path.join(
    COMPLEX_FOLDER_START_NODE,
    f"percentage_{percentage_missing_values}_known_values.npy",
)

np.save(missing_values_path, missing_values)
np.save(damaged_input_path, damaged_dataset)
np.save(known_values_path, known_values)
