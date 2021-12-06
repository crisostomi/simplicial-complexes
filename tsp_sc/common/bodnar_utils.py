import numpy as np
import torch
import gudhi as gd
import itertools
import os

# import graph_tool as gt
# import graph_tool.topology as top
import networkx as nx

from tqdm import tqdm
from tsp_sc.common.simp_complex import Cochain, SimplicialComplex
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from tsp_sc.common.simplices import build_boundaries


def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """Constructs a simplex tree from a PyG graph.

    Args:
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph.
    """
    st = gd.SimplexTree()
    # Add vertices to the simplex.
    for v in range(size):
        st.insert([v])

    # Add the edges to the simplex.
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, num_nodes):
    complex_dim = simplex_tree.dimension()
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(complex_dim + 1)]  # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim + 1)]  # matrix of simplices
    boundaries_tables = [[] for _ in range(complex_dim + 1)]

    simplex_tables[0] = [[v] for v in range(num_nodes)]
    id_maps[0] = {frozenset([v]): v for v in range(num_nodes)}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        # Assign this simplex the next unused ID
        next_id = len(simplex_tables[dim])
        id_maps[dim][frozenset(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(
    simplex_tree, id_maps, complex_dim: int
):
    """Build two maps simplex -> its coboundaries and simplex -> its boundaries"""
    # The extra dimension is added just for convenience to avoid treating it as a special case.
    boundaries = [{} for _ in range(complex_dim + 2)]  # simplex -> boundaries
    coboundaries = [{} for _ in range(complex_dim + 2)]  # simplex -> coboundaries
    boundaries_tables = [[] for _ in range(complex_dim + 1)]

    for simplex, _ in simplex_tree.get_simplices():
        # Extract the relevant boundary and coboundary maps
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        # Add the boundaries of the simplex to the boundaries table
        if simplex_dim > 0:
            boundaries_ids = [
                id_maps[simplex_dim - 1][frozenset(boundary)]
                for boundary in get_simplex_boundaries(simplex)
            ]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries


def construct_features(vx: Tensor, cell_tables, init_method: str) -> List:
    """Combines the features of the component vertices to initialise the cell features"""
    features = [vx]
    for dim in range(1, len(cell_tables)):
        aux_1 = []
        aux_0 = []
        for c, cell in enumerate(cell_tables[dim]):
            aux_1 += [c for _ in range(len(cell))]
            aux_0 += cell
        node_cell_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_cell_index[0])
        features.append(
            scatter(
                in_features,
                node_cell_index[1],
                dim=0,
                dim_size=len(cell_tables[dim]),
                reduce=init_method,
            )
        )

    return features


def extract_labels(y, size):
    v_y, complex_y = None, None
    if y is None:
        return v_y, complex_y

    y_shape = list(y.size())

    if y_shape[0] == 1:
        # This is a label for the whole graph (for graph classification).
        # We will use it for the complex.
        complex_y = y
    else:
        # This is a label for the vertices of the complex.
        assert y_shape[0] == size
        v_y = y

    return v_y, complex_y


def compute_clique_complex_with_gudhi(
    x: Tensor,
    edge_index: Adj,
    size: int,
    expansion_dim: int = 2,
    y: Tensor = None,
    init_method: str = "sum",
) -> SimplicialComplex:
    """Generates a clique complex of a pyG graph via gudhi.

    Args:
        x: The feature matrix for the nodes of the graph
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph
        expansion_dim: The dimension to expand the simplex to.
        y: Labels for the graph nodes or a label for the whole graph.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, Tensor)  # Support only tensor edge_index for now

    # Creates the gudhi-based simplicial complex
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    simplex_tree.expansion(
        expansion_dim
    )  # Computes the clique complex up to the desired dim.
    complex_dim = (
        simplex_tree.dimension()
    )  # See what is the dimension of the complex now.

    # Builds tables of the simplicial complexes at each level and their IDs
    simplex_tables, id_maps = build_tables(simplex_tree, size)

    boundaries = [None] + build_boundaries(id_maps)

    # Construct features for the higher dimensions
    xs = construct_features(x, simplex_tables, init_method)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []

    for i in range(complex_dim + 1):
        y = v_y if i == 0 else None

        num_simplices = len(xs[i])
        num_simplices_up = len(xs[i + 1]) if i < complex_dim else 0
        num_simplices_down = len(xs[i - 1]) if i > 0 else 0

        cochain = Cochain(
            dim=i,
            signal=xs[i],
            boundary=boundaries[i + 1] if i < complex_dim else None,
            coboundary=boundaries[i].T if i > 0 else None,
            complex_dim=complex_dim,
            y=y,
            num_simplices=num_simplices,
            num_simplices_up=num_simplices_up,
            num_simplices_down=num_simplices_down,
        )
        cochains.append(cochain)

    return SimplicialComplex(*cochains, y=complex_y, dimension=complex_dim)


def convert_graph_dataset_with_gudhi(
    dataset, expansion_dim: int, init_method: str = "sum"
):
    # TODO(Cris): Add parallelism to this code like in the cell complex conversion code.
    dimension = -1
    complexes = []
    num_features = [None for _ in range(expansion_dim + 1)]

    for data in tqdm(dataset):
        complex = compute_clique_complex_with_gudhi(
            data.x,
            data.edge_index,
            data.num_nodes,
            expansion_dim=expansion_dim,
            y=data.y,
            init_method=init_method,
        )
        if complex.dimension > dimension:
            dimension = complex.dimension

        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features
        complexes.append(complex)

    assert len(complexes) == len(dataset)

    return complexes, dimension, num_features[: dimension + 1]
