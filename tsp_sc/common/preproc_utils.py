import itertools

from tqdm import tqdm
from tsp_sc.common.simp_complex import Cochain, SimplicialComplex
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from tsp_sc.common.simplices import build_boundaries
import torch
import gudhi as gd


def convert_graph_dataset_with_gudhi(
    dataset, expansion_dim: int, init_method: str = "sum"
):
    """
    :param dataset:
    :param expansion_dim:
    :param init_method:
    :return:
    """
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
    print(f"Computed a total of {len(complexes)} complexes")

    return complexes


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
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, Tensor)

    # Creates the gudhi-based simplicial complex
    simplex_tree = pyg_to_simplex_tree(edge_index, size)

    # compute the clique complex up to the desired dim
    simplex_tree.expansion(expansion_dim)

    # see what is the dimension of the complex now
    complex_dim = simplex_tree.dimension()

    # builds tables of the simplicial complexes at each level and their IDs
    simplex_tables, simplex_id_maps = build_tables(simplex_tree)

    boundaries = [None] + build_boundaries(simplex_id_maps)

    # construct features for the higher dimensions
    xs = construct_features(x, simplex_tables, init_method)

    # initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []

    for dim in range(complex_dim + 1):
        y = v_y if dim == 0 else None

        num_simplices = len(xs[dim])
        num_simplices_up = len(xs[dim + 1]) if dim < complex_dim else 0
        num_simplices_down = len(xs[dim - 1]) if dim > 0 else 0

        cochain = Cochain(
            dim=dim,
            signal=xs[dim],
            boundary=boundaries[dim + 1] if dim < complex_dim else None,
            coboundary=boundaries[dim].T if dim > 0 else None,
            complex_dim=complex_dim,
            y=y,
            num_simplices=num_simplices,
            num_simplices_up=num_simplices_up,
            num_simplices_down=num_simplices_down,
        )
        cochains.append(cochain)

    return SimplicialComplex(*cochains, y=complex_y, dimension=complex_dim)


def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """
    Constructs a simplex tree from a PyG graph.

    :param edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
    :param size: The number of nodes in the graph.
    """
    st = gd.SimplexTree()

    # add vertices to the simplex
    for v in range(size):
        st.insert([v])

    # add edges to the simplex
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def build_tables(simplex_tree):
    """
    :return: simplex_tables: list of lists, each dimension contains the simplices of that dimension
                [
                    [ simp_0_dim_0, ... , simp_n0_dim_0] ,
                    [ simp_0_dim_1, ... , simp_n1_dim_1] ,
                    ...
                ]
             id_maps: list of dictionaries, each dimension d contains for each d-simplex its id
                [
                    { simp_0_dim_0: 0, ... , simp_n0_dim_0: n0 },
                    { simp_0_dim_1: 0, ... , simp_n1_dim_1: n1 },
                    ...
                ]
    """
    complex_dim = simplex_tree.dimension()

    # each of these data structures has a separate entry per dimension
    id_maps = [{} for _ in range(complex_dim + 1)]  # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim + 1)]  # matrix of simplices

    for simplex, _ in simplex_tree.get_simplices():

        dim = len(simplex) - 1

        # assign this simplex the next unused ID
        next_id = len(simplex_tables[dim])
        simplex_tables[dim].append(simplex)
        id_maps[dim][frozenset(simplex)] = next_id

    return simplex_tables, id_maps


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def construct_features(
    vertex_features: Tensor, simplex_tables, init_method: str
) -> List:
    """
    Combines the features of the component vertices to initialise the simplex features.
    For example, for dimension 2 a simplex {v0, v1, v2} will have the result of the aggregation
    of the features of the vertices v0, v1 and v2.
    """

    features = [vertex_features]
    num_dims = len(simplex_tables)

    for dim in range(1, num_dims):
        simplices_ids = []
        simplices_vertices_flat = []

        for i, simplex in enumerate(simplex_tables[dim]):
            simplices_ids += [i for _ in range(len(simplex))]
            simplices_vertices_flat += simplex

        node_cell_index = torch.LongTensor([simplices_vertices_flat, simplices_ids])
        in_features = vertex_features.index_select(0, node_cell_index[0])

        dim_features = scatter(
            src=in_features,
            index=node_cell_index[1],
            dim=0,
            dim_size=len(simplex_tables[dim]),
            reduce=init_method,
        )

        features.append(dim_features)

    return features


def extract_labels(y, size):
    """
    :param y: graph-wise label or node-level labels
    :param size: number of nodes in the graph
    :return:
    """
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
