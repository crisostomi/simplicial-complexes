import networkx as nx
from scipy.sparse import coo_matrix
from mesh_normals.utils.meshes import compute_normal
import numpy as np


def create_simplices(triangles):
    """
    input:
        triangles: list of triangles T_1, ..., T_n where each T_i is a tensor (3, )
                   containing the indices of the nodes that compose the triangle
    return:
        simplices: dict of dicts, simplices[d] is a dict with the d-dimensional simplices as keys and their corresponding ids as value
                   e.g. dict[1] has edges {n_i, n_j} as keys and their corresponding id in the laplacian and boundary matrices as values
    """
    simplices = {0: {}, 1: {}, 2: {}}

    for triangle in triangles:

        nodes = [node.item() for node in triangle]
        node_0, node_1, node_2 = nodes

        # need to wrap the simplex in a frozenset to use it as a dict key
        triangle_simplex = frozenset([node_0, node_1, node_2])

        # assign a progressive id to the triangles
        simplices[2][triangle_simplex] = len(simplices[2])

        # all the edges composing the triangle
        edges = [(node_0, node_1), (node_0, node_2), (node_1, node_2)]

        # give a progressive id to unseen edges
        for edge in edges:
            edge_simplex = frozenset(edge)
            if edge_simplex not in simplices[1]:
                simplices[1][edge_simplex] = len(simplices[1])

        # give a progressive id to unseen nodes
        for node in nodes:
            node_simplex = frozenset({node})
            if node_simplex not in simplices[0]:
                simplices[0][node_simplex] = len(simplices[0])

    return simplices


def create_signals(triangles, positions):
    """
    input:
        triangles: list of triangles T_1, ..., T_n where each T_i is a tensor (3, )
                   containing the indices of the nodes that compose the triangle
    return:
        signals: dict of dicts, signals[0] contains the node positions, signal[2] contains the triangle normals
    """

    node_signals, triangle_signals = {}, {}

    for triangle in triangles:

        nodes = [node.item() for node in triangle]

        # the signal for each node is its position (x, y, z)
        for node in nodes:
            node_signals[frozenset({node})] = positions[node]

        # the signal for each triangle is its normal (n_x, n_y, n_z)
        triangle_normal = compute_normal(positions[nodes])

        triangle_signals[frozenset(nodes)] = triangle_normal

    return node_signals, triangle_signals


def create_graph_from_mesh(positions, triangles):
    # Create graph from mesh

    g = nx.Graph()

    for triangle in triangles:

        nodes = [node.item() for node in triangle]
        node_0, node_1, node_2 = nodes
        edges = [(node_0, node_1), (node_1, node_2), (node_2, node_0)]

        for edge in edges:
            g.add_edge(edge[0], edge[1])

    node_positions_mapping = {node: {'x': pos[0].item(), 'y': pos[1].item(), 'z': pos[2].item()} for node, pos in
                              enumerate(positions)}

    nx.set_node_attributes(g, node_positions_mapping)

    # nx.draw(g)


def build_boundaries(simplices):
    """
    Build the boundary operators from a list of simplices.

    Parameters
    ----------
    simplices:
                List of dictionaries, one per dimension d.
                The size of the dictionary is the number of d-simplices.
                The dictionary's keys are sets (of size d+1) of the vertices that constitute the d-simplices.
                The dictionary's values are the indexes of the simplices in the boundary and Laplacian matrices.
    Returns
    -------
    boundaries:
                List of boundary operators, one per dimension: i-th boundary is in (i-1)-th position
    """
    boundaries = list()

    for dim in range(1, len(simplices)):
        idx_simplices, idx_faces, values = [], [], []

        # simplex is a frozenset of vertices, idx_simplex is the integer progressive id of the simplex
        for simplex, idx_simplex in simplices[dim].items():
            simplices_list_sorted = np.sort(list(simplex))

            for i, left_out in enumerate(simplices_list_sorted):
                # linear combination of the face of the simplex obtained by removing
                # the i-th vertex
                idx_simplices.append(idx_simplex)
                values.append((-1) ** i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[dim - 1][face])

        assert len(values) == (dim + 1) * len(simplices[dim])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                              dtype=np.float32,
                              shape=(len(simplices[dim - 1]), len(simplices[dim])))
        boundaries.append(boundary)
    return boundaries



def build_laplacians(boundaries):
    """Build the Laplacian operators from the boundary operators.

    Parameters
    ----------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension.

    Returns
    -------
    laplacians: list of sparse matrices
       List of Laplacian operators, one per dimension: laplacian of degree i is in the i-th position
    """
    laplacians = list()
    # graph Laplacian L0
    upper = coo_matrix(boundaries[0] @ boundaries[0].T)
    laplacians.append(upper)

    for dim in range(len(boundaries) - 1):
        # lower Laplacian B_{k}^T B_k
        lower = boundaries[dim].T @ boundaries[dim]
        # upper Laplacian B_{k+1} B_{k}^T
        upper = boundaries[dim + 1] @ boundaries[dim + 1].T
        # L_k = L_k_lower + L_k_upper
        laplacians.append(coo_matrix(lower + upper))

    # last Laplacian L_K
    lower = boundaries[-1].T @ boundaries[-1]
    laplacians.append(coo_matrix(lower))
    return laplacians
