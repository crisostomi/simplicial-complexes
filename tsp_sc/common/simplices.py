import networkx as nx
from scipy.sparse import coo_matrix
import numpy as np
import torch
import scipy
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import itertools
import gudhi


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

    node_positions_mapping = {
        node: {"x": pos[0].item(), "y": pos[1].item(), "z": pos[2].item()}
        for node, pos in enumerate(positions)
    }

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
        boundary = coo_matrix(
            (values, (idx_faces, idx_simplices)),
            dtype=np.float32,
            shape=(len(simplices[dim - 1]), len(simplices[dim])),
        )
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


def create_node_triangle_adj_matrix(nodes, triangles):
    """
        nodes: tensor (num_nodes, 3)
        triangles: tensor (num_triangles, 3)
    """

    i = np.arange(triangles.shape[0])
    i = np.concatenate((i, i, i), 0)

    j = triangles.reshape(-1)

    adj = csr_matrix(
        (np.ones_like(j), (i, j)), shape=(triangles.shape[0], nodes.shape[0])
    )  # FxV sparse matrix

    return torch.tensor(adj.todense())


def normalize_laplacian(L, eigenval, half_interval=False):
    """
        Returns the laplacian normalized by the largest eigenvalue
    """
    assert scipy.sparse.isspmatrix(L)

    # L is squared
    M = L.shape[0]
    assert M == L.shape[1]

    # take the first eigenvalue of the Laplacian, i.e. the largest
    # largest_eigenvalue = linalg.eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]

    L_normalized = L.copy()

    if half_interval:
        L_normalized *= 1.0 / eigenval
    else:
        L_normalized *= 2.0 / eigenval
        L_normalized.setdiag(L_normalized.diagonal(0) - np.ones(M), 0)

    return L_normalized


def get_empty_triangles(simplices):
    vertices = simplices[0].keys()
    edges = simplices[1].keys()
    triangles = simplices[2].keys()

    num_empty_triangles = 0
    empty_triangles = set()

    for i in vertices:
        for j in vertices:
            for k in vertices:
                ij = i.union(j)
                jk = j.union(k)
                ik = i.union(k)
                ijk = ij.union(k)
                if ij in edges and jk in edges and ik in edges and ijk not in triangles:
                    ijk = frozenset(ijk)
                    if ijk not in empty_triangles:
                        num_empty_triangles += 1
                        empty_triangles.add(ijk)

    return empty_triangles


def get_largest_eigenvalue(M):
    """
        Given a matrix M, returns its largest eigenvalue
    """
    assert sparse.isspmatrix_coo(M)

    return linalg.eigsh(M, k=1, which="LM", return_eigenvectors=False)[0]


def compute_normal(triangle):
    assert triangle.shape[0] == 3
    v1, v2, v3 = triangle

    A = v2 - v1
    B = v3 - v1

    N = np.zeros(3)

    N[0] = A[1] * B[2] - A[2] * B[1]
    N[1] = A[2] * B[0] - A[0] * B[2]
    N[2] = A[0] * B[1] - A[1] * B[0]

    norm = np.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)

    return N / norm


def list_triangles(G):
    triangles = []
    # sort vertices by increasing degree
    sorted_vertices = sorted(G.nodes(), key=lambda v: G.degree(v))
    visited = set()
    for u in sorted_vertices:

        visited.add(u)
        neighbors_higher_deg = {
            neighbor for neighbor in G.neighbors(u) if G.degree(neighbor) >= G.degree(u)
        } - visited
        possible_edges = list(itertools.combinations(neighbors_higher_deg, 2))

        for possible_edge in possible_edges:
            if possible_edge in G.edges():
                v, w = possible_edge
                triangles.append({u, v, w})
    return triangles


def list_triangles_bruteforce(G):
    triangles = []
    for u in G.nodes:
        for v in G.neighbors(u):
            for w in G.neighbors(v):
                if w != u and (u, w) in G.edges():
                    if {u, v, w} not in triangles:
                        triangles.append({u, v, w})
    return triangles


def count_triangles(graphs):
    num_triangles = 0

    for g in graphs:
        num_triangles += len(list_triangles(g))

    avg_num_triangles = num_triangles / len(graphs)
    return avg_num_triangles


def build_simplex_from_graph(G):
    simplex_tree = gudhi.SimplexTree()

    triangles = list_triangles(G)

    for triangle in triangles:
        simplex_tree.insert(triangle)

    for edge in G.edges():
        simplex_tree.insert(edge)

    for node in G.nodes():
        simplex_tree.insert({node})

    return simplex_tree


def extract_simplices(simplex_tree, max_dim):
    """Create a list of simplices from a gudhi simplex tree."""
    simplices = [dict() for _ in range(max_dim + 1)]
    for simplex, _ in simplex_tree.get_skeleton(max_dim):
        k = len(simplex)
        simplices[k - 1][frozenset(simplex)] = len(simplices[k - 1])

    return simplices


def get_index_from_boundary(boundary):
    return np.argwhere(boundary != 0)


def get_orientation_from_boundary(boundary: coo_matrix) -> np.ndarray:
    """
    :param boundary: sparse matrix containing as indices the ids of incident simplices
                     and as values +1 or -1
    :return: array containing +1 or -1 for each incidence
    """
    return boundary.data
