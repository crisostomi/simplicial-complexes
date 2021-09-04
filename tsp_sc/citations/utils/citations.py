import numpy as np
from tqdm import tqdm
import gzip
import glob
import json
import pickle
import pandas as pd
import os
import scipy.sparse.linalg
from scipy import sparse
import networkx as nx
import sys
import math
import random
import typing
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.linalg import eig
from scipy.linalg import null_space
from networkx.algorithms import bipartite as nxb
import gudhi


def count(papers, edges):
    print(f"  {len(papers):,} papers")
    print("  {:,} authors".format(edges["author"].nunique()))
    print(f"  {len(edges):,} paper-author links")


def count_papers_authors(papers, edges):
    print(f"  {len(papers):,} papers")
    print("  {:,} authors".format(edges["author"].nunique()))
    print(f"  {len(edges):,} paper-author links")


def bipart2simpcochain(
    bipartite, weights_x, indices_x=None, function=np.sum, dimension=3
):
    """From a collaboration bipartite graph X-Y and its weights on X to a
    collaboration simplicial complex and its collaboration values on the
    simplices.

    Parameters
    ----------
    bipartite : scipy sparse matrix
        Sparse matrix representing the collaboration bipartite graph X-Y.
    weights_x : ndarray
        Array of size bipartite.shape[0], containing the weights on the node of X
    function : callable
        Functions that will aggregate the features to build the k-cochains, default=np.sum
    indices_x : array
        Array of the indices of the X nodes to restrict to, default = all nodes of X
    dimension : int
        Maximal dimension of the simplicial complex.

    Returns
    -------
    simplices: list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The dictionary's values are their indices
    cochains:list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The dictionary's values are the k-cochains
    signals_top:
        Features for every maximal dimensional simplex
    """
    simplex_tree, signal_top = bipart2simplex(
        bipartite, weights_x, indices_x, dimension
    )
    simplices = extract_simplices(simplex_tree)
    cochains = build_cochains(simplex_tree, signal_top, function, dimension)
    return simplices, cochains, signal_top


def bipart2simplex(bipartite, weights_x, indices_x=None, dimension=3):
    """Build a Gudhi Simplex Tree from the bipartite graph X-Y by projection on Y and extract the
    features corresponding to maximal dimensional simplices.
    Parameters
    ----------
    bipartite : scipy sparse matrix
        bipartite collaboration graph X-Y
    weights_x : ndarray
        Array of size bipartite.shape[0], containing the weights on the node of X
    indices_x = array
        Array of the indices of the X nodes to restrict to, default: all nodes of X
    dimension: int
        maximal dimension of the simplicial complex = maximal number of individuals collaborating.
    Returns
    -------
    simplex_tree:
        Gudhi simplex tree.
    signals_top:
        Features for every maximal dimensional simplex.
    """
    signals_top = [dict() for _ in range(dimension + 1)]
    simplex_tree = gudhi.SimplexTree()

    if np.all(indices_x) == None:
        indices_x = np.arange(bipartite.shape[0])

    Al = bipartite.tolil()

    # for each paper, if the number of authors is <= the max accepted dimension then we insert it into the simplex complex
    for j, authors in enumerate(Al.rows[indices_x]):
        if len(authors) <= dimension + 1:
            k = len(authors)
            simplex_tree.insert(authors)
            # append the number of citations of the paper to the simplex
            # (which may already exist as it is indexed by the group of authors)
            signals_top[k - 1].setdefault(frozenset(authors), []).append(
                weights_x[indices_x][j]
            )
        else:
            continue

    return (simplex_tree, signals_top)


def extract_simplices(simplex_tree):
    """Create a list of simplices from a gudhi simplex tree."""
    simplices = [dict() for _ in range(simplex_tree.dimension() + 1)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        k = len(simplex)
        simplices[k - 1][frozenset(simplex)] = len(simplices[k - 1])
    return simplices


def build_cochains(simplex_tree, signals_top, function=np.sum, dimension=3):
    """Build the k-cochains using the weights on X (from the X-Y bipartite graph)
     and a chosen aggregating function. Features are aggregated by the provided functions.
     The function takes as input a list of values and must return a single number.

    Parameters
    ----------
    simplex_tree :
        Gudhi simplex tree
    signals_top : ndarray
        Features for every maximal dimensional simplex = weights on the nodes of X (from bipartite graph X-Y)
    function : callable
        Functions that will aggregate the features to build the k-coachains, default=np.sum

    Returns
    -------
    signals : list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The dictionary's values are the k-cochains
    """
    assert len(signals_top) == dimension + 1
    cochains = [dict() for _ in range(simplex_tree.dimension() + 1)]

    for d in range(dimension, -1, -1):
        # simplex represents a group of authors, values are the number of citations they got in their papers
        # len(values) = # papers published together by the group of authors
        for simplex, values in signals_top[d].items():
            st = gudhi.SimplexTree()
            st.insert(simplex)
            for face, _ in st.get_skeleton(st.dimension()):
                # for each "sub-simplex" s contained in the simplex (e.g. if d=2 we have the simplex itself, the edges and the vertices)
                face = frozenset(face)
                # if a simplex has k+1 nodes then it is k-dimensional
                face_dim = len(face) - 1
                cochains[face_dim].setdefault(face, []).extend(signals_top[d][simplex])

    # each simplex now has a list containing the citations of all the simplices it is contained into

    for d in range(dimension, -1, -1):
        for simplex, values in signals_top[d].items():
            st = gudhi.SimplexTree()
            st.insert(simplex)
            for face, _ in st.get_skeleton(st.dimension()):
                face = frozenset(face)
                face_dim = len(face) - 1
                value = np.array(cochains[face_dim][face])
                cochains[face_dim][face] = int(function(value))

    # each simplex has a single value which is the aggregation of the signal on all the simplices it is contained into
    return cochains


def build_missing_values(simplices, percentage_missing_values, max_dim=10):
    """
    The functions randomly deletes a given percentage of the values of simplices in each dimension
    of a simplicial complex.

    Parameters
    ----------

    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the vertices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    percentage_missing_values: integer
        Percentage of values missing

    max_dim: integer
        maximal dimension of the simplices to be considered.

    Returns
    ----------
        missing_values: list of dictionaries
        List of dictionaries, one per dimension d. The dictionary's keys are the missing d-simplices.
        The dictionary's values are the indexes of the simplices in the boundary and Laplacian matrices.

    """
    missing_values = [dict() for _ in range(max_dim + 1)]

    for k in range(max_dim + 1):

        # randomly select percentage_missing_values % of the simplices of each dimension
        simplices_dim_k = list(simplices[k].keys())
        threshold_index = int(
            np.ceil((len(simplices_dim_k) / 100) * percentage_missing_values)
        )
        simplices_dim_k_copy = np.copy(simplices_dim_k)
        random.shuffle(simplices_dim_k_copy)
        lost_simplices = simplices_dim_k_copy[:threshold_index]

        # assign to each simplex in the missing simplices the same id it had before
        for simplex in lost_simplices:
            missing_values[k][simplex] = simplices[k][simplex]

    return missing_values


def build_damaged_dataset(cochains, missing_values, function=np.median):
    """
    The function replaces the missing values in the dataset with a value inferred
    from the known data (eg the missing values are replaced by the median
    or mean of the known values).

    Parameters
    ----------
    cochains: list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The dictionary's values are the k-cochains

    missing_values: list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the missing k-simplices. The dictionary's values are their indices

    function: callable
        missing values are replaced by the function of the known values, defaut median

    Returns
    ----------
    damaged_dataset: list of dictionaries

        List of dictionaries, one per dimension d. The dictionary's keys are the d-simplices.
        The dictionary's values are the d-cochains where the damaged portion has been replaced
        by the given function value.

    """

    # obtain median value
    max_dim = len(missing_values)
    cochains_copy = np.copy(cochains)
    signal = []
    for k in range(max_dim):
        cochains_dim_k = list(cochains_copy[k].values())
        signal.append(cochains_dim_k)
    signal = np.array(signal)

    # medians[k] has the median number of citations for dimension k
    medians = []
    for k in range(max_dim):
        signals_dim_k = [signal[k][j] for j in range(len(signal[k]))]
        median_dim_k = function(signals_dim_k)
        medians.append(median_dim_k)

    # create input using median value for unknown values
    damaged_dataset = np.copy(cochains)

    for k in range(max_dim):
        lost_simplices = list(missing_values[k].keys())
        for simplex in lost_simplices:
            damaged_dataset[k][simplex] = medians[k]
    return damaged_dataset


def build_known_values(missing_values, simplices):
    """
    The functions return the not missing simplices and indices in each dimension

    Parameters
    ----------
    missing_values: list of dictionaries
        List of dictionaries, one per dimension d. The dictionary's keys are the missing d-simplices.
        The dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are the sets (of size d
        + 1) of vertices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.


    Returns
    ----------
    known_values: list of dictionaries
        List of dictionaries, one per dimension d. The dictionary's keys are not missing d-simplices.
        The dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    """
    max_dim = len(missing_values)
    known_values = [dict() for _ in range(max_dim + 1)]

    for k in range(max_dim):
        # take only the simplices which have not been removed
        known_simplices = list(set(simplices[k].keys()) - set(missing_values[k].keys()))

        for index, simplex in enumerate(known_simplices):
            known_values[k][simplex] = simplices[k][simplex]

    return known_values


def get_paths(path_params, data_params):
    (missing_value_ratio, starting_node) = (
        data_params["missing_value_ratio"],
        data_params["starting_node"],
    )

    paths = {k: v for k, v in path_params.items()}
    starting_node_const = "STARTINGNODE"
    missing_value_ratio_const = "MISSINGVALUERATIO"
    for path_name, path_value in path_params.items():
        if starting_node_const in path_value:
            paths[path_name] = path_value.replace(
                starting_node_const, str(starting_node)
            ).replace(missing_value_ratio_const, str(missing_value_ratio))
    return paths
