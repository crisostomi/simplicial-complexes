import torch
import torch.nn as nn
import numpy as np
import scipy
from tqdm import tqdm
from enum import Enum
import os
from inter_order.utils.io import load_dict


def print_torch_version():
    TORCH_VERSION = torch.__version__[:5]
    CUDA_VERSION = torch.version.cuda.replace(".", "")
    print(f"Running torch version {TORCH_VERSION}, with CUDA version {CUDA_VERSION}")


def load_data(to_load, load_folder):
    loaded_data = {}

    for elem_name, elem_type in to_load.items():
        path = os.path.join(load_folder, elem_name)

        if elem_type == dict:
            loaded_data[elem_name] = load_dict(path)
        else:
            loaded_data[elem_name] = np.load(path + ".npy", allow_pickle=True)
    return loaded_data


def print_number_of_parameters(network):
    num_params = 0
    for param in network.parameters():
        p = np.array(param.shape, dtype=int).prod()
        num_params += p
    print("Total number of parameters: %d" % (num_params))


def coo2tensor(A):
    """
        Converts a sparse matrix in COOrdinate format to torch tensor.
    """
    assert scipy.sparse.isspmatrix_coo(A)
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size=A.shape, requires_grad=False)


def compute_per_coord_diff(preds, targets):
    """
    :param preds: (num_triangles, 3)
    :param targets: (num_triangles, 3)
    :return:
    """

    diffs = torch.mean(torch.abs(preds - targets), axis=0)

    return diffs


def compute_angle_diff(preds, targets):
    """
    :param preds: (num_triangles, 3)
    :param targets: (num_triangles, 3)
    :return:
    """
    angle_matrix = preds @ targets.T
    angles = torch.diag(angle_matrix)

    arcs = torch.arccos(angles) / np.pi * 180

    return torch.mean(arcs)


class Phases(Enum):
    train = "training"
    val = "validation"
    test = "test"
