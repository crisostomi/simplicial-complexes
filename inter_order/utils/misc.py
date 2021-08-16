import torch
import torch.nn as nn
import numpy as np
import scipy
from tqdm import tqdm


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

# def normalize_normals_vec(v):
#     re

def train(model, num_epochs, components, inputs, targets, Bs, Bt_s, optimizer, device, verbose=False):
    criterion = nn.MSELoss(reduction="mean")

    for epoch in tqdm(range(0, num_epochs)):

        # (max_simplex_dim+1, 1, num_simplices_dim_k)
        xs = [input.clone().to(device) for input in inputs]

        optimizer.zero_grad()
        ys = model(xs, components, Bs, Bt_s)
        # ys = model(xs[0])

        loss = torch.tensor(0., device=device)

        for i in range(3):
            loss += criterion(ys[:, i], targets[:, i])

        if verbose:
            print(f'Epoch: {epoch}, loss: {round(loss.item(), 4)}')

        loss.backward()
        optimizer.step()


def compute_per_coord_diff(preds, targets):
    """
    :param preds: (num_triangles, 3)
    :param targets: (num_triangles, 3)
    :return:
    """

    diffs = torch.mean(torch.abs(preds - targets), axis=1)

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