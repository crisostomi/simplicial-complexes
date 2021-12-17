import scipy.sparse
import torch
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn.functional as F
from torch_sparse import cat as sparse_cat
from timeit import default_timer as timer
import numpy as np


def block_diagonal(*arrs):
    bad_args = [
        k
        for k in range(len(arrs))
        if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)
    ]

    if bad_args:
        raise ValueError(
            "arguments in the following positions must be 2-dimension tensor: %s"
            % bad_args
        )

    shapes = torch.tensor([a.shape for a in arrs])
    i = []
    v = []
    r, c = 0, 0
    for k, (rr, cc) in enumerate(shapes):
        first_index = torch.arange(r, r + rr, device=arrs[0].device)
        second_index = torch.arange(c, c + cc, device=arrs[0].device)
        index = torch.stack(
            (
                first_index.tile((cc, 1)).transpose(0, 1).flatten(),
                second_index.repeat(rr),
            ),
            dim=0,
        )
        i += [index]
        v += [arrs[k].to_dense().flatten()]
        r += rr
        c += cc
    out_shape = torch.sum(shapes, dim=0).tolist()

    if arrs[0].device == "cpu":
        out = torch.sparse.DoubleTensor(torch.cat(i, dim=1), torch.cat(v), out_shape)
    else:
        out = torch.cuda.sparse.DoubleTensor(
            torch.cat(i, dim=1).to(arrs[0].device), torch.cat(v), out_shape
        )
    return out


# def sparse_slice(tensor, start, end):
#     assert tensor.is_sparse
#     indices, values = tensor._indices(), tensor._values()
#
#     row_start, col_start = start
#     row_end, col_end = end
#     rows, cols = indices[0], indices[1]
#
#     row_mask = (rows >= row_start) & (rows < row_end)
#     col_mask = (cols >= col_start) & (cols < col_end)
#
#     rows = rows[row_mask] - row_start
#     cols = cols[col_mask] - col_start
#
#     row_col_mask = row_mask & col_mask
#
#     new_values = values[row_col_mask]
#     new_indices = torch.stack((rows, cols))
#
#     sparse_result = torch.sparse_coo_tensor(
#         indices=new_indices,
#         values=new_values,
#         size=(row_end - row_start, col_end - col_start),
#     )
#
#     return sparse_result


# def sparse_slice(tensor, start, end):
#     assert tensor.is_sparse
#     indices, values = tensor._indices(), tensor._values()
#
#     rows, cols = indices[0], indices[1]
#     assert is_sorted(rows)
#     assert is_sorted(cols)
#
#     row_start, col_start = start
#     row_end, col_end = end
#
#     row_lb, row_ub = torch.searchsorted(rows, torch.tensor([row_start, row_end]))
#     col_lb, col_ub = torch.searchsorted(cols, torch.tensor([col_start, col_end]))
#
#     row_col_lb = max(row_lb, col_lb)
#     row_col_ub = min(row_ub, col_ub)
#
#     rows_slice = rows[row_col_lb:row_col_ub] - row_start
#     cols_slice = cols[row_col_lb:row_col_ub] - col_start
#     values_slice = values[row_col_lb:row_col_ub]
#
#     indices_slice = torch.stack((rows_slice, cols_slice))
#
#     sparse_result = torch.sparse_coo_tensor(
#         indices=indices_slice,
#         values=values_slice,
#         size=(row_end - row_start, col_end - col_start),
#     )
#
#     return sparse_result


def sparse_slice(sparse_mat, start, end):

    row_start, col_start = start
    row_end, col_end = end

    sparse_matrix = sparse_mat[row_start:row_end, col_start:col_end]
    sparse_matrix_coo = sparse_matrix.tocoo()
    indices = np.stack((sparse_matrix_coo.row, sparse_matrix_coo.col))

    sparse_result = torch.sparse_coo_tensor(
        indices=indices,
        values=sparse_matrix_coo.data,
        size=sparse_matrix_coo.get_shape(),
    )

    return sparse_result


def is_sorted(arr):
    return np.all(np.diff(arr) >= 0)


def get_pooling_fn(readout):
    if readout == "sum":
        return global_add_pool
    elif readout == "mean":
        return global_mean_pool
    else:
        raise NotImplementedError(
            "Readout {} is not currently supported.".format(readout)
        )


def get_nonlinearity(nonlinearity, return_module=True):
    if nonlinearity == "relu":
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == "elu":
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == "id":
        module = torch.nn.Identity
        function = lambda x: x
    elif nonlinearity == "sigmoid":
        module = torch.nn.Sigmoid
        function = F.sigmoid
    elif nonlinearity == "tanh":
        module = torch.nn.Tanh
        function = torch.tanh
    else:
        raise NotImplementedError(
            "Nonlinearity {} is not currently supported.".format(nonlinearity)
        )
    if return_module:
        return module
    return function


def sparse_flatten(tensor):
    tensor = tensor.coalesce()

    indices, values = tensor.indices(), tensor.values()
    rows, cols = indices

    num_cols = tensor.shape[1]

    cols = rows * num_cols + cols

    new_shape = (tensor.shape[0] * tensor.shape[1],)
    return torch.sparse_coo_tensor(
        indices=cols.unsqueeze(0), values=values, size=new_shape
    )
