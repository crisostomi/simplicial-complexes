import torch


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


def sparse_slice(tensor, start, end):
    assert tensor.is_sparse
    tensor = tensor.coalesce()
    indices, values = tensor.indices(), tensor.values()

    row_start, col_start = start
    row_end, col_end = end
    rows, cols = indices[0], indices[1]

    row_mask = (rows >= row_start) & (rows < row_end)
    col_mask = (cols >= col_start) & (cols < col_end)

    rows = rows[row_mask] - row_start
    cols = cols[col_mask] - col_start

    row_col_mask = row_mask & col_mask

    new_values = values[row_col_mask]
    new_indices = torch.stack((rows, cols))

    sparse_result = torch.sparse_coo_tensor(
        indices=new_indices,
        values=new_values,
        size=(row_end - row_start, col_end - col_start),
    )

    return sparse_result
