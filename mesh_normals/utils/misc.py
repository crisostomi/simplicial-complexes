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

def normalize_vector(v):
    return v / torch.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def train(model, num_epochs, components, inputs, targets, Bs, Bts, optimizer, device, verbose=False):

    for epoch in tqdm(range(0, num_epochs)):

        # (max_simplex_dim+1, 1, num_simplices_dim_k)
        xs = [input.clone().to(device) for input in inputs]

        max_simplex_dim = len(xs) - 1

        optimizer.zero_grad()

        # ys = model(xs, components, Bs, Bt_s)
        ys = model(xs[0])

        targets = targets.to(torch.float64)

        loss = torch.tensor(0., device=device, dtype=torch.float64)

        for i in range(3):
            loss += criterion(ys[:, i], targets[:, i])

        if verbose:
            print(f'Epoch: {epoch}, loss: {round(loss.item(), 4)}')

        loss.backward()
        optimizer.step()