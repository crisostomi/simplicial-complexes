import sys
import os
import math
from tqdm.notebook import tqdm
import random
import typing

# DL
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import coo_matrix
from scipy.linalg import eig
from scipy.linalg import null_space

# plots
from matplotlib import pyplot as plt

# reproducibility
RANDOM_SEED = 1337

# seed_everything(RANDOM_SEED)
# torch.backends.cudnn.benchmark = False
# torch.set_deterministic(True)

starting_node = "original"  # Original starting node used by Defferard (150250)
percentage_missing_values = 30

max_simplex_dim = 10
considered_simplex_dim = 2
assert considered_simplex_dim <= max_simplex_dim

consider_last_dim_upper_adj = (
    True if considered_simplex_dim < max_simplex_dim else False
)

device = "cuda" if torch.cuda.is_available else "cpu"

# paths
PROJECT_FOLDER = os.path.join(
    GDRIVE_HOME, "MyDrive/Topological Signal Processing/citations"
)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
OUT_FOLDER = os.path.join(PROJECT_FOLDER, "output")
COMPLEX_FOLDER = os.path.join(DATA_FOLDER, "collaboration_complex")
COMPLEX_PATH = os.path.join(COMPLEX_FOLDER, str(starting_node))

LAPLACIANS_PATH = os.path.join(COMPLEX_PATH, "laplacians.npy")
BOUNDARIES_PATH = os.path.join(COMPLEX_PATH, "boundaries.npy")

# Laplacians loading
laplacians = np.load(LAPLACIANS_PATH, allow_pickle=True)
laplacians = laplacians[: considered_simplex_dim + 1]

num_simplices = [L.shape[0] for L in laplacians]
print(num_simplices)

# Boundaries loading
boundaries = np.load(BOUNDARIES_PATH, allow_pickle=True)

Bs = [boundaries[i] for i in range(considered_simplex_dim + 1)]
Bs = [None] + Bs

# Orthogonal decomposition

components = {"full": laplacians, "sol": [], "irr": [], "har": []}

# Harmonic component

U_har = [null_space(L.todense()) for L in components["full"]]
for i, ker in enumerate(U_har):
    print(f"L{i} has kernel dimension {ker.shape[1]}")

components["irr"] = [None for i in range(considered_simplex_dim + 1)]

for k in range(considered_simplex_dim + 1):
    Btk_upper = Bs[k + 1].transpose().todense()
    Bk_upper = Bs[k + 1].todense()

    BBt = Bk_upper @ Btk_upper

    components["irr"][k] = coo_matrix(BBt)

if not consider_last_dim_upper_adj:
    components["irr"][considered_simplex_dim] = None

# Solenoidal component
components["sol"] = [None for i in range(considered_simplex_dim + 1)]

for k in range(1, considered_simplex_dim + 1):
    Btk = Bs[k].transpose().todense()
    Bk = Bs[k].todense()

    BtB = Btk @ Bk

    components["sol"][k] = coo_matrix(BtB)

# Normalize the components

for i in range(0, considered_simplex_dim + 1):
    for comp in ["sol", "irr", "full"]:
        if components[comp][i] != None:
            largest_eigenvalue = get_largest_eigenvalue(components[comp][i])
            normalized = normalize_laplacian(
                components[comp][i], largest_eigenvalue, half_interval=True
            )
            components[comp][i] = coo2tensor(normalized).cuda()

# Check
# the first irrotational component must be equal to the node Laplacian

tol = 1e-6
num_nodes = len(components["irr"][0])

comparison = (
    np.abs(
        components["irr"][0].cpu().to_dense() - components["full"][0].cpu().to_dense()
    )
    <= tol
)
assert comparison.all()

# the sum of the first solenoidal component and the first irrotational component must be equal to the edge Laplacian.  This does not happen as both solenoidal and irrotational components are normalized separately
print(components["full"][1].to_dense())
print(components["full"][1].shape)
res = components["irr"][1] + components["sol"][1]

print(res.to_dense())
print(res.shape)

# the sum of the last solenoidal component and the last irrotational component must be equal to the triangle Laplacian.  Again, this does not happen as both solenoidal and irrotational components are normalized separately
print(components["full"][considered_simplex_dim].to_dense())
print(components["full"][considered_simplex_dim].shape)
res = (
    components["irr"][considered_simplex_dim]
    + components["sol"][considered_simplex_dim]
)

print(res.to_dense())
print(res.shape)

# Data loading

simplices = np.load(os.path.join(COMPLEX_PATH, "simplices.npy"), allow_pickle=True)

known_simplices = np.load(
    os.path.join(
        COMPLEX_PATH, f"percentage_{percentage_missing_values}_known_values.npy"
    ),
    allow_pickle=True,
)
missing_simplices = np.load(
    os.path.join(
        COMPLEX_PATH, f"percentage_{percentage_missing_values}_missing_values.npy"
    ),
    allow_pickle=True,
)

target = np.load(os.path.join(COMPLEX_PATH, f"cochains.npy"), allow_pickle=True)

input = np.load(
    os.path.join(
        COMPLEX_PATH, f"percentage_{percentage_missing_values}_input_damaged.npy"
    ),
    allow_pickle=True,
)

# Masks creation

known_indices = [
    list(known_simplices[d].values()) for d in range(considered_simplex_dim + 1)
]
missing_indices = [
    list(missing_simplices[d].values()) for d in range(considered_simplex_dim + 1)
]

sorted_input = [
    {key: int(input[k][key]) for key, _ in simplices[k].items()}
    for k in range(0, considered_simplex_dim + 1)
]
sorted_target = [
    {key: int(target[k][key]) for key, _ in simplices[k].items()}
    for k in range(0, considered_simplex_dim + 1)
]

print(sorted_input)
print(sorted_target)

sorted_input_values = [
    list(sorted_input[k].values()) for k in range(0, considered_simplex_dim + 1)
]
sorted_target_values = [
    list(sorted_target[k].values()) for k in range(0, considered_simplex_dim + 1)
]

print(sorted_input_values[0])
print(sorted_target_values[0])

# Check

for k in range(considered_simplex_dim + 1):

    known_simplices_keys = list(known_simplices[k].keys())
    known_simplices_indices = list(known_simplices[k].values())

    for i in range(len(known_simplices[k])):
        key = known_simplices_keys[i]
        assert sorted_input[k][key] == sorted_target[k][key]

        index = known_simplices_indices[i]
        assert sorted_input_values[k][index] == sorted_target_values[k][index]

# Target preparation

targets = []

for k in range(0, considered_simplex_dim + 1):
    # shape (1, num_simplices_dim_k)
    target = torch.tensor(
        sorted_target_values[k], dtype=torch.float, requires_grad=False, device=device
    )
    target = target.unsqueeze(0)

    targets.append(target)

# Input preparation

inputs = []

for k in range(0, considered_simplex_dim + 1):
    # shape (1, num_simplices_dim_k)
    input = torch.tensor(
        sorted_input_values[k], dtype=torch.float, requires_grad=False, device=device
    )
    input = input.unsqueeze(0)

    inputs.append(input)

# Safety check

for k in range(0, considered_simplex_dim + 1):
    assert len(inputs[k]) == len(targets[k])

avg_known_indices_ratio = sum(
    [
        float(len(known_indices[d])) / float(len(targets[d][0, :]))
        for d in range(0, considered_simplex_dim + 1)
    ]
) / (considered_simplex_dim + 1)
print(
    f"average ratio between known_indices and all indices: {avg_known_indices_ratio}\n1-percentage of missing values: {1 - (percentage_missing_values) / 100}"
)

k = 1
print(inputs[k][0][known_indices[k]])
print(targets[k][0][known_indices[k]])

# training


def train(
    model,
    num_epochs,
    components,
    inputs,
    known_indices,
    optimizer,
    device,
    verbose=False,
):

    for epoch in tqdm(range(0, num_epochs)):

        # (considered_simplex_dim+1, batch_size, 1, num_simplices_dim_k)
        xs = [input.clone().to(device) for input in inputs]
        considered_simplex_dim = len(xs) - 1

        optimizer.zero_grad()

        ys = model(components, xs)

        loss = torch.FloatTensor([0.0]).to(device)

        for k in range(0, considered_simplex_dim + 1):
            # compute the loss over the k-th dimension of the sample b over the known simplices
            loss += criterion(
                ys[k][0, known_indices[k]], targets[k][0, known_indices[k]]
            )

        if verbose:
            print(f"Epoch: {epoch}, loss: {round(loss.item(), 4)}")

        loss.backward()
        optimizer.step()


# Evaluation

## Params

learning_rate = 1e-3
criterion = nn.L1Loss(reduction="sum")
batch_size = 1
filter_size = 5

## Single model

# margins = [0.1, 0.05, 0.02, 0.01]
margins = [0.3, 0.2, 0.1]
component_to_use = "both"
aggregation = "sum"

# model = DeffSCNN(colors = 1).to(device)
model = MySCNN(
    filter_size, colors=1, aggregation=aggregation, component_to_use=component_to_use
).to(device)

print_number_of_parameters(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 400
train(
    model,
    num_epochs,
    components,
    inputs,
    known_indices,
    optimizer,
    device,
    verbose=True,
)

only_missing_simplices = False
accuracies = evaluate_accuracies_margins(
    [model], margins, inputs, targets, components, only_missing_simplices
)
print(accuracies)


## Set of models

n_models = 3
aggregation = "sum"
component_to_use = "both"
num_epochs = 400
margins = [0.3, 0.2, 0.1, 0.05]

models = []

for i in range(n_models):
    # model = DeffSCNN(colors = 1).to(device)
    model = MySCNN(
        filter_size,
        colors=1,
        aggregation=aggregation,
        component_to_use=component_to_use,
        keep_separated=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(
        model,
        num_epochs,
        components,
        inputs,
        known_indices,
        optimizer,
        device,
        verbose=False,
    )
    models.append(model)

accuracies = evaluate_accuracies_margins(
    models, margins, inputs, targets, components, only_missing_simplices=True
)
summarize_accuracies(accuracies, margins)

## Grid search

n_models = 5
models = []
num_epochs = 400
aggregations = ["MLP", "sum"]
components_to_use = ["irr", "sol", "both"]
margins = [0.1, 0.05, 0.02, 0.01]

for comp in components_to_use:

    for aggregation in aggregations:

        if comp != "both" and aggregation != "sum":
            continue

        print(f"Components to use: {comp}, aggregation: {aggregation}")

        for i in range(n_models):
            model = MySCNN(
                filter_size, colors=1, aggregation=aggregation, component_to_use=comp
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train(
                model,
                num_epochs,
                components,
                inputs,
                known_indices,
                optimizer,
                device,
                verbose=False,
            )
            models.append(model)

        accuracies = evaluate_accuracies_margins(models, margins)
        summarize_accuracies(accuracies, margins)
