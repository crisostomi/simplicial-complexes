# deep learning
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize
import torch.nn as nn
import torch.optim
import torch.utils.data as data

# meshes and graphs
import trimesh
import networkx as nx

# data
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import linalg
from scipy.sparse import coo_matrix
from scipy.linalg import eig
from scipy.linalg import null_space

# i/o
import dill
import json

# ad hoc
from inter_order.utils.meshes import plot_mesh, transform_normals_to_rgb
from inter_order.utils.io import load_dict, print_evaluation_report
from inter_order.utils.misc import *
from inter_order.models.scnn import MySCNN
from inter_order.utils.simplices import normalize_laplacian

# misc
import os
import math
import random

# config

TORCH_VERSION = torch.__version__[:5]
CUDA_VERSION = torch.version.cuda.replace(".", "")
print(f"Running torch version {TORCH_VERSION}, with CUDA version {CUDA_VERSION}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_folder = "data/inter_order"

# run settings
USE_NOISY = True
mesh_name = "bob"

# data loading

to_load = {
    "laplacians": list,
    "boundaries": list,
    "positions": dict,
    "noisy_positions": dict,
    "original_positions": list,
    "normals": dict,
    "triangles": list,
}
loaded_data = {}
load_folder = os.path.join(data_folder, mesh_name)

for elem_name, elem_type in to_load.items():
    path = os.path.join(load_folder, elem_name)

    if elem_type == dict:
        loaded_data[elem_name] = load_dict(path)
    else:
        loaded_data[elem_name] = np.load(path + ".npy", allow_pickle=True)

laplacians, boundaries = loaded_data["laplacians"], loaded_data["boundaries"]
original_positions, triangles = (
    loaded_data["original_positions"],
    loaded_data["triangles"],
)

signals = {}
signals[0] = loaded_data["noisy_positions"] if USE_NOISY else loaded_data["positions"]
signals[2] = loaded_data["normals"]

# Boundaries and laplacians loading

max_simplex_dim = 2
normalized_laplacians = [
    coo2tensor(normalize_laplacian(laplacians[i], half_interval=True)).to(device)
    for i in range(max_simplex_dim + 1)
]

Bt_s = [boundaries[i].transpose() for i in range(max_simplex_dim)]
Bs = [boundaries[i] for i in range(max_simplex_dim)]
Bt_s = [None] + Bt_s
Bs = [None] + Bs

num_nodes, num_edges, num_triangles = [L.shape[0] for L in normalized_laplacians]
print(f"There are {num_nodes} nodes, {num_edges} edges and {num_triangles} triangles")


# Signal preparation
inputs = {0: [], 1: [], 2: []}

# node signal
node_signal = [torch.tensor(signal) for signal in list(signals[0].values())]
node_signal = torch.stack(node_signal)
inputs[0] = node_signal


# lift node signal to edges
edge_signal = Bs[1].T @ node_signal
inputs[1] = torch.tensor(edge_signal)

# create triangle signal
triangle_signal = torch.rand((num_triangles, 3))
inputs[2] = triangle_signal

targets = [torch.tensor(signal).float() for signal in list(signals[2].values())]
targets = torch.stack(targets)

targets = targets.to(device)

inputs = list(inputs.values())
inputs = [input.transpose(1, 0).float() for input in inputs]

Bs = [coo2tensor(B).to(device) for B in Bs if B is not None]
Bt_s = [coo2tensor(Bt).to(device) for Bt in Bt_s if Bt is not None]

# training params

learning_rate = 1e-3
filter_size = 30

# loop

# model = LinearBaseline(num_nodes, num_triangles).to(device)
# model = TopologyAwareBaseline(num_nodes, num_triangles, node_triangle_adj).to(device)
model = MySCNN(filter_size, colors=3).to(device)

print_number_of_parameters(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 400

train(
    model,
    num_epochs,
    normalized_laplacians,
    inputs,
    targets,
    Bs,
    Bt_s,
    optimizer,
    device,
    verbose=True,
)

# Evaluation

with torch.no_grad():
    xs = [input.clone().to(device) for input in inputs]
    preds = model(xs, normalized_laplacians, Bs, Bt_s)
    # ys = model(xs[0])
    num_triangles = inputs[2].shape[1]
    assert num_triangles == len(preds)

per_coord_diffs = compute_per_coord_diff(preds, targets)
normalized_preds = normalize(preds, dim=1)

angle_diff = compute_angle_diff(normalized_preds, targets)
print_evaluation_report(per_coord_diffs, angle_diff)

positions = original_positions

# plots

triangles = np.stack(triangles)

target_norm_colors = transform_normals_to_rgb(targets)
plot_mesh(positions, triangles, "True normals", target_norm_colors)

predicted_norm_colors = transform_normals_to_rgb(normalized_preds)
plot_mesh(positions, triangles, "Predicted normals", predicted_norm_colors)
