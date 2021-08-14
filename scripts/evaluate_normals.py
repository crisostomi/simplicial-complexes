# deep learning
import torch
import torch.nn.functional as F
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
from mesh_normals.utils.meshes import plot_mesh
from mesh_normals.utils.io import load_dict
from mesh_normals.utils.misc import normalize_vector

# misc
import os
import math
import random

# config
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_folder = 'data'

# run settings
USE_NOISY = True
mesh_name = 'bob'

# data loading

to_load = {'laplacians': list, 'boundaries': list, 'positions': dict,
                      'noisy_positions': dict, 'original_positions': list,
                      'normals': dict, 'triangles': list}
loaded_data = {}

for elem_name, elem_type in to_load.items():
    filename = f'{mesh_name}_{elem_name}'
    path = os.path.join(data_folder, filename)

    if elem_type == dict:
        loaded_data[elem_name] = load_dict(path)
    else:
        loaded_data[elem_name] = np.load(path, allow_pickle=True)

laplacians, boundaries = loaded_data['laplacians'], loaded_data['boundaries']
original_positions, triangles = loaded_data['original_positions'], loaded_data['triangles']

signals = {0: {}, 1: {}, 2: {}}
signals[0], signals[2] = loaded_data['positions'], loaded_data['normals']

noisy_node_signals = loaded_data['noisy_positions']

# Preparation

# Boundaries and laplacians loading
#
# max_simplex_dim = 2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# components = {}
# components['lap'] = [coo2tensor(normalize(laplacians[i], half_interval=True)).to(device) for i in
#                      range(max_simplex_dim + 1)]
#
# Bt_s = [boundaries[i].transpose() for i in range(max_simplex_dim)]
# Bs = [boundaries[i] for i in range(max_simplex_dim)]
#
#
# Bt_s = [None] + Bt_s
# Bs = [None] + Bs
#
# num_simplices = [L.shape[0] for L in components['lap']]
# print(num_simplices)
#
# # Signal preparation
#
# # Input
#
# inputs = {0: [], 1: [], 2: []}
#
# node_signal = noisy_node_signals if USE_NOISY else signals[0]
# node_signal = [torch.tensor(signal) for signal in list(signals[0].values())]
# node_signal = torch.stack(node_signal)
# inputs[0] = node_signal
#
# print(node_signal.shape)
#
# # Lift node signal to edges
#
# edge_signal = Bs[1].T @ node_signal
# print(edge_signal.shape)
#
# print(edge_signal)
# inputs[1] = torch.tensor(edge_signal)
#
# # Create triangle signal
#
# num_triangles = Bs[2].shape[1]
# triangle_signal = torch.rand((num_triangles, 3))
# inputs[2] = triangle_signal
#
# targets = [torch.tensor(signal) for signal in list(signals[2].values())]
# targets = torch.stack(targets)
#
# targets = targets.to(device).float()
#
# inputs = list(inputs.values())
# inputs = [input.transpose(1, 0) for input in inputs]
#
#
# Bs = [coo2tensor(B).to(device) for B in Bs if B is not None]
# Bt_s = [coo2tensor(Bt).to(device) for Bt in Bt_s if Bt is not None]
#
# ## Params
#
# learning_rate = 1e-3
# criterion = nn.MSELoss(reduction="mean")
# filter_size = 30
#
# ## Loop
#
# model = MySCNN(filter_size, colors=3).to(device)
#
# model = LinearBaseline(num_nodes, num_triangles).to(device)
#
# model = TopologyAwareBaseline(num_nodes, num_triangles, node_triangle_adj).to(device)
#
# print_number_of_parameters(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# num_epochs = 400
#
# train(model, num_epochs, components, inputs, targets, Bs, Bt_s, optimizer, device, verbose=True)
#
# # Evaluation
#
# xs = [input.clone().to(device) for input in inputs]
# # ys = model(xs, components, Bs, Bt_s)
# ys = model(xs[0])
# num_triangles = inputs[2].shape[1]
# assert num_triangles == len(ys)
#
#
# x_diffs, y_diffs, z_diffs = 0, 0, 0
# arcs = []
# predicted_norms = []
#
# for i in range(num_triangles):
#     x_diff, y_diff, z_diff = abs(ys[i] - targets[i])
#     y_normalized = normalize_vector(ys[i])
#     angle = y_normalized.float() @ targets[i]
#     arc = np.arccos(angle.cpu().detach().numpy()) / np.pi * 180
#     arcs.append(arc)
#     predicted_norms.append(y_normalized)
#     x_diffs += x_diff
#     y_diffs += y_diff
#     z_diffs += z_diff
#
# print(f'Average differences:')
# print(f'\tx: {(x_diffs / num_triangles).item()}')
# print(f'\ty: {(y_diffs / num_triangles).item()}')
# print(f'\tz: {(z_diffs / num_triangles).item()}')
# print(f'average angle: {np.mean(arcs)}')
#
# positions = original_positions
#
# # plots
#
# triangles = np.stack(triangles)
#
# eval_targets = targets.cpu().numpy()
#
# norm_targets = 255 * (eval_targets - np.min(eval_targets)) / np.ptp(eval_targets)
# norm_targets = list(norm_targets)
#
# norm_targets_colors = [f'rgb({x}, {y}, {z})' for x, y, z in norm_targets]
#
#
# plot_mesh(positions, triangles, norm_targets_colors, 'True normals')
#
# predicted_norms = [pred_norm.detach().cpu().numpy() for pred_norm in predicted_norms]
# predicted_norms = 255 * (predicted_norms - np.min(predicted_norms)) / np.ptp(predicted_norms)
# predicted_norms = list(predicted_norms)
#
# predicted_norm_colors = [f'rgb({x}, {y}, {z})' for x, y, z in predicted_norms]
#
# plot_mesh(positions, triangles, predicted_norm_colors, 'Predicted normals')
#
