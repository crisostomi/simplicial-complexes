# deep learning
import torch
import numpy as np
import torch_geometric

# misc
import os
import dill

# ad hoc
from mesh_normals.utils.meshes import plot_mesh
from mesh_normals.utils.io import save_dict
from mesh_normals.utils.simplices import create_simplices, create_signals, build_boundaries, build_laplacians

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

PROJECT_HOME = "."
data_folder = os.path.join(PROJECT_HOME, f'data')

use_dummy_data = False

meshes = torch_geometric.datasets.FAUST(data_folder)
mesh = meshes[0]
mesh_name = 'Faust-subj0-pose0'

triangles = mesh.face.transpose(1, 0)
positions = mesh.pos
num_triangles = triangles.shape[0]
num_vertices = positions.shape[0]

plot_mesh(positions, triangles, mesh_name)

print(positions)
# Data sorting

triangles = sorted(triangles, key=lambda tr: (tr[0], tr[1], tr[2]))
triangles = torch.stack(triangles)

# Adding noise

sigma = 10
noisy_positions = np.random.normal(positions, sigma)


# Simplices creation

simplices = create_simplices(triangles)


# Signal creation

node_signals, triangle_signals = create_signals(triangles, positions)
noisy_node_signals, _ = create_signals(triangles, noisy_positions)

assert len(triangle_signals) == num_triangles


# Incidence matrices and Laplacians creation

boundaries = build_boundaries(simplices)

laplacians = build_laplacians(boundaries)


# Save
prefix = 'dummy_' if use_dummy_data else ''
laplacian_path = f'{data_folder}/{prefix}laplacians.npy'
boundaries_path = f'{data_folder}/{prefix}boundaries.npy'
positions_path = f'{data_folder}/{prefix}positions'
noisy_positions_path = f'{data_folder}/{prefix}noisy_positions'
original_positions_path = f'{data_folder}/{prefix}original_positions'
normals_path = f'{data_folder}/{prefix}normals'
triangles_path = f'{data_folder}/{prefix}triangles.npy'

np.save(laplacian_path, laplacians)
np.save(boundaries_path, boundaries)
np.save(triangles_path, triangles)
np.save(positions_path, positions)

save_dict(node_signals, positions_path)
save_dict(triangle_signals, normals_path)
save_dict(noisy_node_signals, noisy_positions_path)
