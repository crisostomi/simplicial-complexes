# deep learning
import torch
import numpy as np

# misc
import os

# ad hoc
from mesh_normals.utils.meshes import plot_mesh
from mesh_normals.utils.io import save_dict, load_mesh_positions_triangles
from mesh_normals.utils.simplices import create_simplices, create_signals, build_boundaries, build_laplacians

# params
data_folder = 'data'

mesh_name = 'bob'

# mesh loading
positions, triangles = load_mesh_positions_triangles(mesh_name, data_folder)
num_triangles, num_vertices = triangles.shape[0], positions.shape[0]

plot_mesh(positions, triangles, mesh_name)

# data sorting
triangles = sorted(triangles, key=lambda tr: (tr[0], tr[1], tr[2]))
triangles = torch.stack(triangles)

# adding noise
sigma = 10
noisy_positions = np.random.normal(positions, sigma)


# simplices creation
simplices = create_simplices(triangles)


# signal creation
node_signals, triangle_signals = create_signals(triangles, positions)
noisy_node_signals, _ = create_signals(triangles, noisy_positions)

assert len(triangle_signals) == num_triangles

# incidence matrices and Laplacians creation
boundaries = build_boundaries(simplices)

laplacians = build_laplacians(boundaries)

# save
to_save = {'laplacians': laplacians, 'boundaries': boundaries, 'positions': node_signals,
                      'noisy_positions': noisy_node_signals, 'original_positions': positions,
                      'normals': triangle_signals, 'triangles': triangles}

for elem_name, elem_data in to_save.items():
    filename = f'{mesh_name}_{elem_name}'
    path = os.path.join(data_folder, filename)
    if isinstance(elem_data, dict):
        save_dict(elem_data, path)
    else:
        np.save(path, elem_data)
