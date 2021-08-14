import numpy as np
import dill
import torch_geometric
import trimesh
import os
import torch

def save_dict(dictionary, path):
    keys = list(dictionary.keys())

    values = [list(tens) for tens in list(dictionary.values())]
    values = np.array(values)
    keys_path = path + '_keys'
    values_path = path + '_values'

    with open(keys_path, 'wb+') as f:
        dill.dump(keys, f)
    np.save(values_path, values)


def load_mesh_positions_triangles(mesh_name, data_folder):

    if mesh_name == 'bob':
        mesh = trimesh.load(os.path.join(data_folder, f'{mesh_name}_tri.obj'))
        positions = torch.tensor(mesh.vertices)
        triangles = torch.tensor(mesh.faces)

    elif mesh_name.startswith('faust'):
        meshes = torch_geometric.datasets.FAUST(data_folder)
        mesh = meshes[0]
        triangles = mesh.face.transpose(1, 0)
        positions = mesh.pos

    elif mesh_name == 'dummy':
        triangles = torch.tensor([
            [0, 1, 2],
            [1, 2, 3],
            [2, 4, 5]
        ])
        positions = torch.tensor([
            [3, 2, 0],
            [5, 2, 0],
            [4, 4, 0],
            [6, 4, 0],
            [3, 6, 0],
            [2, 4, 0]
        ])

    else:
        raise NotImplementedError

    return positions, triangles
