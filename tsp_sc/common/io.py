import numpy as np
import dill

# import torch_geometric
import trimesh
import os
import torch
import yaml
import networkx as nx


def get_run_config(model_name, config):
    new_config = {"data": config["data"], "run_params": config["run_params"]}
    for k, v in config.items():
        if k == model_name:
            new_config[model_name] = v
    return new_config


def add_cli_args(config, cli_args):
    for k in cli_args.__dict__:
        if cli_args.__dict__[k] is not None:
            config[k] = cli_args.__dict__[k]
    if "starting_node" in cli_args.__dict__:
        config["data"]["starting_node"] = cli_args.starting_node


# def load_config(config_path, cli_args):
#
#     with open(config_path) as file:
#         config = yaml.safe_load(file)
#     print(config)
#
#     for k_1, v_1 in config.items():
#
#         if type(v_1) == dict:
#             for k_2, v_2 in v_1.items():
#
#                 if type(v_2) == dict:
#                     for k_3, v_3 in v_2.items():
#
#                         if k_3 in cli_args.__dict__:
#                             config[k_1][k_2][k_3] = cli_args.__dict__[k_3]
#                 else:
#                     if k_2 in cli_args.__dict__:
#                         config[k_1][k_2] = cli_args.__dict__[k_2]
#         else:
#             if k_1 in cli_args.__dict__:
#                 config[k_1] = cli_args.__dict__[k_1]
#
#     print(config)
#     return config


def load_config(config_path):

    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def save_dict(dictionary, path):
    keys = list(dictionary.keys())

    values = [list(tens) for tens in list(dictionary.values())]
    values = np.array(values)
    keys_path = path + "_keys"
    values_path = path + "_values"

    with open(keys_path, "wb+") as f:
        dill.dump(keys, f)
    np.save(values_path, values)


def load_dict(path):
    keys_path = path + "_keys"
    values_path = path + "_values.npy"

    dictionary = {}

    with open(keys_path, "rb") as f:
        keys = dill.load(f)

    values = np.load(values_path)

    for k, v in zip(keys, values):
        dictionary[k] = v

    return dictionary


def load_mesh_positions_triangles(mesh_name, data_folder):

    if mesh_name == "bob":
        mesh = trimesh.load(os.path.join(data_folder, f"{mesh_name}_tri.obj"))
        positions = torch.tensor(mesh.vertices).unsqueeze(0)
        triangles = torch.tensor(mesh.faces).unsqueeze(0)

    elif mesh_name.startswith("faust"):
        meshes = torch_geometric.datasets.FAUST(data_folder)
        mesh = meshes[0]
        triangles = mesh.face.transpose(1, 0)
        positions = mesh.pos

    elif mesh_name == "dummy":
        triangles = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 4, 5]])
        positions = torch.tensor(
            [[3, 2, 0], [5, 2, 0], [4, 4, 0], [6, 4, 0], [3, 6, 0], [2, 4, 0]]
        )

    else:
        raise NotImplementedError

    return positions, triangles


def tud_to_networkx(dataset_path, dataset_name):
    graph_filename = dataset_name + "_graph_indicator.txt"
    with open(os.path.join(dataset_path, graph_filename), "r") as f:
        graph_indicator = [int(i) - 1 for i in list(f)]

    # Nodes.
    num_graphs = max(graph_indicator)
    node_indices = []
    offset = []
    c = 0

    for i in range(num_graphs + 1):
        offset.append(c)
        c_i = graph_indicator.count(i)
        node_indices.append((c, c + c_i - 1))
        c += c_i

    graph_db = []
    for i in node_indices:
        g = nx.Graph()
        for j in range(i[1] - i[0] + 1):
            g.add_node(j)

        graph_db.append(g)

    # Edges.
    edges_filename = dataset_name + "_A.txt"
    with open(os.path.join(dataset_path, edges_filename), "r") as f:
        edges = [i.split(",") for i in list(f)]

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]
    edge_list = []
    edgeb_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        g = graph_db[g_id]
        off = offset[g_id]

        # Avoid multigraph (for edge_list)
        if ((e[0] - off, e[1] - off) not in list(g.edges())) and (
            (e[1] - off, e[0] - off) not in list(g.edges())
        ):
            g.add_edge(e[0] - off, e[1] - off)
            edge_list.append((e[0] - off, e[1] - off))
            edgeb_list.append(True)
        else:
            edgeb_list.append(False)

    # Node labels.
    node_labels_filename = dataset_name + "_node_labels.txt"
    node_labels_path = os.path.join(dataset_path, node_labels_filename)
    if os.path.exists(node_labels_path):
        with open(node_labels_path, "r") as f:
            node_labels = [str.strip(i) for i in list(f)]

        node_labels = [i.split(",") for i in node_labels]
        int_labels = []
        for i in range(len(node_labels)):
            int_labels.append([int(j) for j in node_labels[i]])

        i = 0
        for g in graph_db:
            for v in range(g.number_of_nodes()):
                g.nodes[v]["labels"] = int_labels[i]
                i += 1

    # Node Attributes.
    node_attr_filename = dataset_name + "_node_attributes.txt"
    node_attr_path = os.path.join(dataset_path, node_attr_filename)
    if os.path.exists(node_attr_path):
        with open(node_attr_path, "r") as f:
            node_attributes = [str.strip(i) for i in list(f)]

        node_attributes = [i.split(",") for i in node_attributes]
        float_attributes = []
        for i in range(len(node_attributes)):
            float_attributes.append([float(j) for j in node_attributes[i]])
        i = 0
        for g in graph_db:
            for v in range(g.number_of_nodes()):
                g.nodes[v]["attributes"] = float_attributes[i]
                i += 1

    # Edge Labels.
    edge_labels_filename = dataset_name + "_edge_labels.txt"
    edge_labels_path = os.path.join(dataset_path, edge_labels_filename)
    if os.path.exists(edge_labels_path):
        with open(edge_labels_path, "r") as f:
            edge_labels = [str.strip(i) for i in list(f)]

        edge_labels = [i.split(",") for i in edge_labels]
        e_labels = []
        for i in range(len(edge_labels)):
            if edgeb_list[i]:
                e_labels.append([int(j) for j in edge_labels[i]])

        i = 0
        for g in graph_db:
            for e in range(g.number_of_edges()):
                g.edges[edge_list[i]]["labels"] = e_labels[i]
                i += 1

    # Edge Attributes.
    edge_attr_filename = dataset_name + "_edge_attributes.txt"
    edge_attr_path = os.path.join(dataset_path, edge_attr_filename)
    if os.path.exists(edge_attr_path):
        with open(edge_attr_path, "r") as f:
            edge_attributes = [str.strip(i) for i in list(f)]

        edge_attributes = [i.split(",") for i in edge_attributes]
        e_attributes = []
        for i in range(len(edge_attributes)):
            if edgeb_list[i]:
                e_attributes.append([float(j) for j in edge_attributes[i]])

        i = 0
        for g in graph_db:
            for e in range(g.number_of_edges()):
                g.edges[edge_list[i]]["attributes"] = e_attributes[i]
                i += 1

    # Classes.
    class_filename = dataset_name + "_graph_labels.txt"
    class_path = os.path.join(dataset_path, class_filename)
    if os.path.exists(class_path):
        with open(class_path, "r") as f:
            classes = [str.strip(i) for i in list(f)]

        classes = [i.split(",") for i in classes]
        cs = []
        for i in range(len(classes)):
            cs.append([int(j) for j in classes[i]])

        i = 0
        for g in graph_db:
            g.graph["classes"] = cs[i]
            i += 1

    # # Targets.
    targets_filename = dataset_name + "_graph_attributes.txt"
    targets_path = os.path.join(dataset_path, targets_filename)
    if os.path.exists(targets_path):
        with open(targets_path, "r") as f:
            targets = [str.strip(i) for i in list(f)]

        targets = [i.split(",") for i in targets]
        ts = []
        for i in range(len(targets)):
            ts.append([float(j) for j in targets[i]])

        i = 0
        for g in graph_db:
            g.graph["targets"] = ts[i]
            i += 1

    return graph_db
