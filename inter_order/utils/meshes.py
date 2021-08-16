import numpy as np
import plotly.figure_factory as ff


def compute_normal(triangle):
    assert triangle.shape[0] == 3
    v1, v2, v3 = triangle

    A = v2 - v1
    B = v3 - v1

    N = np.zeros(3)

    N[0] = A[1] * B[2] - A[2] * B[1]
    N[1] = A[2] * B[0] - A[0] * B[2]
    N[2] = A[0] * B[1] - A[1] * B[0]

    norm = np.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)

    return N / norm


def plot_mesh(positions, triangles, title, colors=None):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    if colors:
        fig = ff.create_trisurf(x=x, y=y, z=z,
                                simplices=triangles,
                                title=title, aspectratio=dict(x=1, y=1, z=0.3), color_func=colors)
    else:
        fig = ff.create_trisurf(x=x, y=y, z=z,
                                simplices=triangles,
                                title=title, aspectratio=dict(x=1, y=1, z=0.3))
    fig.show()

def transform_normals_to_rgb(normals):
    normals = [norm.detach().cpu().numpy() for norm in normals]
    normals = 255 * (normals - np.min(normals)) / np.ptp(normals)
    normals = list(normals)

    normals_colors = [f'rgb({x}, {y}, {z})' for x, y, z in normals]
    return normals_colors