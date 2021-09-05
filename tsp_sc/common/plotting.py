def get_positions(simplices, dim):
    """
    simplices: list of dictionaries, simplices[i] = { simplex_dim_i_0, : id_i_0, ... , simplex_dim_i_n: id_i_n}
               i.e. the i-th position in simplices is a dictionary containing simplices of dimension i mapped to a progressive integer id
    """
    polygons = list()
    for i, simplex in enumerate(simplices[dim].keys()):
        # dictionary is ordered
        assert simplices[dim][simplex] == i
        polygon = list()
        for vertex in simplex:
            polygon.append(points[vertex])
        polygons.append(polygon)
    return polygons


lines = get_positions(simplices, 1)
triangles = get_positions(simplices, 2)


def value2color(values):
    values -= values.min()
    values /= values.max()
    return mpl.cm.viridis(values)


def plot_nodes(colors, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c=colors, **kwargs)
    return ax.figure, ax


def plot_edges(colors, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subfigs()
    colors = value2color(colors)
    collection = mpl.collections.LineCollection(lines, colors=colors, **kwargs)
    ax.add_collection(collection)
    ax.autoscale()


def plot_triangles(colors, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subfigs()
    colors = value2color(colors)
    for triangle, color in zip(triangles, colors):
        triangle = plt.Polygon(triangle, color=color, **kwargs)
        ax.add_patch(triangle)
    ax.autoscale()


def plot_triangles_plain(ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subfigs()
    colors = len(simplices[1]) * ["linen"]
    for triangle, color in zip(triangles, colors):
        triangle = plt.Polygon(triangle, color=color, **kwargs)
        ax.add_patch(triangle)
    ax.autoscale()
