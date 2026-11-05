import numpy as np
from scipy.sparse import csr_array

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


def save_graph_sparse(filename, csr, node_to_voxel, voxel_to_node, nodes_mask, affine, peak_dirs):
    """
    Save a sparse graph and its node mapping arrays to a single .npz file.

    Parameters
    ----------
    filename : str
        Output file path.
    csr : scipy.sparse.csr_array
        Sparse adjacency matrix.
    node_to_voxel : np.ndarray
        Maps each node ID to [x, y, z, direction_index].
    voxel_to_node : np.ndarray
        Maps each voxel and direction index to a node ID (-1 if none).
    nodes_mask : 
        this is probably temporary
    """
    np.savez(filename,
             csr_data=csr.data,
             csr_indices=csr.indices,
             csr_indptr=csr.indptr,
             csr_shape=csr.shape,
             node_to_voxel=node_to_voxel,
             voxel_to_node=voxel_to_node,
             nodes_mask=nodes_mask,
             affine=affine,
             peak_dirs=peak_dirs)


def load_graph_sparse(filename):
    """
    Load a sparse graph and its node mapping arrays from a single .npz file.

    Parameters
    ----------
    filename : str
        Path to the .npz file.

    Returns
    -------
    csr : scipy.sparse.csr_array
        Sparse adjacency matrix.
    node_to_voxel : np.ndarray
        Maps each node ID to [x, y, z, direction_index].
    voxel_to_node : np.ndarray
        Maps each voxel and direction index to a node ID (-1 if none).
    nodes_mask :
        binary mask of which voxels are assigned to graph nodes. 
    affine : 
        this is temporary (probably)
    peak_dirs :
        Peak directions of nodes

    """

    f = np.load(filename)
    csr = csr_array((f["csr_data"], f["csr_indices"], f["csr_indptr"]),
                    shape=tuple(f["csr_shape"]))
    node_to_voxel = f["node_to_voxel"]
    voxel_to_node = f["voxel_to_node"]
    nodes_mask = f["nodes_mask"]
    affine = f["affine"]
    peak_dirs = f["peak_dirs"]

    return csr, node_to_voxel, voxel_to_node, nodes_mask, affine, peak_dirs

def show_graph_2d(csr, node_to_voxel, output_name, path=None, save_only=False, axis='axial', heatmap=False, paths=None):
    
    G = nx.from_scipy_sparse_array(csr)
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    w_min, w_max = min(edge_weights), max(edge_weights)
    norm = mcolors.Normalize(vmin=w_min, vmax=w_max)
    
    cmap = LinearSegmentedColormap.from_list("dark_gray", [ "#111111", "#F6F6F6"])
    edge_colors = [cmap(norm(w) + 0.1) for w in edge_weights]
    edge_widths = [0.5 + 2.5 * norm(w) for w in edge_weights]

    if axis == 'axial':
        pos = {node: (node_to_voxel[node][0], node_to_voxel[node][1]) 
               for node in G.nodes()}
    if axis == 'coronal':
            pos = {node: (node_to_voxel[node][0], node_to_voxel[node][2]) 
                   for node in G.nodes()}
    if axis == 'sagittal':
            pos = {node: (node_to_voxel[node][1], node_to_voxel[node][2]) 
                   for node in G.nodes()}

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(G, pos=pos,
        node_size=20,
        node_color="steelblue",
        edge_color=edge_colors,
        width=edge_widths,
        font_color="black",
        ax=ax)

    # Shortest path
    if path is not None and heatmap is False:
        _add_path(path, G, pos, ax)

    # Heatmap
    if heatmap is True and paths is not None:
        _add_heatmap(paths, G, pos, ax)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Poids", shrink=0.5)
    
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.savefig(output_name, dpi=150)
    if not save_only:
        plt.show()

def _add_path(path, G, pos, ax):
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G, pos=pos,
                           edgelist=path_edges,
                           edge_color="red",
                           width=3,
                           ax=ax)
    # Start and end nodes
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=[path[0], path[-1]],
                           node_color="red",
                           node_size=80,
                           ax=ax)

def _add_heatmap(paths, G, pos, ax):
    paths = paths[::-1]
    cmap = LinearSegmentedColormap.from_list("heatmap", ["#4D0000", "#870000", "#C84300", "#FFC800",  "#FBF382"])
    n = len(paths)

    for i, path in enumerate(paths):
        color = cmap(i / max(n - 1, 1))
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=path_edges,
                               edge_color=[color],
                               width=5,
                               ax=ax,
                               alpha=0.8)