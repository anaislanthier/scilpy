import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import binary_dilation
import nibabel as nib


def solve_dijkstra_single_path(graph, voxel_to_node, nodes_mask, start_node=None, exit_mask=None):

    # This is temporary
    # Finds shortest path from one node to any node in the exit mask
    # The result is often disappointing if we don't use a precise exit mask.
    # Exits right away most of the time

    # Create exit mask if not provided
    if exit_mask is None:
        footprint = np.ones((5, 5, 5), dtype=bool)
        expanded_mask = binary_dilation(nodes_mask, structure=footprint)
        exit_mask = expanded_mask & ~nodes_mask

    if start_node is None:
        start_node = _get_start_u(voxel_to_node)

    targets = []
    for voxel in np.argwhere(exit_mask):
        nodes = voxel_to_node[tuple(voxel)]
        for node in nodes:
            targets.append(node)

    dist_matrix, predecessors = dijkstra(graph,
                                         directed=False,
                                         indices=start_node,
                                         return_predecessors=True)

    best_target = min(
        targets,
        key=lambda t: dist_matrix[t] /
        _count_nodes_in_path(predecessors, start_node, t)
    )

    path = _get_path(predecessors, start_node, best_target)
    return path


def solve_dijkstra_multiple_paths(graph, voxel_to_node, nodes_mask, node_to_voxel=None, start_node=None, exit_mask=None, nb_paths=100):

    # Create exit mask if not provided
    if exit_mask is None:
        footprint = np.ones((5, 5, 5), dtype=bool)
        expanded_mask = binary_dilation(nodes_mask, structure=footprint)
        exit_mask = expanded_mask & ~nodes_mask

    if start_node is None:
        start_node = _get_start_u(voxel_to_node)

    targets = []
    for voxel in np.argwhere(exit_mask):
        nodes = voxel_to_node[tuple(voxel)]
        for node in nodes:
            if node != -1:
                targets.append(node)

    dist_matrix, predecessors = dijkstra(graph,
                                         directed=False,
                                         indices=start_node,
                                         return_predecessors=True)

    reachable = [t for t in targets
                 if dist_matrix[t] != np.inf and predecessors[t] != -9999]

    scored = sorted(
        reachable,
        key=lambda t: dist_matrix[t] /
        _count_nodes_in_path(predecessors, start_node, t)
    )

    shortest_paths = [_get_path(predecessors, start_node, t)
                      for t in scored[:nb_paths]]

    return shortest_paths


def _count_nodes_in_path(predecessors, start, end):
    # Could probably be replaced by len(_get_path(predecessors, start, end))

    if end == -1:
        return 1e-6     # To avoid divisions by 0 in the functions
    count = 1
    node = end
    while node != start:
        if node == -9999:
            return 1e-6     # To avoid divisions by 0 in the functions
        node = predecessors[node]
        count += 1
    return count


def _get_path(predecessors, start, end):
    path = []
    node = end
    while node != start:
        if node == -9999:
            return None
        path.append(node)
        node = predecessors[node]
    path.append(start)
    return path[::-1]


def path_is_valid(path, max_angle_deg=60):
    # Checks the angle between every pair of two consecutive points
    # in a path.

    max_angle_rad = np.cos(np.radians(max_angle_deg))
    prev_direction = None
    for i in range(len(path) - 1):
        direction = path[i + 1] - path[i]
        dir_norm = direction / np.linalg.norm(direction)

        if prev_direction is not None:
            dot = np.dot(dir_norm, prev_direction)
            if dot < max_angle_rad:
                return False

        prev_direction = dir_norm
    return True

# TODO remove or move to tests.
# Hardcoded target / starting points for Fibercup tracking.


def _get_start_u(voxel_to_node):
    return voxel_to_node[25, 41, 1, 0]
    # return voxel_to_node[45, 43, 1, 1]
    # return voxel_to_node[17, 41, 1, 1]


def _get_targets_u(voxel_to_node):

    targets = []
    # Autre bout du U (on veut ça)
    targets.append(voxel_to_node[33, 43, 1, 0])
    targets.append(voxel_to_node[34, 43, 1, 0])
    targets.append(voxel_to_node[35, 43, 1, 0])
    targets.append(voxel_to_node[36, 43, 1, 0])
    targets.append(voxel_to_node[37, 43, 1, 0])
    targets.append(voxel_to_node[38, 43, 1, 0])

    # Autre bout de la ligne droite (on ne veut pas ça)
    targets.append(voxel_to_node[52, 35, 1, 0])
    targets.append(voxel_to_node[52, 36, 1, 0])
    targets.append(voxel_to_node[52, 37, 1, 0])
    targets.append(voxel_to_node[52, 38, 1, 0])
    targets.append(voxel_to_node[52, 39, 1, 0])
    targets.append(voxel_to_node[52, 40, 1, 0])
    targets.append(voxel_to_node[52, 35, 1, 1])
    targets.append(voxel_to_node[52, 36, 1, 1])
    targets.append(voxel_to_node[52, 37, 1, 1])
    targets.append(voxel_to_node[52, 38, 1, 1])
    targets.append(voxel_to_node[52, 39, 1, 1])
    targets.append(voxel_to_node[52, 40, 1, 1])

    # En bas du U (on ne veut pas ça)
    targets.append(voxel_to_node[28, 37, 1, 0])
    targets.append(voxel_to_node[29, 37, 1, 0])
    targets.append(voxel_to_node[30, 37, 1, 0])
    targets.append(voxel_to_node[31, 37, 1, 0])
    targets.append(voxel_to_node[32, 37, 1, 0])
    targets.append(voxel_to_node[28, 37, 1, 1])
    targets.append(voxel_to_node[29, 37, 1, 1])
    targets.append(voxel_to_node[30, 37, 1, 1])
    targets.append(voxel_to_node[31, 37, 1, 1])
    targets.append(voxel_to_node[32, 37, 1, 1])

    targets = [t for t in targets if t != -1]

    return targets


def _get_start_x(voxel_to_node):
    return voxel_to_node[29, 18, 1, 0]


def _get_targets_x(voxel_to_node):
    targets = []

    # Opposé (on veut ça)
    targets.append(voxel_to_node[34, 18, 1, 0])
    targets.append(voxel_to_node[35, 18, 1, 0])
    targets.append(voxel_to_node[34, 19, 1, 0])
    targets.append(voxel_to_node[33, 19, 1, 0])
    targets.append(voxel_to_node[32, 20, 1, 0])

    # Côté gauche (on ne veut pas ça)
    targets.append(voxel_to_node[28, 16, 1, 0])
    targets.append(voxel_to_node[28, 18, 1, 0])
    targets.append(voxel_to_node[28, 17, 1, 0])
    targets.append(voxel_to_node[29, 18, 1, 0])
    targets.append(voxel_to_node[30, 19, 1, 0])

    targets.append(voxel_to_node[28, 16, 1, 1])
    targets.append(voxel_to_node[28, 18, 1, 1])
    targets.append(voxel_to_node[28, 17, 1, 1])
    targets.append(voxel_to_node[29, 18, 1, 1])
    targets.append(voxel_to_node[30, 19, 1, 1])

    # Côté droit (on ne veut pas ça)
    targets.append(voxel_to_node[33, 13, 1, 0])
    targets.append(voxel_to_node[34, 13, 1, 0])
    targets.append(voxel_to_node[34, 14, 1, 0])
    targets.append(voxel_to_node[35, 14, 1, 0])

    targets.append(voxel_to_node[33, 13, 1, 1])
    targets.append(voxel_to_node[34, 13, 1, 1])
    targets.append(voxel_to_node[34, 14, 1, 1])
    targets.append(voxel_to_node[35, 14, 1, 1])

    targets = [t for t in targets if t != -1]

    return targets
