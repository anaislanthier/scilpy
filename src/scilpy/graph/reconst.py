import numpy as np
from dipy.reconst.shm import sh_to_sf
from dipy.core.sphere import Sphere
from scipy.sparse import coo_array, diags
from scipy.ndimage import binary_dilation

from dipy.data import get_sphere
from dipy.reconst.bingham import sh_to_bingham, _single_bingham_to_sf


def build_sparse_array(nodes_mask, sh, sh_basis, is_legacy, threshold, nufo, wm_mask):
    """
    Build the sparse array for a given set of nodes, with weights.
    For each voxel in nodes_mask, one node is created per fiber orientation.
    Each node is connected to all other nodes in its voxel's neighborhod.
    Neighborhood size is 5x5x5 voxels. 

    Parameters
    ----------
    nodes_mask : np.ndarray
        Binary mask of voxels that will be graph nodes. Neighboring voxels 
        outside the mask can still be connected to as targets.
    sh : ndarray
        Spherical harmonics volume. Ex: ODF or fODF.
    sh_basis : str
        Name of the SH basis convention.
    is_legacy : bool
        Whether to use the legacy SH basis convention.
    threshold : float
        Minimum edge weight below which an edge is discarded.
    nufo : int
        Maximum number of fiber orientations per voxel.
    wm_mask : np.ndarray
        White matter mask, used to constrain edge weight computation.

    Returns
    -------
    csr : scipy.sparse.csr_array
        Normalized sparse adjacency matrix. Weights are stored as 1/w so that
        shortest-path algorithms minimize cost.
    node_to_voxel : np.ndarray
        Maps each node ID to its voxel coordinates and direction index:
        node_to_voxel[node_id] = [x, y, z, direction_index].
    voxel_to_node : np.ndarray
        Maps each voxel and direction index to a node ID:
        voxel_to_node[x, y, z, i] = node_id, or -1 if no node exists.
    peak_dirs : np.ndarray
        Peak direction of each node.
        peak_dirs[x, y, z, direction_index] = [i, j ,k]
    """
    
    # Create a mask that will include neighboring voxels of nodes with a graph label
    footprint = np.ones((5, 5, 5), dtype=bool)
    expanded_mask = binary_dilation(nodes_mask, structure=footprint)

    sphere = get_sphere()
    bingham_metrics = sh_to_bingham(sh, sphere, mask=expanded_mask, max_search_angle=45, sh_basis=sh_basis, legacy=is_legacy, npeaks=nufo)
    bingham_params = bingham_metrics.model_params

    peak_dirs = bingham_metrics.peak_dirs
    peak_values = bingham_metrics.fd_lobe

    n_edges = int(nodes_mask.sum()) * 98 * (nufo ** 2)

    rows = np.zeros((n_edges), dtype=int)
    cols = np.zeros((n_edges), dtype=int)
    weights = np.zeros((n_edges), dtype=float)

    n_nodes = int(expanded_mask.sum()) * nufo

    voxel_to_node = np.full_like(bingham_metrics.fd_lobe[:, :, :, :nufo], -1, dtype=np.int32)
    node_to_voxel = np.zeros((n_nodes, 4), dtype=int)

    node_count = 0
    # Create nodes for every voxel and their neighbors
    for voxel in np.argwhere(expanded_mask):
        x, y, z = voxel
        # Create a node for every peak direction
        for i in range(nufo):
            voxel_to_node[x, y, z, i] = node_count
            node_to_voxel[node_count] = [x, y, z, i]
            node_count = node_count + 1

    count = 0
    for voxel in np.argwhere(nodes_mask):
        nodes = voxel_to_node[tuple(voxel)]
        # Adds possibility of an edge to the actual node closest neighbour in 26 directions
        neighbors, directions = _get_neighbors(
            voxel, voxel_to_node=voxel_to_node)
        for node in nodes:
            for i, neighbor in enumerate(neighbors):
                w = _compute_weight_bingham(node, neighbor, directions[i], sphere, wm_mask, bingham_params, node_to_voxel)
                # w = _compute_weight_peaks(node, neighbor, node_to_voxel=node_to_voxel, 
                #                     peak_dirs=peak_dirs, peak_values=peak_values)
                if w >= threshold:
                    rows[count] = node
                    cols[count] = neighbor
                    # We will later want to minimize the path weight
                    weights[count] = 1/w
                    count = count + 1
        print(f"\r{count}/{n_edges}  ({100*count/n_edges:.1f}%)", end="", flush=True)

    n_total = int(voxel_to_node.max()) + 1

    coo = coo_array((weights[:count], (rows[:count], cols[:count])),
                    shape=(n_total, n_total)
                    )

    # Convert to a CSR as CSRs are easier to work with
    csr = coo.tocsr()

    # Normalisation
    csr = _normalize_graph(coo, csr)

    return csr, node_to_voxel, voxel_to_node, peak_dirs


def _compute_weight_peaks(start_node, connected_node, node_to_voxel, peak_dirs, peak_values):
    # Returns the dot product of the node's peak directions, multiplied by their amplitude 
    start_voxel = node_to_voxel[start_node]
    start_dir = peak_dirs[tuple(start_voxel)]
    start_val = peak_values[tuple(start_voxel)]

    connected_voxel = node_to_voxel[connected_node]
    connected_dir = peak_dirs[tuple(connected_voxel)]
    connected_val = peak_values[tuple(connected_voxel)]

    w = np.abs(np.dot(start_dir * start_val, connected_dir * connected_val))
    
    return w


def _compute_weight_bingham(start, end, direction, sphere, wm_mask, bingham_params, node_to_voxel, theta_max=4*np.pi/26):

    # Get voxel and lobe id ([x, y, z, lobe])
    vox_start = node_to_voxel[start]
    vox_end = node_to_voxel[end]

    if wm_mask is not None: 
        # Probability of both nodes being in the white matter
        pmat_start = wm_mask[tuple(vox_start[:3])]
        pmat_end = wm_mask[tuple(vox_end[:3])]
        pmat = pmat_start * pmat_end
    else:
        # If no mask is given, we assume that all nodes are part of the white matter
        pmat = 1

    # Reconstruct ODF lobe corresponding to the node
    lobe_params_end = bingham_params[tuple(vox_end)]

    f0  = lobe_params_end[0]
    k1  = lobe_params_end[1]
    k2  = lobe_params_end[2]
    mu1 = lobe_params_end[6:9]
    mu2 = lobe_params_end[9:12]

    sf_end = _single_bingham_to_sf(f0, k1, k2, mu1, mu2, sphere.vertices)

    # Define a cone to calculate the alignments of the ODFs  
    dir_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction /= dir_norm

    mask = np.dot(sphere.vertices, direction) > np.cos(theta_max)

    pdiff = np.sum(sf_end[mask])

    return pmat * pdiff / dir_norm


def _get_neighbors(voxel, voxel_to_node):
    # Get neighboring nodes of a voxel
    x, y, z = voxel
    shape = voxel_to_node.shape[:3]
    neighbors = []
    directions = []
    for x_offset in range(-2, 3):
        for y_offset in range(-2, 3):
            for z_offset in range(-2, 3):
                if x_offset % 2 == 0 and y_offset % 2 == 0 and z_offset % 2 == 0:
                    continue
                nx, ny, nz = x + x_offset, y + y_offset, z + z_offset
                if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                    continue

                neighbor_voxel = [nx, ny, nz]
                neighbor_nodes = voxel_to_node[tuple(neighbor_voxel)]
                
                for neighbor_node in neighbor_nodes:
                    if neighbor_node != -1:
                        neighbors.append(neighbor_node)
                        directions.append([x_offset, y_offset, z_offset])
    return neighbors, directions


def _normalize_graph(coo, csr):
    # We want to keep all weights below 0.5
    row_max = np.zeros(csr.shape[0], dtype=csr.dtype)
    np.maximum.at(row_max, coo.row, coo.data)   

    # Avoid division by 0
    scale = scale = 0.5 / np.clip(row_max, a_min=1e-10, a_max=None)
    csr_norm = diags(scale) @ csr   

    # Edge weight = weight(A -> B) + weight(B -> A)
    csr = csr_norm + csr_norm.T

    return csr