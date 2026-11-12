import numpy as np
from dipy.reconst.shm import sh_to_sf
from scipy.sparse import coo_array
from scipy.ndimage import binary_dilation

from dipy.data import get_sphere
from dipy.reconst.bingham import sf_to_bingham, _single_bingham_to_sf
from dipy.reconst.shm import sh_to_sf, calculate_max_order


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
    expanded_mask = binary_dilation(nodes_mask.astype(bool), structure=footprint)

    sphere = get_sphere(name="repulsion724")

    sf = np.zeros(sh.shape[:3] + (len(sphere.vertices),))
    sf[expanded_mask] = sh_to_sf(sh[expanded_mask], sphere,
                                 sh_order_max=calculate_max_order(sh.shape[-1]),
                                 basis_type=sh_basis, legacy=is_legacy)


    sf_max = np.clip(sf.max(axis=-1, keepdims=True), 0, None)
    sf = np.maximum(sf, 0.01 * sf_max)

    bingham_metrics = sf_to_bingham(sf, sphere, 25, mask=expanded_mask,
                                    npeaks=nufo, min_sep_angle=25)
    bingham_params = bingham_metrics.model_params

    peak_dirs = bingham_metrics.peak_dirs
    peak_values = bingham_metrics.fd_lobe

    n_edges = int(nodes_mask.sum()) * 98 * (nufo ** 2)

    rows = np.zeros((n_edges), dtype=int)
    cols = np.zeros((n_edges), dtype=int)
    weights = np.zeros((n_edges), dtype=float)

    n_nodes = int(expanded_mask.sum()) * nufo

    voxel_to_node = np.full_like(
        bingham_metrics.fd_lobe[:, :, :, :nufo], -1, dtype=np.int32)
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
                w = _compute_weight_bingham(
                    node, neighbor, directions[i], sphere, wm_mask, bingham_params, node_to_voxel)
                dist = np.linalg.norm(directions[i])
                if w * dist >= threshold:
                    rows[count] = node
                    cols[count] = neighbor
                    # We will later want to minimize the path weight
                    weights[count] = 1/w
                    count = count + 1
        print(f"\r{count}/{n_edges}  ({100*count/n_edges:.1f}%)",
              end="", flush=True)

    n_total = int(voxel_to_node.max()) + 1

    coo = coo_array((weights[:count], (rows[:count], cols[:count])),
                    shape=(n_total, n_total)
                    )

    # Convert to a CSR as CSRs are easier to work with
    csr = coo.tocsr()

    # Normalisation
    csr = _normalize_graph(coo, csr)

    return csr, node_to_voxel, voxel_to_node, peak_dirs

def _compute_weight_bingham(start, end, direction, sphere, wm_mask, bingham_params, node_to_voxel, theta_max=4*np.pi/26, max_turn_deg=25):

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

    # Define a cone to calculate the alignments of the ODFs
    direction = np.asarray(direction, dtype=float)
    dir_norm = np.linalg.norm(direction)
    direction = direction / dir_norm
    mask = np.dot(sphere.vertices, direction) > np.cos(theta_max)

    lobe_params_start = bingham_params[tuple(vox_start)]
    lobe_params_end = bingham_params[tuple(vox_end)]

    p_end = _lobe_mass_in_cone(lobe_params_end, sphere, mask)
    p_start = _lobe_mass_in_cone(lobe_params_start, sphere, mask)

    # Both lobes must be aligned with the step. The geometric mean is
    # symmetric: weight(A -> B) == weight(B -> A).
    pdiff = np.sqrt(p_start * p_end)

    cos_turn = abs(np.dot(lobe_params_start[3:6], lobe_params_end[3:6]))
    turn = np.degrees(np.arccos(np.clip(cos_turn, 0, 1)))
    if turn / dir_norm > max_turn_deg:
        return 0.0

    return float(pmat * pdiff * cos_turn / dir_norm)

def _lobe_mass_in_cone(lobe_params, sphere, mask):
    f0 = lobe_params[0]
    k1 = lobe_params[1]
    k2 = lobe_params[2]
    mu1 = lobe_params[6:9]
    mu2 = lobe_params[9:12]
    sf = _single_bingham_to_sf(f0, k1, k2, mu1, mu2, sphere.vertices)
    total = np.sum(sf)
    if total <= 0:
        return 0.0
    # Fraction of the lobe inside the cone
    return np.sum(sf[mask]) / total


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
    scale = 0.5 / max(csr.max(), 1e-10)
    csr_norm = csr * scale
    csr_norm_t = csr_norm.T

    has_fwd = (csr_norm != 0).astype(float)
    has_rev = (csr_norm_t != 0).astype(float)

    # Seen in both directions: weight(A -> B) + weight(B -> A)
    both = csr_norm.multiply(has_rev) + csr_norm_t.multiply(has_fwd)

    # Seen in one direction only: count it twice, so that a missing
    # direction does not make an edge cheaper
    one_way = (csr_norm - csr_norm.multiply(has_rev)) + \
              (csr_norm_t - csr_norm_t.multiply(has_fwd))

    return (both + 2 * one_way).tocsr()