import numpy as np

def remove_orphan_nodes_sparse(csr, node_to_voxel, voxel_to_node):
    """
    Remove nodes with no connections from a CSR sparse graph.
    Updates node_to_voxel and voxel_to_node to reflect the new node IDs.

    Parameters
    ----------
    csr : scipy.sparse.csr_array
        Sparse adjacency matrix of shape (n_nodes, n_nodes).
    node_to_voxel : np.ndarray, shape (n_nodes, 4)
        Maps node ID → [x, y, z, direction_index].
    voxel_to_node : np.ndarray, shape (x, y, z, n_dirs)
        Maps voxel + direction index → node ID (or -1 if no node).

    Returns
    -------
    csr_filtered : scipy.sparse.csr_array
        Sparse adjacency matrix without orphan nodes.
    node_to_voxel_filtered : np.ndarray, shape (n_kept, 4)
        Updated node_to_voxel for kept nodes only.
    voxel_to_node_updated : np.ndarray, shape (x, y, z, n_dirs)
        Updated voxel_to_node with remapped node IDs (-1 for removed nodes).
    """
    n_total = csr.shape[0]

    # Find nodes with at least one connection
    row_sums = np.array(csr.sum(axis=1)).flatten()
    col_sums = np.array(csr.sum(axis=0)).flatten()
    node_mask = (row_sums > 0) | (col_sums > 0)

    csr_filtered = csr[node_mask][:, node_mask]

    # Filter node_to_voxel
    node_to_voxel_filtered = node_to_voxel[:n_total][node_mask]

    # Build old_id --> new_id mapping
    # Kept nodes get sequential new IDs; removed nodes map to -1
    old_to_new = np.full(n_total, -1, dtype=int)
    old_to_new[node_mask] = np.arange(node_mask.sum())

    # Update voxel_to_node: remap valid old IDs to new IDs
    voxel_to_node_updated = voxel_to_node.copy()
    valid = (voxel_to_node >= 0) & (voxel_to_node < n_total)
    voxel_to_node_updated[valid] = old_to_new[voxel_to_node[valid]]
    voxel_to_node_updated[~valid] = -1

    return csr_filtered, node_to_voxel_filtered, voxel_to_node_updated
