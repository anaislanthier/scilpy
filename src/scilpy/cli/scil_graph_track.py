import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

from scilpy.graph.io import load_graph_sparse
from scilpy.graph.solve import solve_dijkstra_multiple_paths, path_is_valid
"""
Prototype to create streamlines from a graph.
Only produces streamlines starting and ending in graph regions. 
Seeds from the perimeter of the graph region. 
This is temporary.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_graph",
                   help="Graph file (.npz file)")
    p.add_argument("out_tractogram",
                   help="Output file name (.trk file)")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    csr, node_to_voxel, voxel_to_node, nodes_mask, affine, peak_dirs = load_graph_sparse(
        args.in_graph)

    # Create seeding and exit masks
    footprint_3 = np.ones((3, 3, 3), dtype=bool)
    footprint_5 = np.ones((5, 5, 5), dtype=bool)

    expanded_mask = binary_dilation(nodes_mask.astype(bool), structure=footprint_5)
    exit_mask = expanded_mask & ~nodes_mask.astype(bool)

    exit_dilated = binary_dilation(exit_mask, structure=footprint_3)
    perimeter_mask = exit_dilated & nodes_mask.astype(bool)

    nib.save(nib.Nifti1Image(exit_mask.astype(np.uint8), affine),
             "exit_mask.nii.gz")

    nib.save(nib.Nifti1Image(perimeter_mask.astype(np.uint8), affine),
             "perimeter_mask.nii.gz")

    streamlines = []

    # Create streamlines starting from each node in the perimeter of the graph
    for voxel in np.argwhere(perimeter_mask):
        for node in voxel_to_node[tuple(voxel)]:
            paths = solve_dijkstra_multiple_paths(csr, node_to_voxel, 
                                                  voxel_to_node, nodes_mask,
                                                  start_node=node,
                                                  exit_mask=exit_mask,
                                                  nb_paths=10)
            for path in paths:
                path_voxel = [node_to_voxel[p][:3] for p in path]
                if path_is_valid(path_voxel):
                    streamlines.append(np.array(path_voxel, dtype=np.float32))

    # Should we use expanded_mask instead of nodes_mask?
    reference_img = nib.Nifti1Image(nodes_mask.astype(np.uint8), affine)

    our_space = Space.VOX
    our_origin = Origin('center')

    sft = StatefulTractogram(streamlines, reference_img,
                             space=our_space, origin=our_origin)

    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
