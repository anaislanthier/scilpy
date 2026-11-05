#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import nibabel as nib

from scilpy.graph.reconst import build_sparse_array
from scilpy.graph.filter import remove_orphan_nodes_sparse
from scilpy.graph.io import save_graph_sparse
from scilpy.io.utils import add_sh_basis_args, parse_sh_basis_arg

"""
Tool to build sparse array from diffusion data
"""

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_nufo_map", 
                   help="Input NUFO map. (.nii.gz file)")
    # Ony one label for now, will need to support multiple labels later.
    # This is just for a simple first draft implementation.
    p.add_argument("in_nufo_label",
                   type=int,
                   help="Label corresponding to graph tracking on the NUFO map.")
    p.add_argument("in_fodfs", 
                   help="Input fODFs image. (.nii.gz file)")
    p.add_argument("out_graph",
                   help="Output graph file name.")
    p.add_argument("--wm_mask",
                    help="White matter mask. (.nii.gz file)")
    p.add_argument(
            "--threshold",
            default=0.05,
            type=float,
            help="Cut all weights below a given threshold. [%(default)s]",
        )
    p.add_argument(
            '--sh_order_max',
            type=int,
            default=12,
            help='Maximum SH order. [%(default)s]'
        )

    add_sh_basis_args(p)
    
    return p

def main():

    parser = _build_arg_parser()
    args = parser.parse_args()
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    nufo_map = nib.load(args.in_nufo_map)

    fodfs = nib.load(args.in_fodfs)
    sh = fodfs.get_fdata()

    wm_mask = None
    if args.wm_mask is not None: 
        wm = nib.load(args.wm_mask)
        wm_mask = wm.get_fdata()

    nodes_mask = (nufo_map.get_fdata() == args.in_nufo_label).astype(int)
    affine = nufo_map.affine

    csr, node_to_voxel, voxel_to_node, peak_dirs = build_sparse_array(nodes_mask, 
                                                           sh, 
                                                           sh_basis=sh_basis,
                                                           is_legacy=is_legacy,
                                                           threshold=args.threshold, 
                                                           nufo=args.in_nufo_label,
                                                           wm_mask=wm_mask)

    csr, node_to_voxel, voxel_to_node = remove_orphan_nodes_sparse(csr, 
                                                    node_to_voxel,
                                                    voxel_to_node)

    save_graph_sparse(args.out_graph, csr, node_to_voxel, voxel_to_node, nodes_mask, affine, peak_dirs)


if __name__ == "__main__":
    main()
