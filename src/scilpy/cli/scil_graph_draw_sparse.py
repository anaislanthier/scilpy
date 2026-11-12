#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

from scilpy.graph.io import load_graph_sparse, show_graph_2d
from scilpy.graph.solve import solve_dijkstra_single_path, solve_dijkstra_multiple_paths

"""
Tool to visualize the graph constructed with
diffusion data (white matter mask and fodf peaks).

For shortest_path and heatmap options, the user cannot choose
the starting point from the command line interface. 

If you want to modify the starting point, you have to modify the default 
value in graph/solve.py. This is, of course, temporary. 

For now, fuses all slices together. 
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_graph",
                   help="Graph file (npz file)")
    p.add_argument("out_image",
                   help="Output image file name (png file)")
    p.add_argument("--fig_type", choices=['graph', 'shortest_path', 'heatmap'], default='graph',
                   help="Type of figure you want to draw")
    p.add_argument('--axis_name', choices=['sagittal', 'coronal', 'axial'], default='axial',
                   help='Axis to draw. [%(default)s]')
    p.add_argument(
        "--save_only",
        help="Save only the figure without displaying it",
        action="store_true",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    output_name = os.path.basename(args.out_image)

    csr, node_to_voxel, voxel_to_node, nodes_mask, _, _ = load_graph_sparse(
        args.in_graph)

    if args.fig_type == 'graph':
        show_graph_2d(csr, node_to_voxel, output_name=output_name,
                      save_only=args.save_only, axis=args.axis_name)
    if args.fig_type == 'shortest_path':
        path = solve_dijkstra_single_path(csr, node_to_voxel, voxel_to_node, nodes_mask)
        show_graph_2d(csr, node_to_voxel, output_name=output_name,
                      path=path, save_only=args.save_only, axis=args.axis_name)
    if args.fig_type == 'heatmap':
        paths = solve_dijkstra_multiple_paths(csr, node_to_voxel, voxel_to_node, nodes_mask)
        show_graph_2d(csr, node_to_voxel, output_name=output_name,
                      save_only=args.save_only, axis=args.axis_name, heatmap=True, paths=paths)

    print("Graph saved as " + str(output_name))


if __name__ == "__main__":
    main()
