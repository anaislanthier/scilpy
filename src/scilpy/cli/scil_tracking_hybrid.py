import argparse
import random

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamlinespeed import length
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix

from scilpy.graph.io import load_graph_sparse
from scilpy.graph.solve import solve_dijkstra_multiple_paths, path_is_valid
"""
Prototype to create streamlines from local tracking and graph tracking.
This is temporary.
Does not use other scilpy functions or classes such as Tracker or Propagator. 
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_fodf",
                   help="fODFs (.nii.gz file)")
    p.add_argument("in_graph",
                   help="Graph file (.npz file)")
    p.add_argument("in_wm_mask",
                   help="White matter mask (.nii.gz file)")
    p.add_argument("out_tractogram",
                   help="Output file name (.trk file)")
    p.add_argument("--npv",
                   help="Number of seeds per voxel",
                   type=int,
                   default=1)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    wm = nib.load(args.in_wm_mask)
    wm_mask = wm.get_fdata().astype(bool)

    fodfs = nib.load(args.in_fodf)
    sh = fodfs.get_fdata()

    voxres = fodfs.header.get_zooms()[:3]

    csr, node_to_voxel, voxel_to_node, nodes_mask, affine, peak_dirs = load_graph_sparse(
        args.in_graph)

    # Create seeding mask
    # For now, we avoid seeding in the graph
    seed_mask = wm_mask & ~nodes_mask

    nib.save(nib.Nifti1Image(seed_mask.astype(np.uint8), affine),
             "seed_mask.nii.gz")

    exit_mask = get_exit_mask(nodes_mask)

    # Get necessary params for local tracking (non-graph tracking)
    sphere = get_sphere(name='symmetric724')
    dirs = sphere.vertices
    order = int((-3 + np.sqrt(1 + 8 * sh.shape[-1])) / 2)
    B, _ = sh_to_sf_matrix(sphere, sh_order_max=order,
                           basis_type='descoteaux07', legacy=True)
    cos_theta = np.cos(np.radians(20))

    streamlines = []
    count = 0
    nb_seed = np.sum(seed_mask)
    for seed in np.argwhere(seed_mask):
        for i in range(args.npv):
            line = get_line_from_seed(seed, nodes_mask, voxres, peak_dirs, csr,
                                      node_to_voxel, voxel_to_node, wm_mask, sh, B, dirs, cos_theta, exit_mask)
            if line is not None:
                streamlines.append(line)
        count += 1
        print(f"\r{count}/{nb_seed}  ({100*count/nb_seed:.1f}%)",
              end="", flush=True)

    our_space = Space.VOX
    our_origin = Origin('center')

    sft = StatefulTractogram(streamlines, wm,
                             space=our_space, origin=our_origin)

    sft.remove_invalid_streamlines()

    save_tractogram(sft, args.out_tractogram, bbox_valid_check=False)


def get_line_from_seed(seed, nodes_mask, voxres, peak_dirs, csr, node_to_voxel, voxel_to_node, wm_mask, sh, B, dirs, cos_theta, exit_mask, min_length=10):
    d0 = _get_initial_direction(seed, sh, B, dirs)

    # Forward
    line = _propagate([seed], d0, nodes_mask, voxres, peak_dirs, csr,
                      node_to_voxel, voxel_to_node, wm_mask, sh, B, dirs, cos_theta, exit_mask)

    if line is None:
        return None

    line.reverse()

    # Backward
    line = _propagate(line, -d0, nodes_mask, voxres, peak_dirs, csr,
                      node_to_voxel, voxel_to_node, wm_mask, sh, B, dirs, cos_theta, exit_mask)

    if line is None or length(np.array(line)) * voxres[0] < min_length:
        return None

    return line


def _get_initial_direction(pos, sh, B, dirs, sf_threshold_init=0.5):
    sf = sh[tuple(np.round(pos).astype(int))] @ B
    if sf.max() <= 0:
        return None
    sf = sf / sf.max()
    sf[sf < sf_threshold_init] = 0
    return dirs[np.random.choice(len(dirs), p=sf / sf.sum())]


def _propagate(line, initial_direction, nodes_mask, voxres, peak_dirs, csr, node_to_voxel, voxel_to_node, wm_mask, sh, B, dirs, cos_theta, exit_mask=None, max_length=300, step_size=0.5):

    prev_dir = initial_direction
    prev_dir /= np.linalg.norm(prev_dir, axis=-1, keepdims=True)

    # We use a cooldown to make sure that the streamline doesn't
    # immediately go back into a graph region right after exiting.
    cooldown = 0

    while length(np.array(line)) * voxres[0] < max_length:
        pos = line[-1]

        # Not the best way to do this
        # This is temporary
        pos_round = np.round(pos).astype(int)
        is_in_graph = nodes_mask[tuple(pos_round)]

        if is_in_graph and cooldown == 0:
            peak_dirs_at_pos = peak_dirs[tuple(pos_round)]

            # Get best node in the voxel depending on where we come from
            max_dot = 0
            max_idx = 0
            for i, peak_dir in enumerate(peak_dirs_at_pos):
                dir_norm = np.linalg.norm(peak_dir, axis=-1, keepdims=True)
                if dir_norm < 1e-6:
                    continue
                peak_dir = peak_dir / dir_norm
                dot = np.dot(peak_dir, prev_dir)
                if abs(dot) > max_dot:
                    max_dot = abs(dot)
                    max_idx = i

            start_node = voxel_to_node[pos_round[0],
                                       pos_round[1], pos_round[2], max_idx]
            paths = solve_dijkstra_multiple_paths(csr, node_to_voxel,
                                                  voxel_to_node, nodes_mask,
                                                  start_node=start_node,
                                                  exit_mask=exit_mask,
                                                  nb_paths=30)

            # TODO Find another way to chose the path.
            # There is surely a better way to do this
            # This is temporary
            candidate_paths = []
            candidate_endpoints = []
            for path in paths:
                path_voxel_lobe = [node_to_voxel[p] for p in path]
                path_voxel = [p[:3] for p in path_voxel_lobe]
                if path_is_valid(path_voxel):
                    candidate_paths.append(
                        np.array(path_voxel, dtype=np.float32))
                    candidate_endpoints.append(path_voxel_lobe[-1])

            if len(candidate_paths) == 0:
                # print("No valid graph paths")
                return None

            chosen_idx = random.randrange(len(candidate_paths))
            chosen_path = candidate_paths[chosen_idx]
            chosen_endpoint = candidate_endpoints[chosen_idx]

            prev_dir = peak_dirs[tuple(chosen_endpoint)]
            prev_dir = prev_dir / \
                np.linalg.norm(prev_dir, axis=-1, keepdims=True)

            seg = chosen_path[-1] - chosen_path[-2]
            if np.dot(prev_dir, seg) < 0:
                prev_dir = -prev_dir

            line.extend(chosen_path)

            # This is an arbitrary value for testing
            cooldown = 3

            if not wm_mask[tuple(chosen_endpoint[:3])]:
                return line
        else:
            next_dir = _get_next_direction(
                pos, prev_dir, sh, B, dirs, cos_theta)
            if next_dir is None:
                return line
            next_pos = pos + step_size * next_dir
            line.append(next_pos)
            prev_dir = next_dir

            if cooldown > 0:
                cooldown = cooldown - 1

            # Again, clearly there's a better way to do this
            next_pos_round = np.round(next_pos).astype(int)
            try:
                if not wm_mask[tuple(next_pos_round)]:
                    return line
            except:
                return line

    return line


def get_exit_mask(nodes_mask):

    footprint = np.ones((5, 5, 5), dtype=bool)
    expanded_mask = binary_dilation(nodes_mask.astype(bool), structure=footprint)
    exit_mask = expanded_mask & ~nodes_mask.astype(bool)

    return exit_mask


def _get_next_direction(pos, prev_dir, sh, B, dirs, cos_theta, sf_threshold=0.1):
    sf = sh[tuple(np.round(pos).astype(int))] @ B
    if sf.max() <= 0:
        return None
    sf = sf / sf.max()

    in_cone = dirs @ prev_dir >= cos_theta
    sf = np.where(in_cone & (sf >= sf_threshold), sf, 0)
    if sf.sum() == 0:
        return None

    idx = np.random.choice(len(dirs), p=sf / sf.sum())
    return dirs[idx]
