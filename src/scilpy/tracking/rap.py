# -*- coding: utf-8 -*-

import logging
import numpy as np
import random

from scipy.ndimage import binary_dilation

from scilpy.graph.io import load_graph_sparse
from scilpy.graph.solve import solve_dijkstra_multiple_paths, path_is_valid
from scilpy.tracking.utils import TrackingDirection

class RAP:
    def __init__(self, rap_volume, propagator):
        """
        rap_volume: DataVolume
            Region-Adaptive Propagation tractography volume.
        propagator: ODFPropagator
            The propagator.
        """
        self.rap_volume = rap_volume
        self.propagator = propagator
        self._current_label = None
        self._total_steps = 0

    def is_in_rap_region(self, curr_pos, space, origin):
        return self.rap_volume.get_value_at_coordinate(
            *curr_pos, space=space, origin=origin) > 0

    def rap_multistep_propagate(self, line, prev_direction):
        """
        All child classes must implement this method. Must receive and return
        the parameters as defined here:


        Params
        ------
        line: list
            The beginning of the streamline

        Returns
        -------
        line: list
            The streamline extended with RAP in the RAP neighborhood.
        prev_direction: tuple
            The last direction (x, y, z).
        is_line_valid: bool
            If the line generated with RAP is valid.
        """
        raise NotImplementedError


class RAPContinue(RAP):
    """Dummy RAP class for tests. Goes straight"""

    def __init__(self, rap_volume, propagator, step_size):
        """
        Step size: float
            The step size inside the RAP mask. Could be different from the step
            size elsewhere. In voxel world.
        """
        super().__init__(rap_volume, propagator)
        self.step_size = step_size

    def rap_multistep_propagate(self, line, prev_direction):
        is_line_valid = True
        if len(line) > 3:
            pos = line[-2] + self.step_size * np.array(prev_direction)
            line[-1] = pos
            return line, prev_direction, is_line_valid
        return line, prev_direction, is_line_valid


class RAPSwitch(RAP):
    """RAP class that switches tracking parameters when inside the RAP mask/label."""

    def __init__(self, rap_volume, propagators: dict):
        """
        Parameters
        ----------
        rap_volume : DataVolume
            Region-Adaptive Propagation mask.
        propagators : dict
            Dictionary of ODFPropagator instances keyed by label (str).
            If --in_odf is provided, contains {odf_path: propagator}
            as default. Additional propagators are keyed by their label,
            loaded from the 'filename' key in rap_policies.json.
        """
        base_propagator = list(propagators.values())[
            0] if propagators else None
        super().__init__(rap_volume, base_propagator)
        self._propagators = propagators

        if self.propagator is not None:
            self._base = {
                'step_size': self.propagator.step_size,
                'theta': self.propagator.theta,
                'algo': getattr(self.propagator, 'algo', None),
                'tracking_neighbours': getattr(self.propagator,
                                               'tracking_neighbours', None)
            }
        else:
            self._base = {}

        # Check if all labels in the volume are covered by the configuration
        unique_labels = np.unique(rap_volume.data)
        # Remove 0 (background) and convert to int
        unique_labels = [int(label) for label in unique_labels if label > 0]

        if unique_labels:
            missing_labels = [label for label in unique_labels
                              if str(label) not in self._propagators.keys()]
            if missing_labels:
                logging.warning(
                    f"Labels {missing_labels} found in RAP volume but not in "
                    f"methods config. Base params will be used for these labels."
                )

    def rap_multistep_propagate(self, line, prev_direction):
        """
        Propagate within the RAP region using modified parameters.

        Parameters
        ----------
        line : list
            The current streamline.
        prev_direction : np.ndarray
            The previous tracking direction.

        Returns
        -------
        line : list
            The extended streamline.
        prev_direction : np.ndarray
            The last direction.
        is_line_valid : bool
            Whether the line is valid.
        """
        # Switch to RAP parameters
        label = self._get_label(line[-1],
                                self.propagator.space,
                                self.propagator.origin)
        if label <= 0:
            return line, prev_direction, False

        # Logging debug when label changes        # Apply the parameters of the RAP labels
        if label != self._current_label:
            if self._current_label is not None:
                logging.debug(f"STEP[{self._total_steps}] label={self._current_label}"
                              f", algo={self.propagator.algo}"
                              f", theta (rad)={self.propagator.theta}"
                              f", vox step size={self.propagator.step_size}"
                              f" -> switching label to label {label}")
            self._current_label = label

        # Switch propagator based on label
        if str(label) in self._propagators:
            new_propagator = self._propagators[str(label)]
            if new_propagator is not self.propagator:
                new_propagator.line_rng_generator = self.propagator.line_rng_generator
                self.propagator = new_propagator
                logging.debug(f"RAP propagator switched to label {label}")
        else:
            new_propagator = self._propagators[self._propagators.keys()[0]]
            if new_propagator is not self.propagator:
                new_propagator.line_rng_generator = self.propagator.line_rng_generator
                self.propagator = new_propagator
                logging.debug(f"RAP propagator switched to default label {self._propagators.keys()[0]}")

        # Perform propagation with new parameters
        new_pos, new_dir, is_direction_valid = self.propagator.propagate(
            line, prev_direction)

        # Add the new point to the line
        if is_direction_valid:
            line.append(new_pos)
            self._total_steps += 1
            return line, new_dir, True
        return line, prev_direction, False

    def _get_label(self, curr_pos, space, origin):
        """
        Receive label (int) at current position in RAP label volume.

        Parameters
        ----------
        curr_pos: np.ndarray
            This is the current 3D position of the streamline.

        space: Space
            Coordinate space (here Space.VOX.).

        origin: Origin
            Origin convention ('center').

        Returns
        -------
        int
            The integer label at current position.
        """
        v = self.rap_volume.get_value_at_coordinate(
            *curr_pos, space=space, origin=origin)
        try:
            return int(v)
        except Exception:
            return int(np.round(v))


class RAPHybrid(RAPSwitch):

    # This class could probably be fused with RAPSwitch eventuallt

    def __init__(self, rap_volume, propagators: dict):

        # Graph entries are stored as strings in the propagators dict
        # (ex. "graph.npz") while ODF propagators are stored as 
        # ODFPropagator objects
        graph_entries = {k: v for k, v in propagators.items()
                     if isinstance(v, str)}
        odf_propagators = {k: v for k, v in propagators.items()
                     if k not in graph_entries}    
        
        super().__init__(rap_volume, odf_propagators)

        # Right now we assume that only one graph is provided.
        # Multiple regions can have a graph label, as long as all 
        # regions are associated to the same graph file.

        self.graph_labels = {int(k) for k in graph_entries}
        graph_file = next(iter(graph_entries.values()))

        self.csr, self.node_to_voxel, self.voxel_to_node, self.nodes_mask, _, self.peak_dirs = load_graph_sparse(graph_file)
        self.exit_mask = self._get_exit_mask()

    def rap_multistep_propagate(self, line, prev_direction):
        label = self._get_label(line[-1], self.propagator.space,
                                self.propagator.origin)
        if label in self.graph_labels:
            return self._graph_step(line, prev_direction)
        return super().rap_multistep_propagate(line, prev_direction)

    def _graph_step(self, line, prev_direction):
        pos = line[-1]
        pos_round = np.round(pos).astype(int)

        peak_dirs_at_pos = self.peak_dirs[tuple(pos_round)]

        # Get most likely node in the voxel depending on where we come from
        max_dot = 0
        max_idx = 0
        for i, peak_dir in enumerate(peak_dirs_at_pos):
            dir_norm = np.linalg.norm(peak_dir, axis=-1, keepdims=True)
            if dir_norm < 1e-6:
                continue
            peak_dir = peak_dir / dir_norm
            dot = np.dot(peak_dir, prev_direction)
            if abs(dot) > max_dot:
                max_dot = abs(dot)
                max_idx = i

        start_node = self.voxel_to_node[pos_round[0], pos_round[1], pos_round[2], max_idx]
        
        # TODO Find another way to choose the path
        # This is temporary
        #
        # We take the 10 most likely paths starting from 
        # start_node. We filter out the "impossible" paths and 
        # randomly select one of the "possible" paths
        paths = solve_dijkstra_multiple_paths(self.csr,
                                              self.voxel_to_node, self.nodes_mask, 
                                              node_to_voxel=self.node_to_voxel, 
                                              start_node=start_node, 
                                              exit_mask=self.exit_mask,
                                              nb_paths=10)       
        candidate_paths = []
        candidate_endpoints = []
        for path in paths:
            if path is None: 
                continue
            path_voxel_lobe = [self.node_to_voxel[p] for p in path]
            path_voxel = [p[:3] for p in path_voxel_lobe]
            if path_is_valid(path_voxel):
                candidate_paths.append(np.array(path_voxel, dtype=np.float32))
                candidate_endpoints.append(path_voxel_lobe[-1])

        if len(candidate_paths) == 0:
            return line, prev_direction, False

        # Randomly select a path and it's corresponding endpoint
        chosen_idx = random.randrange(len(candidate_paths))
        chosen_path = candidate_paths[chosen_idx]
        chosen_endpoint = candidate_endpoints[chosen_idx]

        endpoint_dir = self.peak_dirs[tuple(chosen_endpoint)]
        endpoint_dir = endpoint_dir / np.linalg.norm(endpoint_dir, axis=-1, keepdims=True)
        seg = chosen_path[-1] - chosen_path[-2]

        if np.dot(endpoint_dir, seg) < 0:
            endpoint_dir = -endpoint_dir

        # Endpoint directions needs to be of type TrackingDirection
        ind = self.propagator.sphere.find_closest(endpoint_dir)
        new_dir = TrackingDirection(self.propagator.sphere.vertices[ind], ind)
        
        line.extend(chosen_path)
        return line, new_dir, True

    def _get_exit_mask(self):
        footprint = np.ones((5, 5, 5), dtype=bool)
        expanded_mask = binary_dilation(self.nodes_mask, structure=footprint)
        exit_mask = expanded_mask & ~self.nodes_mask
        return exit_mask