#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected DEM contact-pair simulation.

Purpose
-------
This script is a correctness-oriented revision of the uploaded
`NN4DEM_contact_Pair(1).py`.

It computes:
1. Particle-particle normal contact forces using a linear spring-dashpot model;
2. Particle-wall normal contact forces on six rigid walls;
3. Gravity;
4. Semi-implicit Euler time integration;
5. Per-particle force components (particle contact, wall, gravity, total).

Important scope
---------------
This is a *frictionless, non-rotational* soft-sphere DEM model.
The original uploaded script defined `friction_coefficient`, but it did not
actually compute tangential forces or rotational motion. This revision removes
that misleading unused parameter rather than silently introducing a different
physical model.

If a frictional/rotational DEM is required, the contact law must be extended
with tangential force history, angular velocity, torque, and Coulomb limiting.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =============================================================================
# 0. Reproducibility and device
# =============================================================================
SEED = 20260520
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

print("Using device:", DEVICE)


# =============================================================================
# 1. Physical and numerical parameters
# =============================================================================
# The uploaded script used `cell_size = 0.05` as the particle radius.
# We retain the same physical meaning here.
particle_radius = 0.05
particle_diameter = 2.0 * particle_radius

rho_p = 2700.0
particle_mass = (4.0 / 3.0) * math.pi * particle_radius**3 * rho_p

kn = 60000.0
restitution_coefficient = 0.5

gravity = 9.8

# The uploaded code used:
#   domain_length = domain_size * particle_radius
# We keep that convention for a like-for-like physical domain.
domain_size_x = 50
domain_size_y = 50
domain_size_z = 50

domain_length_x = domain_size_x * particle_radius
domain_length_y = domain_size_y * particle_radius
domain_length_z = domain_size_z * particle_radius
DOMAIN_LENGTHS = (domain_length_x, domain_length_y, domain_length_z)

# Contact-pair recording grid, following the CP formulation in the paper:
# particle_radius / cell_size is taken inside the stated [0.8, 0.85] range.
# The midpoint 0.825 is used here, so the 5x5x5 recording stencil and the
# displacement-triggered rebuild threshold match the paper's local CP logic.
CP_RADIUS_TO_CELL_RATIO = 0.825
neighbor_cell_width = particle_radius / CP_RADIUS_TO_CELL_RATIO
pair_rebuild_displacement_threshold = 2.0 * (
    neighbor_cell_width - particle_radius
)

# Time integration
dt = 0.001
simulation_time = 1.0
ntime = int(round(simulation_time / dt))

# Output controls
PRINT_INTERVAL = 100
SAVE_FIGURES = True
FIGURE_INTERVAL = 1000
FIGURE_DIR = "3D_new_corrected"

SAVE_NUMPY = False
NUMPY_INTERVAL = 1000
NUMPY_DIR = "DEM_contact_pair_corrected_npz"

RUN_SANITY_CHECKS = True


# =============================================================================
# 2. Derived damping coefficients and contact-time estimate
# =============================================================================
def compute_damping_coefficients(
    stiffness: float,
    mass: float,
    restitution: float,
) -> Tuple[float, float]:
    """
    Linear spring-dashpot damping coefficients used in the original script.

    For equal-mass particle-particle contacts:
        m_eff = m / 2

    For particle-wall contacts:
        m_eff = m
    """
    if not (0.0 < restitution <= 1.0):
        raise ValueError("restitution_coefficient must satisfy 0 < e <= 1.")

    alpha = -math.log(restitution) / math.pi
    gamma_ratio = alpha / math.sqrt(alpha * alpha + 1.0)

    eta_pp = 2.0 * gamma_ratio * math.sqrt(stiffness * mass / 2.0)
    eta_wall = 2.0 * gamma_ratio * math.sqrt(stiffness * mass)
    return eta_pp, eta_wall


def estimate_binary_contact_time(
    stiffness: float,
    damping: float,
    effective_mass: float,
) -> float:
    """
    Estimate the damped contact half-period for a linear spring-dashpot pair.
    """
    omega0_sq = stiffness / effective_mass
    damping_term_sq = (damping / (2.0 * effective_mass)) ** 2
    omega_d_sq = max(omega0_sq - damping_term_sq, 1.0e-30)
    omega_d = math.sqrt(omega_d_sq)
    return math.pi / omega_d


damping_coefficient_eta, damping_coefficient_eta_wall = compute_damping_coefficients(
    kn, particle_mass, restitution_coefficient
)

binary_contact_time = estimate_binary_contact_time(
    kn, damping_coefficient_eta, particle_mass / 2.0
)

print(f"Particle mass                    : {particle_mass:.8e} kg")
print(f"Particle-particle damping eta    : {damping_coefficient_eta:.8e}")
print(f"Particle-wall damping eta_wall   : {damping_coefficient_eta_wall:.8e}")
print(f"Estimated binary contact time    : {binary_contact_time:.8e} s")
print(f"dt/contact_time                  : {dt / binary_contact_time:.6f}")
print(f"CP recording cell size           : {neighbor_cell_width:.8e} m")
print(f"CP radius/cell-size ratio         : {CP_RADIUS_TO_CELL_RATIO:.6f}")
print(f"CP rebuild displacement threshold : {pair_rebuild_displacement_threshold:.8e} m")


# =============================================================================
# 3. Initial particle field
# =============================================================================
def create_initial_particle_lattice(
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reproduce the initial lattice and initial velocities from the uploaded code.

    Original index ranges:
        i, j, k = 2, ..., half_domain_size - 3
    For domain_size = 50:
        half_domain_size = 26
        range(2, 24) -> 22 points per direction -> 10648 particles
    """
    half_domain_size_x = int(domain_size_x / 2) + 1
    half_domain_size_y = int(domain_size_y / 2) + 1
    half_domain_size_z = int(domain_size_z / 2) + 1

    coords = []
    for i in range(2, half_domain_size_x - 2):
        for j in range(2, half_domain_size_y - 2):
            for k in range(2, half_domain_size_z - 2):
                coords.append(
                    [
                        i * particle_diameter,
                        j * particle_diameter,
                        k * particle_diameter,
                    ]
                )

    positions = torch.tensor(coords, dtype=dtype, device=device)

    velocities = torch.zeros_like(positions)
    # Uploaded script:
    #   vx = 0
    #   vy = random.uniform(-1,1) * 0.1
    #   vz = 0.001
    velocities[:, 0] = 0.0
    velocities[:, 1] = 0.2 * torch.rand(
        positions.shape[0], dtype=dtype, device=device
    ) - 0.1
    velocities[:, 2] = 0.001

    return positions, velocities


# =============================================================================
# 4. Contact-pair candidate recording: 5x5x5 stencil
# =============================================================================
def _linear_cell_id(
    cell_xyz: torch.Tensor,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """Map integer (cx, cy, cz) to a linear cell id."""
    return cell_xyz[:, 0] + nx * (cell_xyz[:, 1] + ny * cell_xyz[:, 2])


def _shift_grid_no_wrap(
    grid: torch.Tensor,
    dz: int,
    dy: int,
    dx: int,
    fill_value: int = -1,
) -> torch.Tensor:
    """
    Shift a 3-D integer grid without periodic wrap-around.

    This preserves the corrected non-periodic wall-domain treatment while
    applying the paper-style CP recording stencil. Cells shifted outside the
    domain are filled with `fill_value`.
    """
    out = torch.full_like(grid, fill_value)
    nz, ny, nx = grid.shape

    z_src_start = max(0, -dz)
    z_src_end = nz - max(0, dz)
    y_src_start = max(0, -dy)
    y_src_end = ny - max(0, dy)
    x_src_start = max(0, -dx)
    x_src_end = nx - max(0, dx)

    z_dst_start = max(0, dz)
    z_dst_end = nz - max(0, -dz)
    y_dst_start = max(0, dy)
    y_dst_end = ny - max(0, -dy)
    x_dst_start = max(0, dx)
    x_dst_end = nx - max(0, -dx)

    if (
        z_src_end > z_src_start
        and y_src_end > y_src_start
        and x_src_end > x_src_start
    ):
        out[
            z_dst_start:z_dst_end,
            y_dst_start:y_dst_end,
            x_dst_start:x_dst_end,
        ] = grid[
            z_src_start:z_src_end,
            y_src_start:y_src_end,
            x_src_start:x_src_end,
        ]

    return out


def _build_candidate_pairs_unique_cell_fast_path(
    cells: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast 5x5x5 candidate-pair recording when every recording cell contains at
    most one particle.

    This is the paper-style local CP recording stage:
    - particle ids are stored in a particle-number grid;
    - candidate pairs are recorded from a 5x5x5 neighbourhood;
    - each unordered pair is retained once only;
    - there is no periodic wrap-around across physical boundaries.
    """
    device = cells.device
    n_particles = cells.shape[0]

    particle_ids = torch.arange(n_particles, dtype=torch.long, device=device)
    particle_id_grid = torch.full(
        (nz, ny, nx), -1, dtype=torch.long, device=device
    )
    particle_id_grid[cells[:, 2], cells[:, 1], cells[:, 0]] = particle_ids

    pair_i_chunks = []
    pair_j_chunks = []

    for dz in range(-2, 3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                shifted_grid = _shift_grid_no_wrap(
                    particle_id_grid, dz=dz, dy=dy, dx=dx, fill_value=-1
                )
                valid = (
                    (particle_id_grid >= 0)
                    & (shifted_grid >= 0)
                    & (particle_id_grid < shifted_grid)
                )
                pair_i_chunks.append(particle_id_grid[valid])
                pair_j_chunks.append(shifted_grid[valid])

    if not pair_i_chunks:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    pair_i = torch.cat(pair_i_chunks, dim=0)
    pair_j = torch.cat(pair_j_chunks, dim=0)
    return pair_i, pair_j


def _build_candidate_pairs_multi_occupancy_fallback(
    positions: torch.Tensor,
    cells: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised sparse 5x5x5 CP recording path.

    This implements the same local candidate-pair stencil as the paper while
    avoiding a dense whole-domain tensor sweep and avoiding 125 Python-level
    search loops. It retains all particles when several particles occupy the
    same recording cell.
    """
    del positions  # The recording stage only needs integer cell coordinates.

    device = cells.device
    n_particles = cells.shape[0]
    if n_particles < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    cell_ids = _linear_cell_id(cells, nx, ny)
    sorted_cell_ids, sorted_particle_ids = torch.sort(cell_ids)

    # 5x5x5 candidate recording offsets, matching the paper's local mask.
    offset_range = torch.arange(-2, 3, dtype=torch.long, device=device)
    try:
        dz, dy, dx = torch.meshgrid(
            offset_range, offset_range, offset_range, indexing="ij"
        )
    except TypeError:
        # Compatibility fallback for older torch versions without `indexing=`.
        dz, dy, dx = torch.meshgrid(offset_range, offset_range, offset_range)

    offsets = torch.stack((dx, dy, dz), dim=-1).reshape(-1, 3)  # (125, 3)

    # Query all neighbouring recording cells for all particles in one batch.
    neighbour_cells = cells[:, None, :] + offsets[None, :, :]  # (N,125,3)
    valid = (
        (neighbour_cells[..., 0] >= 0)
        & (neighbour_cells[..., 0] < nx)
        & (neighbour_cells[..., 1] >= 0)
        & (neighbour_cells[..., 1] < ny)
        & (neighbour_cells[..., 2] >= 0)
        & (neighbour_cells[..., 2] < nz)
    )

    if not bool(valid.any().item()):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    flat_valid = valid.reshape(-1)
    neighbour_cells_valid = neighbour_cells.reshape(-1, 3)[flat_valid]

    source_ids_full = torch.arange(
        n_particles, dtype=torch.long, device=device
    ).repeat_interleave(offsets.shape[0])
    source_ids = source_ids_full[flat_valid]

    neighbour_cell_ids = _linear_cell_id(neighbour_cells_valid, nx, ny)
    left = torch.searchsorted(sorted_cell_ids, neighbour_cell_ids, right=False)
    right = torch.searchsorted(sorted_cell_ids, neighbour_cell_ids, right=True)
    counts = right - left

    non_empty = counts > 0
    if not bool(non_empty.any().item()):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    source_ids = source_ids[non_empty]
    left = left[non_empty]
    counts = counts[non_empty]

    source_repeated = torch.repeat_interleave(source_ids, counts)
    left_repeated = torch.repeat_interleave(left, counts)

    cumulative_counts = torch.cumsum(counts, dim=0)
    group_starts = torch.repeat_interleave(cumulative_counts - counts, counts)
    local_offsets = (
        torch.arange(source_repeated.shape[0], dtype=torch.long, device=device)
        - group_starts
    )

    neighbour_sorted_positions = left_repeated + local_offsets
    neighbour_particle_ids = sorted_particle_ids[neighbour_sorted_positions]

    # Retain each unordered pair once and remove self-pairs.
    unique_pair_mask = source_repeated < neighbour_particle_ids
    pair_i = source_repeated[unique_pair_mask]
    pair_j = neighbour_particle_ids[unique_pair_mask]
    return pair_i, pair_j


def build_candidate_pairs(
    positions: torch.Tensor,
    domain_lengths: Tuple[float, float, float],
    broadphase_cell_width: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Record candidate contact pairs with the paper-style 5x5x5 CP stencil.

    The returned pair lists are intended to be *stored and reused* by the
    time-stepper, rather than reconstructed every step. Rebuild scheduling is
    handled in `DEMContactPair.forward()` through the cumulative displacement
    criterion.
    """
    device = positions.device
    n_particles = positions.shape[0]

    if n_particles < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    lx, ly, lz = domain_lengths
    nx = max(1, int(math.ceil(lx / broadphase_cell_width)))
    ny = max(1, int(math.ceil(ly / broadphase_cell_width)))
    nz = max(1, int(math.ceil(lz / broadphase_cell_width)))

    broadphase_origin = 0.5 * broadphase_cell_width
    cells = torch.floor(
        (positions - broadphase_origin + 1.0e-5 * broadphase_cell_width)
        / broadphase_cell_width
    ).to(torch.long)
    cells[:, 0].clamp_(0, nx - 1)
    cells[:, 1].clamp_(0, ny - 1)
    cells[:, 2].clamp_(0, nz - 1)

    # Use the sparse 5x5x5 cell-neighbour recording path as the default.
    # It implements the same candidate-pair stencil while avoiding a full
    # dense-grid sweep over empty cells, which is substantially more robust for
    # CPU execution and for sparse particle sets.
    return _build_candidate_pairs_multi_occupancy_fallback(
        positions=positions,
        cells=cells,
        nx=nx,
        ny=ny,
        nz=nz,
    )


# =============================================================================
# 5. Force models
# =============================================================================
def compute_particle_contact_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    radius: float,
    stiffness: float,
    damping_eta: float,
    eps: float = 1.0e-12,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute particle-particle normal contact forces.

    Contact law on particle i for pair (i,j):
        F_ij = max(k_n * overlap - eta_n * v_n, 0) * n_ij

    where:
        n_ij points from particle j to particle i,
        overlap = 2R - |x_i - x_j|,
        v_n = (v_i - v_j) · n_ij.

    The max(...,0) clamp prevents a nonphysical attractive normal force during
    the unloading/separation part of a linear dashpot contact.
    """
    device = positions.device
    dtype = positions.dtype
    n_particles = positions.shape[0]

    force = torch.zeros_like(positions)

    if pair_i.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_scalar = torch.empty(0, dtype=dtype, device=device)
        empty_vec = torch.empty((0, 3), dtype=dtype, device=device)
        return force, empty_long, empty_long, empty_vec, empty_scalar

    displacement = positions[pair_i] - positions[pair_j]
    raw_distance = torch.linalg.norm(displacement, dim=1)
    overlap = 2.0 * radius - raw_distance

    in_contact = overlap > 0.0
    if not bool(in_contact.any().item()):
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_scalar = torch.empty(0, dtype=dtype, device=device)
        empty_vec = torch.empty((0, 3), dtype=dtype, device=device)
        return force, empty_long, empty_long, empty_vec, empty_scalar

    active_i = pair_i[in_contact]
    active_j = pair_j[in_contact]
    displacement = displacement[in_contact]
    raw_distance = raw_distance[in_contact]
    overlap = overlap[in_contact]

    safe_distance = raw_distance.clamp_min(eps)
    normal = displacement / safe_distance.unsqueeze(1)

    # Extremely rare degenerate case: exactly coincident centers.
    # The direction is mathematically undefined; choose a deterministic axis
    # rather than silently generating zero force.
    coincident = raw_distance <= eps
    if bool(coincident.any().item()):
        normal[coincident] = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)

    relative_velocity = velocities[active_i] - velocities[active_j]
    relative_normal_velocity = torch.sum(relative_velocity * normal, dim=1)

    normal_force_magnitude = (
        stiffness * overlap - damping_eta * relative_normal_velocity
    ).clamp_min(0.0)

    pair_force_on_i = normal_force_magnitude.unsqueeze(1) * normal

    # Newton's third law: +F on i, -F on j.
    force.index_add_(0, active_i, pair_force_on_i)
    force.index_add_(0, active_j, -pair_force_on_i)

    return force, active_i, active_j, pair_force_on_i, overlap


def compute_wall_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    domain_lengths: Tuple[float, float, float],
    radius: float,
    stiffness: float,
    damping_eta_wall: float,
) -> torch.Tensor:
    """
    Compute normal elastic+damping forces from six rigid non-periodic walls.

    Corrected wall geometry:
        lower wall contact: x < R
        upper wall contact: x > Lx - R
    and analogously for y/z.

    The uploaded script used L - 2R on the upper walls, which shortened the
    container by one radius on each positive side.
    """
    force = torch.zeros_like(positions)

    for axis, domain_length in enumerate(domain_lengths):
        # Lower wall: outward normal on particle = +axis.
        lower_overlap = radius - positions[:, axis]
        lower_active = lower_overlap > 0.0
        lower_vn = velocities[:, axis]
        lower_force_mag = (
            stiffness * lower_overlap - damping_eta_wall * lower_vn
        ).clamp_min(0.0)
        force[:, axis] += torch.where(
            lower_active, lower_force_mag, torch.zeros_like(lower_force_mag)
        )

        # Upper wall: outward normal on particle = -axis.
        upper_overlap = positions[:, axis] - (domain_length - radius)
        upper_active = upper_overlap > 0.0
        upper_vn = -velocities[:, axis]
        upper_force_mag = (
            stiffness * upper_overlap - damping_eta_wall * upper_vn
        ).clamp_min(0.0)
        force[:, axis] -= torch.where(
            upper_active, upper_force_mag, torch.zeros_like(upper_force_mag)
        )

    return force


# =============================================================================
# 6. DEM time-stepper
# =============================================================================
@dataclass
class ForceDiagnostics:
    particle_contact: torch.Tensor
    wall: torch.Tensor
    gravity: torch.Tensor
    total: torch.Tensor
    active_pair_i: torch.Tensor
    active_pair_j: torch.Tensor
    pair_force_on_i: torch.Tensor
    overlap: torch.Tensor


class DEMContactPair(nn.Module):
    """
    Frictionless normal-contact DEM stepper with paper-style contact-pair (CP)
    recording, reuse and displacement-triggered reconstruction.

    Only the contact-pair neighbour-maintenance logic is changed here. The
    contact-force kernel, wall-force kernel, gravity term and time integration
    remain the same as in the corrected baseline script.
    """

    def __init__(
        self,
        radius: float,
        mass: float,
        stiffness: float,
        damping_eta: float,
        damping_eta_wall: float,
        domain_lengths: Tuple[float, float, float],
        broadphase_cell_width: float,
        rebuild_displacement_threshold: float,
        gravitational_acceleration: float,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.mass = mass
        self.stiffness = stiffness
        self.damping_eta = damping_eta
        self.damping_eta_wall = damping_eta_wall
        self.domain_lengths = domain_lengths
        self.broadphase_cell_width = broadphase_cell_width
        self.rebuild_displacement_threshold = rebuild_displacement_threshold
        self.gravitational_acceleration = gravitational_acceleration

        # Persistent CP storage: pair lists A and B from the paper.
        self.register_buffer(
            "recorded_pair_i",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "recorded_pair_j",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

        # Paper-style adaptive rebuild controller:
        #   Δx <- Δx + ||v||_max Δt
        # pair reconstruction is triggered before the next force evaluation
        # when Δx >= Δx_max.
        self.register_buffer(
            "cumulative_pair_displacement",
            torch.zeros((), dtype=DTYPE),
            persistent=False,
        )
        self.register_buffer(
            "pair_list_valid",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )

    def _rebuild_recorded_pairs(self, positions: torch.Tensor) -> None:
        pair_i, pair_j = build_candidate_pairs(
            positions,
            self.domain_lengths,
            self.broadphase_cell_width,
        )
        self._buffers["recorded_pair_i"] = pair_i
        self._buffers["recorded_pair_j"] = pair_j
        self.cumulative_pair_displacement.zero_()
        self.pair_list_valid.fill_(True)

    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, ForceDiagnostics]:
        need_rebuild = (
            (not bool(self.pair_list_valid.item()))
            or bool(
                (
                    self.cumulative_pair_displacement
                    >= self.rebuild_displacement_threshold
                ).item()
            )
        )

        if need_rebuild:
            self._rebuild_recorded_pairs(positions)

        # Reuse the stored CP lists at every force-evaluation step until the
        # cumulative displacement criterion requests reconstruction.
        pair_i = self.recorded_pair_i
        pair_j = self.recorded_pair_j

        particle_contact_force, active_i, active_j, pair_force_on_i, overlap = (
            compute_particle_contact_forces(
                positions=positions,
                velocities=velocities,
                pair_i=pair_i,
                pair_j=pair_j,
                radius=self.radius,
                stiffness=self.stiffness,
                damping_eta=self.damping_eta,
            )
        )

        wall_force = compute_wall_forces(
            positions=positions,
            velocities=velocities,
            domain_lengths=self.domain_lengths,
            radius=self.radius,
            stiffness=self.stiffness,
            damping_eta_wall=self.damping_eta_wall,
        )

        gravity_force = torch.zeros_like(positions)
        gravity_force[:, 2] = -self.mass * self.gravitational_acceleration

        total_force = particle_contact_force + wall_force + gravity_force

        # Semi-implicit Euler, unchanged from the corrected baseline.
        new_velocities = velocities + (dt / self.mass) * total_force
        new_positions = positions + dt * new_velocities

        # Article-style cumulative displacement update:
        #   Δx <- Δx + ||v||_max Δt
        # The threshold test is applied at the start of the next local step.
        if new_velocities.numel() > 0:
            max_speed = torch.linalg.norm(new_velocities, dim=1).max()
            self.cumulative_pair_displacement.add_(max_speed * dt)

        diagnostics = ForceDiagnostics(
            particle_contact=particle_contact_force,
            wall=wall_force,
            gravity=gravity_force,
            total=total_force,
            active_pair_i=active_i,
            active_pair_j=active_j,
            pair_force_on_i=pair_force_on_i,
            overlap=overlap,
        )
        return new_positions, new_velocities, diagnostics



# =============================================================================
# 7. Built-in sanity checks
# =============================================================================
def run_sanity_checks(model: DEMContactPair) -> None:
    """
    Lightweight correctness checks that target the failure modes in the uploaded code:
    1. Contact action-reaction balance;
    2. No false contact between opposite domain boundaries;
    3. Wall force directions.
    """
    print("\nRunning sanity checks...")

    # 1. Two overlapping particles: internal contact forces sum to zero.
    p = torch.tensor(
        [[0.50, 0.50, 0.50], [0.59, 0.50, 0.50]],
        dtype=DTYPE,
        device=DEVICE,
    )
    v = torch.zeros_like(p)
    pair_i, pair_j = build_candidate_pairs(p, DOMAIN_LENGTHS, neighbor_cell_width)
    f_contact, *_ = compute_particle_contact_forces(
        positions=p,
        velocities=v,
        pair_i=pair_i,
        pair_j=pair_j,
        radius=particle_radius,
        stiffness=kn,
        damping_eta=damping_coefficient_eta,
    )
    force_balance_error = torch.linalg.norm(f_contact.sum(dim=0)).item()
    if force_balance_error > 1.0e-4:
        raise RuntimeError(
            f"Sanity check failed: particle contact forces do not balance. "
            f"Residual = {force_balance_error:.6e}"
        )

    # 2. No false cross-boundary contact.
    p = torch.tensor(
        [
            [particle_radius, 0.50, 0.50],
            [domain_length_x - particle_radius, 0.50, 0.50],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    v = torch.zeros_like(p)
    pair_i, pair_j = build_candidate_pairs(p, DOMAIN_LENGTHS, neighbor_cell_width)
    f_contact, active_i, active_j, *_ = compute_particle_contact_forces(
        positions=p,
        velocities=v,
        pair_i=pair_i,
        pair_j=pair_j,
        radius=particle_radius,
        stiffness=kn,
        damping_eta=damping_coefficient_eta,
    )
    if active_i.numel() != 0 or torch.linalg.norm(f_contact).item() != 0.0:
        raise RuntimeError("Sanity check failed: false periodic cross-boundary contact detected.")

    # 3. Wall force directions.
    p = torch.tensor(
        [
            [0.50 * particle_radius, 0.50, 0.50],
            [domain_length_x - 0.50 * particle_radius, 0.50, 0.50],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    v = torch.zeros_like(p)
    f_wall = compute_wall_forces(
        positions=p,
        velocities=v,
        domain_lengths=DOMAIN_LENGTHS,
        radius=particle_radius,
        stiffness=kn,
        damping_eta_wall=damping_coefficient_eta_wall,
    )
    if not (f_wall[0, 0] > 0.0 and f_wall[1, 0] < 0.0):
        raise RuntimeError("Sanity check failed: wall force directions are incorrect.")

    print("Sanity checks passed.\n")


# =============================================================================
# 8. Output utilities
# =============================================================================
def save_snapshot_figure(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    step: int,
) -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)

    xyz = positions.detach().cpu().numpy()
    vz = velocities[:, 2].detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=vz, s=4)
    cbar = plt.colorbar(sc, orientation="horizontal", shrink=0.45, pad=0.08)
    cbar.set_label(r"$v_z$")

    ax.set_xlim([0.0, domain_length_x])
    ax.set_ylim([0.0, domain_length_y])
    ax.set_zlim([0.0, domain_length_z])
    ax.view_init(elev=45, azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Corrected DEM contact-pair simulation, step = {step}")

    save_path = os.path.join(FIGURE_DIR, f"{step:08d}.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_snapshot_npz(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    diagnostics: ForceDiagnostics,
    step: int,
) -> None:
    os.makedirs(NUMPY_DIR, exist_ok=True)
    save_path = os.path.join(NUMPY_DIR, f"state_{step:08d}.npz")

    np.savez_compressed(
        save_path,
        positions=positions.detach().cpu().numpy(),
        velocities=velocities.detach().cpu().numpy(),
        force_particle_contact=diagnostics.particle_contact.detach().cpu().numpy(),
        force_wall=diagnostics.wall.detach().cpu().numpy(),
        force_gravity=diagnostics.gravity.detach().cpu().numpy(),
        force_total=diagnostics.total.detach().cpu().numpy(),
        active_pair_i=diagnostics.active_pair_i.detach().cpu().numpy(),
        active_pair_j=diagnostics.active_pair_j.detach().cpu().numpy(),
        pair_force_on_i=diagnostics.pair_force_on_i.detach().cpu().numpy(),
        overlap=diagnostics.overlap.detach().cpu().numpy(),
    )


# =============================================================================
# 9. Main simulation
# =============================================================================
def run_simulation() -> None:
    positions, velocities = create_initial_particle_lattice(
        device=DEVICE,
        dtype=DTYPE,
    )

    print(f"Number of particles               : {positions.shape[0]}")
    print(f"Domain lengths                    : {DOMAIN_LENGTHS}")
    print(f"Total number of time steps        : {ntime}")

    model = DEMContactPair(
        radius=particle_radius,
        mass=particle_mass,
        stiffness=kn,
        damping_eta=damping_coefficient_eta,
        damping_eta_wall=damping_coefficient_eta_wall,
        domain_lengths=DOMAIN_LENGTHS,
        broadphase_cell_width=neighbor_cell_width,
        rebuild_displacement_threshold=pair_rebuild_displacement_threshold,
        gravitational_acceleration=gravity,
    ).to(DEVICE)

    if RUN_SANITY_CHECKS:
        run_sanity_checks(model)

    last_diagnostics: ForceDiagnostics | None = None

    start = time.time()
    with torch.no_grad():
        for itime in range(1, ntime + 1):
            positions, velocities, diagnostics = model(positions, velocities, dt)
            last_diagnostics = diagnostics

            if itime % PRINT_INTERVAL == 0 or itime == 1:
                n_contacts = diagnostics.active_pair_i.numel()
                max_overlap = (
                    diagnostics.overlap.max().item()
                    if diagnostics.overlap.numel() > 0
                    else 0.0
                )
                max_total_force = torch.linalg.norm(diagnostics.total, dim=1).max().item()
                print(
                    f"Step {itime:7d}/{ntime:7d} | "
                    f"contacts = {n_contacts:8d} | "
                    f"max overlap = {max_overlap:.6e} | "
                    f"max |F_total| = {max_total_force:.6e}"
                )

            if SAVE_FIGURES and itime % FIGURE_INTERVAL == 0:
                save_snapshot_figure(positions, velocities, itime)

            if SAVE_NUMPY and itime % NUMPY_INTERVAL == 0:
                save_snapshot_npz(positions, velocities, diagnostics, itime)

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.6f} s")

    # Always make the final force state explicit to the user/code.
    if last_diagnostics is not None:
        net_internal_contact_force = last_diagnostics.particle_contact.sum(dim=0)
        print(
            "Final net particle-contact force residual: "
            f"{net_internal_contact_force.detach().cpu().numpy()}"
        )


if __name__ == "__main__":
    run_simulation()
