"""
Heterogeneous graph construction for the BlueROV2 Physics-Informed GNN.

Graph topology (from the Technical Feedback Response):
    Node types:  hull (1), thruster (8), hydrodynamic (1), buoyancy (1)
    Edge types:  thruster→hull, hydrodynamic→hull, buoyancy→hull

All edges are directed, reflecting the causal direction of physical influence:
only thrusters introduce external forces; environmental forces act on the hull.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData

from src.parameters import (
    m, X_ud, Y_vd, Z_wd, I_zz, N_rd,
    X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc,
    N_r, N_rc, g, F_bouy, THRUSTER_CONFIG,
)

NUM_THRUSTERS = 8


# ---------------------------------------------------------------------------
# Thruster allocation: map 4-DOF control τ = [X, Y, Z, M_z] → 8 thrusts
# ---------------------------------------------------------------------------
def _build_allocation_matrix():
    """
    Build the (4 × 8) thruster allocation matrix B such that τ = B · f,
    where f ∈ ℝ⁸ are individual thruster forces.

    For the inverse mapping (given τ, recover f) we use the pseudoinverse.
    """
    B = np.zeros((4, NUM_THRUSTERS), dtype=np.float32)
    for i in range(NUM_THRUSTERS):
        pos  = np.array(THRUSTER_CONFIG[i]["position"],    dtype=np.float32)
        odir = np.array(THRUSTER_CONFIG[i]["orientation"],  dtype=np.float32)
        # Force contribution to [X, Y, Z]
        B[0, i] = odir[0]
        B[1, i] = odir[1]
        B[2, i] = odir[2]
        # Moment about z  (M_z = x·Fy − y·Fx)
        B[3, i] = pos[0] * odir[1] - pos[1] * odir[0]
    return B


_B_alloc = _build_allocation_matrix()
_B_pinv  = np.linalg.pinv(_B_alloc).astype(np.float32)   # (8 × 4)


def allocate_thrusts(tau: torch.Tensor) -> torch.Tensor:
    """
    Map generalised forces τ (B, 4) → individual thruster forces f (B, 8).

    Uses the Moore-Penrose pseudoinverse of the allocation matrix.
    """
    B_pinv = torch.tensor(_B_pinv, dtype=tau.dtype, device=tau.device)
    return tau @ B_pinv.T                                   # (B, 8)


# ---------------------------------------------------------------------------
# Static edge features (computed once, shared across all samples)
# ---------------------------------------------------------------------------
def _static_thruster_edge_features():
    """
    For each thruster→hull edge, return:
        [thrust_dir_x, thrust_dir_y, thrust_dir_z,
         lever_arm_x,  lever_arm_y,  lever_arm_z,
         efficiency]                                     → dim 7
    Efficiency is initialised to 1.0 (perfect).
    """
    feats = []
    for i in range(NUM_THRUSTERS):
        pos  = THRUSTER_CONFIG[i]["position"]
        odir = THRUSTER_CONFIG[i]["orientation"]
        feats.append(odir + pos + [1.0])
    return torch.tensor(feats, dtype=torch.float32)          # (8, 7)


def _static_hydro_edge_features():
    """
    Hydrodynamic→hull edge features:
        [drag_coeff × 4, added_mass_coupling × 4]        → dim 8
    Drag coefficients = [X_u, Y_v, Z_w, N_r] (linear part).
    Added-mass coupling = [X_ud, Y_vd, Z_wd, N_rd].
    """
    return torch.tensor(
        [[X_u, Y_v, Z_w, N_r, X_ud, Y_vd, Z_wd, N_rd]],
        dtype=torch.float32,
    )                                                        # (1, 8)


def _static_buoyancy_edge_features():
    """
    Buoyancy→hull edge features:
        [buoyancy_force_magnitude, moment_arm_x, moment_arm_y, moment_arm_z]
                                                           → dim 4
    """
    return torch.tensor(
        [[F_bouy, 0.0, 0.0, -0.1]],
        dtype=torch.float32,
    )                                                        # (1, 4)


# Pre-compute once at import time
_THRUSTER_EDGE_STATIC = _static_thruster_edge_features()
_HYDRO_EDGE_STATIC    = _static_hydro_edge_features()
_BUOY_EDGE_STATIC     = _static_buoyancy_edge_features()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(
    hull_state: torch.Tensor,
    tau: torch.Tensor,
    device: torch.device = None,
) -> HeteroData:
    """
    Build a single heterogeneous graph for one time-step.

    Parameters
    ----------
    hull_state : (9,)  — [x, y, z, cos(ψ), sin(ψ), u, v, w, r]
    tau        : (4,)  — [X, Y, Z, M_z]
    device     : target device (inferred from hull_state if None)

    Returns
    -------
    HeteroData with node / edge features as specified in the technical
    feedback document.
    """
    if device is None:
        device = hull_state.device

    data = HeteroData()

    # --- Node features ---------------------------------------------------
    data["hull"].x = hull_state.unsqueeze(0).to(device)       # (1, 9)

    # Thrusters (8 nodes, dim 8 each)
    f_individual = allocate_thrusts(tau.unsqueeze(0)).squeeze(0)  # (8,)
    thruster_feats = []
    for i in range(NUM_THRUSTERS):
        pos  = torch.tensor(THRUSTER_CONFIG[i]["position"],    dtype=torch.float32, device=device)
        odir = torch.tensor(THRUSTER_CONFIG[i]["orientation"],  dtype=torch.float32, device=device)
        cmd  = f_individual[i].unsqueeze(0)                     # thrust command
        act  = cmd                                              # steady-state model
        feat = torch.cat([cmd, act, pos, odir])                 # (8,)
        thruster_feats.append(feat)
    data["thruster"].x = torch.stack(thruster_feats)          # (8, 8)

    # Hydrodynamic (1 node, dim 4)
    data["hydrodynamic"].x = torch.tensor(
        [[X_u + X_uc, Y_v + Y_vc, Z_w + Z_wc, N_r + N_rc]],
        dtype=torch.float32, device=device,
    )                                                         # (1, 4)

    # Buoyancy (1 node, dim 6)
    data["buoyancy"].x = torch.tensor(
        [[F_bouy, 0.0, 0.0, 0.0, 0.0, -0.1]],
        dtype=torch.float32, device=device,
    )                                                         # (1, 6)

    # --- Edge indices & features -----------------------------------------
    src = torch.arange(NUM_THRUSTERS, dtype=torch.long, device=device)
    dst = torch.zeros(NUM_THRUSTERS, dtype=torch.long, device=device)
    data["thruster", "forces", "hull"].edge_index = torch.stack([src, dst])

    # Dynamic part: actual thrust vector + lever arm + efficiency
    thrust_vecs = []
    for i in range(NUM_THRUSTERS):
        odir = torch.tensor(THRUSTER_CONFIG[i]["orientation"], dtype=torch.float32, device=device)
        pos  = torch.tensor(THRUSTER_CONFIG[i]["position"],    dtype=torch.float32, device=device)
        fi   = f_individual[i]
        tv   = fi * odir                                      # thrust vector (3,)
        eff  = torch.tensor([1.0], device=device)
        thrust_vecs.append(torch.cat([tv, pos, eff]))          # (7,)
    data["thruster", "forces", "hull"].edge_attr = torch.stack(thrust_vecs)

    # Hydrodynamic → Hull  (1 edge)
    data["hydrodynamic", "drag", "hull"].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long, device=device,
    )
    data["hydrodynamic", "drag", "hull"].edge_attr = _HYDRO_EDGE_STATIC.clone().to(device)

    # Buoyancy → Hull  (1 edge)
    data["buoyancy", "restoring", "hull"].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long, device=device,
    )
    data["buoyancy", "restoring", "hull"].edge_attr = _BUOY_EDGE_STATIC.clone().to(device)

    return data


# ---------------------------------------------------------------------------
# Batched builder (for training efficiency)
# ---------------------------------------------------------------------------
def build_graph_batch(
    hull_states: torch.Tensor,
    taus: torch.Tensor,
) -> "list[HeteroData]":
    """
    Build a list of HeteroData graphs for a batch.

    Parameters
    ----------
    hull_states : (B, 9)
    taus        : (B, 4)

    Returns
    -------
    List of HeteroData, one per sample.
    """
    B = hull_states.shape[0]
    graphs = []
    for b in range(B):
        graphs.append(build_graph(hull_states[b], taus[b]))
    return graphs
