"""
Physics-Informed Heterogeneous Graph Neural Network (PIGNN) for BlueROV2.

Architecture
------------
* Heterogeneous message-passing on the BlueROV2 subsystem graph.
* Node types: hull, thruster, hydrodynamic, buoyancy.
* Edge types: thruster→hull, hydrodynamic→hull, buoyancy→hull.
* Message passing follows the physical causal chain:
      force sources → hull (acceleration → velocity → position).

The model predicts the *next state* x(t+dt) given the current state x(t)
and control input τ(t), using a residual formulation:

    x̂(t+dt) = x(t) + Δx̂

where Δx̂ is computed from the GNN's aggregated hull embedding.
Body-frame increments in x,y are rotated to world frame using the
predicted heading (cos ψ̂, sin ψ̂), matching the existing PINC approach.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HeteroConv, Linear as PyGLinear

from models.graph_builder import build_graph, allocate_thrusts, NUM_THRUSTERS
from src.parameters import THRUSTER_CONFIG


# ---------------------------------------------------------------------------
# Adaptive Softplus (matching the original PINC codebase)
# ---------------------------------------------------------------------------
class AdaptiveSoftplus(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.sp   = nn.Softplus()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return torch.reciprocal(self.beta) * self.sp(self.beta * x)


# ---------------------------------------------------------------------------
# Small MLP helper
# ---------------------------------------------------------------------------
def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
    layers = []
    layers.append(nn.Linear(in_dim, hidden))
    layers.append(AdaptiveSoftplus())
    layers.append(nn.LayerNorm(hidden))
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(AdaptiveSoftplus())
        layers.append(nn.LayerNorm(hidden))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Per-edge-type message functions
# ---------------------------------------------------------------------------
class ThrusterToHullConv(nn.Module):
    """
    Message from each thruster node to the hull.

    Input per edge: concat(thruster_feat, edge_attr, hull_feat)
    Output: message vector of dim `out_channels`.
    """

    def __init__(self, thruster_dim: int, edge_dim: int, hull_dim: int,
                 out_channels: int, hidden: int = 32):
        super().__init__()
        self.mlp = _mlp(thruster_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


class HydroToHullConv(nn.Module):
    """Message from the hydrodynamic node to the hull."""

    def __init__(self, hydro_dim: int, edge_dim: int, hull_dim: int,
                 out_channels: int, hidden: int = 32):
        super().__init__()
        self.mlp = _mlp(hydro_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


class BuoyToHullConv(nn.Module):
    """Message from the buoyancy node to the hull."""

    def __init__(self, buoy_dim: int, edge_dim: int, hull_dim: int,
                 out_channels: int, hidden: int = 32):
        super().__init__()
        self.mlp = _mlp(buoy_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


# ---------------------------------------------------------------------------
# Heterogeneous message-passing layer
# ---------------------------------------------------------------------------
class PIGNNLayer(nn.Module):
    """
    One round of heterogeneous message passing.

    All three edge types send messages to the hull node.  Messages from
    the 8 thrusters are *summed* (reflecting superposition of forces).
    """

    def __init__(self, node_dims: dict, edge_dims: dict,
                 hidden: int = 32, msg_dim: int = 32):
        super().__init__()

        self.thruster_conv = ThrusterToHullConv(
            node_dims["thruster"], edge_dims[("thruster", "forces", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )
        self.hydro_conv = HydroToHullConv(
            node_dims["hydrodynamic"], edge_dims[("hydrodynamic", "drag", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )
        self.buoy_conv = BuoyToHullConv(
            node_dims["buoyancy"], edge_dims[("buoyancy", "restoring", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )

        # Update function for hull node after aggregating messages
        self.hull_update = _mlp(node_dims["hull"] + 3 * msg_dim, hidden,
                                node_dims["hull"], n_layers=2)

    def forward(self, data: HeteroData) -> HeteroData:
        # Compute messages
        ei_t = data["thruster", "forces", "hull"].edge_index
        ea_t = data["thruster", "forces", "hull"].edge_attr
        msg_t = self.thruster_conv(
            data["thruster"].x, data["hull"].x, ea_t, ei_t,
        )
        # Aggregate thruster messages by summing per destination hull node
        num_hull = data["hull"].x.size(0)
        agg_t = torch.zeros(num_hull, msg_t.size(-1),
                            device=msg_t.device, dtype=msg_t.dtype)
        dst_t = ei_t[1]
        agg_t.scatter_add_(0, dst_t.unsqueeze(-1).expand_as(msg_t), msg_t)

        ei_h = data["hydrodynamic", "drag", "hull"].edge_index
        ea_h = data["hydrodynamic", "drag", "hull"].edge_attr
        msg_h = self.hydro_conv(
            data["hydrodynamic"].x, data["hull"].x, ea_h, ei_h,
        )
        # Single source → single hull; direct assignment
        agg_h = msg_h

        ei_b = data["buoyancy", "restoring", "hull"].edge_index
        ea_b = data["buoyancy", "restoring", "hull"].edge_attr
        msg_b = self.buoy_conv(
            data["buoyancy"].x, data["hull"].x, ea_b, ei_b,
        )
        agg_b = msg_b

        # Concatenate aggregated messages with current hull features
        hull_in = torch.cat([data["hull"].x, agg_t, agg_h, agg_b], dim=-1)
        data["hull"].x = self.hull_update(hull_in)

        return data


# ---------------------------------------------------------------------------
# Full PIGNN model
# ---------------------------------------------------------------------------
class PIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network for BlueROV2 dynamics.

    Input:  Z = [x(t), τ(t), dt]   — concatenated state + control + time
            Same interface as the original DNN for drop-in compatibility.
    Output: x̂(t+dt)                — predicted next state (dim 9)

    Internally the model:
        1. Builds the heterogeneous graph from state & control.
        2. Runs N_mp rounds of message passing.
        3. Reads out the hull embedding → predicts Δx̂.
        4. Applies the residual + body→world rotation for (x, y).
    """

    def __init__(
        self,
        N_in: int = 14,        # 9 state + 4 control + 1 time
        N_out: int = 9,         # next state dim
        hidden: int = 32,
        msg_dim: int = 32,
        n_mp_layers: int = 2,
    ):
        super().__init__()
        self.N_in  = N_in
        self.N_out = N_out

        # Node / edge feature dimensions (after encoding)
        hull_enc_dim = hidden
        self.node_dims = {
            "hull":          hull_enc_dim,
            "thruster":      8,
            "hydrodynamic":  4,
            "buoyancy":      6,
        }
        self.edge_dims = {
            ("thruster",     "forces",    "hull"): 7,
            ("hydrodynamic", "drag",      "hull"): 8,
            ("buoyancy",     "restoring", "hull"): 4,
        }

        # Hull node encoder: 9 (state) + 4 (control) + 1 (time) → hull_enc_dim
        self.hull_encoder = _mlp(N_in, hidden, hull_enc_dim)

        # Message-passing layers
        self.mp_layers = nn.ModuleList([
            PIGNNLayer(self.node_dims, self.edge_dims, hidden, msg_dim)
            for _ in range(n_mp_layers)
        ])

        # Readout: hull embedding → state increment Δx̂
        self.readout = _mlp(hull_enc_dim, hidden, N_out, n_layers=3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----- graph construction helpers (work on flattened Z) ---------------

    def _z_to_graph(self, z: torch.Tensor) -> HeteroData:
        """Convert a single flattened input vector z to a HeteroData graph."""
        state = z[:9]          # [x,y,z,cosψ,sinψ,u,v,w,r]
        tau   = z[9:13]        # [X,Y,Z,Mz]
        graph = build_graph(state.detach(), tau.detach())
        # Move all tensors to same device
        for store in graph.node_stores:
            store.x = store.x.to(z.device)
        for store in graph.edge_stores:
            store.edge_index = store.edge_index.to(z.device)
            store.edge_attr  = store.edge_attr.to(z.device)
        return graph

    # ----- forward --------------------------------------------------------

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        Z : (B, N_in)  or  (N_total, N_in)
            Flattened input: [state (9) | control (4) | time (1)].
            This matches the interface of the original DNN so all existing
            training / evaluation code works unchanged.

        Returns
        -------
        X_hat : (B, N_out) — predicted next state.
        """
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)

        B = Z.shape[0]
        device = Z.device

        # Build graphs and encode hull nodes
        # For efficiency we process all samples in a single batched graph.
        graphs = []
        for b in range(B):
            g = self._z_to_graph(Z[b])
            # Encode hull node: replace raw features with encoded ones
            g["hull"].x = self.hull_encoder(Z[b].unsqueeze(0))
            graphs.append(g)

        # Batch into a single HeteroData
        batch = Batch.from_data_list(graphs)

        # Message passing
        for mp_layer in self.mp_layers:
            batch = mp_layer(batch)

        # Readout — hull embeddings (one per graph in the batch)
        hull_emb = batch["hull"].x                            # (B, hull_enc_dim)
        delta    = self.readout(hull_emb)                     # (B, N_out)

        # Residual connection + body→world rotation (matching PINC approach)
        state_in = Z[:, :self.N_out]                          # (B, 9)

        cos_psi_hat = delta[:, 3] + state_in[:, 3]
        sin_psi_hat = delta[:, 4] + state_in[:, 4]

        # Rotate body-frame position increments to world frame
        x_world = cos_psi_hat * delta[:, 0] - sin_psi_hat * delta[:, 1] + state_in[:, 0]
        y_world = sin_psi_hat * delta[:, 0] + cos_psi_hat * delta[:, 1] + state_in[:, 1]

        X_hat = delta + state_in                              # residual
        X_hat = X_hat.clone()
        X_hat[:, 0] = x_world
        X_hat[:, 1] = y_world

        return X_hat
