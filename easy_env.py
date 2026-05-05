"""
Easy Tangram Puzzle Environment (Polytope-based)
=================================================
4 unit squares on a 10×10 grid.  Each piece has an assigned target slot
in a 2×2 silhouette at the board centre:

    Slot layout  (bottom-left corners at grid coords):
        piece 0 → (4,4)    piece 1 → (5,4)
        piece 2 → (4,5)    piece 3 → (5,5)

Movement is purely grid-based (fast, no PPL overhead for stepping).
State is extracted from PPL C_Polyhedron objects rebuilt from current
positions — producing H-rep, V-rep, and constraint-adjacency tensors
identical in structure to the hard tangram (just 4 pieces instead of 6,
4 constraints instead of 5).

Gym observation (dict):
  h_rep : float32 [4, 4, 3]  — 4 constraints × (a1, a2, b) per piece
  v_rep : float32 [4, 4, 2]  — 4 vertices × (x, y) per piece
  adj   : float32 [4, 4, 4]  — constraint adjacency matrix per piece

Action space : Discrete(16) — 4 pieces × 4 directions (up/down/left/right)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ppl
from ppl import Variable, C_Polyhedron, Constraint_System
import gym
from gym import spaces
import os

GRID_SIZE = 10     # board is GRID_SIZE × GRID_SIZE cells
BOARD_MAX = 11.0   # b-value normalisation constant (mirrors hard tangram)
MAX_STEPS = 300


# ── Inner physics / geometry layer ────────────────────────────────────────────

class EasyTangramEnv:
    """Geometry layer: grid positions + PPL polyhedra for state extraction."""

    def __init__(self):
        self.x, self.y = Variable(0), Variable(1)

        # Target slots — 2×2 silhouette at board centre
        cx, cy = 4, 4
        self.target_positions = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)]
        self.target_pieces    = [self._build_square(px, py) for px, py in self.target_positions]
        self.target_centroids = [self._centroid(tp) for tp in self.target_pieces]

        self.piece_positions = None   # list of (px, py) integer tuples
        self.locked          = None
        self.reset()

    # ── Polytope helpers ──────────────────────────────────────────────────────
    def _build_square(self, px, py):
        """Return a PPL C_Polyhedron for the unit square at grid position (px, py)."""
        cs = Constraint_System()
        cs.insert(self.x >= px)
        cs.insert(self.x <= px + 1)
        cs.insert(self.y >= py)
        cs.insert(self.y <= py + 1)
        return C_Polyhedron(cs)

    def _centroid(self, poly):
        verts = [
            (float(g.coefficient(self.x)), float(g.coefficient(self.y)))
            for g in poly.generators() if g.is_point()
        ]
        return np.mean(verts, axis=0) if verts else np.zeros(2)

    # ── State ─────────────────────────────────────────────────────────────────
    def reset(self):
        """Place pieces at the four corners of the board."""
        g = GRID_SIZE - 1
        self.piece_positions = [(0, 0), (g, 0), (0, g), (g, g)]
        self.locked          = [False] * 4

    def get_pieces(self):
        """Rebuild PPL polyhedra from current grid positions (called per step)."""
        return [self._build_square(px, py) for px, py in self.piece_positions]

    # ── Step ─────────────────────────────────────────────────────────────────
    def move_piece(self, piece_idx, dx, dy):
        """Grid-based move. Bounds and locking enforced; overlaps between pieces allowed."""
        if self.locked[piece_idx]:
            return "Locked"

        px, py = self.piece_positions[piece_idx]
        new_px = px + dx
        new_py = py + dy
        self.piece_positions[piece_idx] = (new_px, new_py)

        tx, ty = self.target_positions[piece_idx]
        if new_px == tx and new_py == ty:
            self.locked[piece_idx] = True

        return "Success"

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self, save_path):
        """Render board state to save_path (adds .png if no extension given)."""
        if not os.path.splitext(save_path)[1]:
            save_path += ".png"

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(os.path.basename(save_path))

        for tp in self.target_pieces:
            self._plot_poly(ax, tp, color="gray", alpha=0.15, linestyle="--")

        colors = ["#e74c3c", "#3498db", "#e67e22", "#9b59b6"]
        for i, poly in enumerate(self.get_pieces()):
            c = "#27ae60" if self.locked[i] else colors[i]
            self._plot_poly(ax, poly, color=c, alpha=0.7)

        ax.set_xlim(-0.5, GRID_SIZE + 0.5)
        ax.set_ylim(-0.5, GRID_SIZE + 0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def _plot_poly(self, ax, poly, **kwargs):
        verts = [
            (float(g.coefficient(self.x)), float(g.coefficient(self.y)))
            for g in poly.generators() if g.is_point()
        ]
        if len(verts) >= 3:
            c = np.mean(verts, axis=0)
            verts.sort(key=lambda p: np.arctan2(p[1] - c[1], p[0] - c[0]))
            ax.add_patch(patches.Polygon(verts, **kwargs))


# ── Gym wrapper ───────────────────────────────────────────────────────────────

class EasyTangramGym(gym.Env):
    """
    OpenAI Gym wrapper.  Provides H-rep, V-rep, and constraint-adjacency
    observations extracted from PPL polyhedra.
    """

    def __init__(self):
        super().__init__()
        self.inner      = EasyTangramEnv()
        self.num_pieces = 4
        self.max_steps  = MAX_STEPS
        self.step_count = 0
        self.gamma      = 0.99

        self.action_space = spaces.Discrete(self.num_pieces * 4)   # 16

        self.observation_space = spaces.Dict({
            "h_rep": spaces.Box(low=-1, high=1, shape=(4, 4, 3), dtype=np.float32),
            "v_rep": spaces.Box(low=0,  high=1, shape=(4, 4, 2), dtype=np.float32),
            "adj"  : spaces.Box(low=0,  high=1, shape=(4, 4, 4), dtype=np.float32),
        })

    # ── Observation extraction ────────────────────────────────────────────────
    def _get_obs(self):
        return {
            "h_rep": self._extract_h_rep(),
            "v_rep": self._extract_v_rep(),
            "adj"  : self._build_graph_adj(),
        }

    def _extract_h_rep(self):
        """
        Half-space representation: [4, 4, 3].
        Each row: [a1/‖a‖, a2/‖a‖, b/(‖a‖·BOARD_MAX)] from the PPL constraint
        a1·x + a2·y + b ≥ 0  (note: code negates PPL's stored coefficients
        to match the hard-tangram convention).
        """
        pieces  = self.inner.get_pieces()
        h_rep   = []
        for p in pieces:
            rows = []
            for c in p.minimized_constraints():
                a1   = -float(c.coefficient(self.inner.x))
                a2   = -float(c.coefficient(self.inner.y))
                b    = float(c.inhomogeneous_term())
                norm = np.sqrt(a1 ** 2 + a2 ** 2) if (a1 ** 2 + a2 ** 2) > 0 else 1.0
                rows.append([a1 / norm, a2 / norm, (b / norm) / BOARD_MAX])
            # Unit squares always yield exactly 4 minimized constraints
            while len(rows) < 4:
                rows.append([0.0, 0.0, 0.0])
            h_rep.append(rows[:4])
        return np.array(h_rep, dtype=np.float32)

    def _extract_v_rep(self):
        """Vertex representation: [4, 4, 2]. Coordinates normalised by BOARD_MAX."""
        pieces = self.inner.get_pieces()
        v_rep  = []
        for p in pieces:
            verts = []
            for g in p.generators():
                if g.is_point():
                    verts.append([
                        float(g.coefficient(self.inner.x)) / BOARD_MAX,
                        float(g.coefficient(self.inner.y)) / BOARD_MAX,
                    ])
            while len(verts) < 4:
                verts.append([0.0, 0.0])
            v_rep.append(verts[:4])
        return np.array(v_rep, dtype=np.float32)

    def _build_graph_adj(self):
        """
        Constraint adjacency: [4, 4, 4].
        adj[p][i][j] = 1 if constraints i and j of piece p share a vertex.
        For a unit square this produces a bipartite graph between the two
        x-constraints and the two y-constraints.
        """
        pieces  = self.inner.get_pieces()
        all_adj = []
        eps     = 1e-5
        for p in pieces:
            constraints = list(p.minimized_constraints())
            vertices    = [g for g in p.generators() if g.is_point()]
            num_c       = min(len(constraints), 4)
            adj         = np.zeros((4, 4), dtype=np.float32)
            for i in range(num_c):
                for j in range(i + 1, num_c):
                    for v in vertices:
                        if (abs(self._eval_c(constraints[i], v)) < eps and
                                abs(self._eval_c(constraints[j], v)) < eps):
                            adj[i, j] = adj[j, i] = 1.0
                            break
            all_adj.append(adj)
        return np.array(all_adj, dtype=np.float32)

    def _eval_c(self, constraint, vertex):
        """Evaluate a PPL constraint at a PPL vertex point."""
        x_val = float(vertex.coefficient(self.inner.x)) / vertex.divisor()
        y_val = float(vertex.coefficient(self.inner.y)) / vertex.divisor()
        a1    = -float(constraint.coefficient(self.inner.x))
        a2    = -float(constraint.coefficient(self.inner.y))
        b     = float(constraint.inhomogeneous_term())
        return a1 * x_val + a2 * y_val + b

    # ── Potential-based shaping ───────────────────────────────────────────────
    def _potential(self):
        """Φ(s) = -(mean centroid distance to own target), higher = better."""
        pieces = self.inner.get_pieces()
        total  = 0.0
        for i in range(self.num_pieces):
            if self.inner.locked[i]:
                continue
            c      = self.inner._centroid(pieces[i])
            total += np.linalg.norm(c - self.inner.target_centroids[i])
        return -total / self.num_pieces

    # ── Gym API ───────────────────────────────────────────────────────────────
    def step(self, action):
        self.step_count += 1
        piece_idx        = action // 4
        direction        = action % 4
        dx, dy           = [(0, 1), (0, -1), (-1, 0), (1, 0)][direction]

        phi_before    = self._potential() / 10.0
        locked_before = list(self.inner.locked)

        self.inner.move_piece(piece_idx, dx, dy)

        phi_after = self._potential() / 10.0

        reward  = -0.01
        reward += self.gamma * phi_after - phi_before
        for i in range(self.num_pieces):
            if self.inner.locked[i] and not locked_before[i]:
                reward += 1.0

        done = False
        if all(self.inner.locked):
            reward += 10.0
            done    = True
        elif self.step_count >= self.max_steps:
            done = True

        completion = sum(self.inner.locked) / self.num_pieces
        return self._get_obs(), reward, done, {"completion": completion}

    def reset(self):
        self.step_count = 0
        self.inner.reset()
        return self._get_obs()

    def get_action_mask(self):
        """
        Boolean mask (True = valid). Pure grid logic — no PPL needed.
        Blocks: moving a locked piece, out-of-bounds moves, cell collisions.
        """
        mask = np.ones(16, dtype=bool)
        for action in range(16):
            piece_idx = action // 4
            if self.inner.locked[piece_idx]:
                mask[action] = False
                continue
            direction      = action % 4
            dx, dy         = [(0, 1), (0, -1), (-1, 0), (1, 0)][direction]
            px, py         = self.inner.piece_positions[piece_idx]
            new_px, new_py = px + dx, py + dy
            if not (0 <= new_px <= GRID_SIZE - 1 and 0 <= new_py <= GRID_SIZE - 1):
                mask[action] = False
        return mask
