"""BYOL-Explore: curiosity-driven exploration for ZX-diagram composition.

Implements a pure-NumPy version of the BYOL-Explore algorithm (Guo et al., 2022)
adapted for the ZX-Webs quantum algorithm discovery pipeline.

Instead of pixel observations and continuous actions, this module operates on:
- **States**: Feature vectors extracted from ZX-diagrams (graph statistics)
- **Actions**: Composition operations (sequential, parallel, phase-perturb)
  parameterised by (web_i, web_j, mode)
- **Intrinsic reward**: Prediction error in a learned latent space

The curiosity-driven policy prioritises compositions that produce *surprising*
diagrams — those the predictor cannot yet anticipate — focusing search on
novel regions of the algorithm space rather than exhaustive enumeration.

Architecture
------------
- **Online encoder** f_θ: Maps ZX-diagram features → latent z
- **Predictor** q_θ: Given (z_t, action_embedding), predicts z_{t+1}
- **Target encoder** f_ξ: EMA of online encoder, provides stable targets
- **Intrinsic reward**: ||q_θ(z_t, a) - sg(f_ξ(x_{t+1}))||²

All networks are simple MLPs (2-3 hidden layers) operating on hand-crafted
graph features — no GNN needed since the feature extraction is explicit.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyzx as zx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction from ZX-diagrams
# ---------------------------------------------------------------------------

_VT_BOUNDARY = 0
_VT_Z = 1
_VT_X = 2
_VT_H = 3


def extract_graph_features(graph: zx.Graph) -> np.ndarray:
    """Extract a fixed-size feature vector from a PyZX graph.

    Features (16-dimensional):
      0: n_qubits (number of input boundaries)
      1: n_outputs
      2: n_z_spiders
      3: n_x_spiders
      4: n_h_boxes
      5: n_simple_edges
      6: n_hadamard_edges
      7: total_vertices
      8: total_edges
      9: mean_phase (of Z spiders, normalised to [0,1])
     10: std_phase (of Z spiders)
     11: mean_phase_x (of X spiders)
     12: std_phase_x
     13: density (edges / max possible edges)
     14: boundary_ratio (boundaries / total vertices)
     15: spider_type_ratio (n_z / (n_z + n_x + 1))
    """
    vertices = list(graph.vertices())
    n_verts = len(vertices)

    input_set = set(graph.inputs()) if graph.inputs() else set()
    output_set = set(graph.outputs()) if graph.outputs() else set()

    n_z = n_x = n_h = n_boundary = 0
    z_phases = []
    x_phases = []

    for v in vertices:
        vtype = graph.type(v)
        if vtype == _VT_BOUNDARY:
            n_boundary += 1
        elif vtype == _VT_Z:
            n_z += 1
            z_phases.append(float(graph.phase(v)) % 2.0)
        elif vtype == _VT_X:
            n_x += 1
            x_phases.append(float(graph.phase(v)) % 2.0)
        elif vtype == _VT_H:
            n_h += 1

    edges = list(graph.edges())
    n_edges = len(edges)
    n_simple = sum(1 for e in edges if graph.edge_type(e) == 1)
    n_hadamard = n_edges - n_simple

    max_edges = n_verts * (n_verts - 1) / 2 if n_verts > 1 else 1
    density = n_edges / max_edges

    z_phases_arr = np.array(z_phases) if z_phases else np.array([0.0])
    x_phases_arr = np.array(x_phases) if x_phases else np.array([0.0])

    features = np.array([
        len(input_set),
        len(output_set),
        n_z,
        n_x,
        n_h,
        n_simple,
        n_hadamard,
        n_verts,
        n_edges,
        z_phases_arr.mean() / 2.0,  # normalise to [0,1]
        z_phases_arr.std() / 2.0,
        x_phases_arr.mean() / 2.0,
        x_phases_arr.std() / 2.0,
        density,
        n_boundary / max(n_verts, 1),
        n_z / max(n_z + n_x + 1, 1),
    ], dtype=np.float32)

    return features


FEATURE_DIM = 16
ACTION_DIM = 4  # (web_i_idx_normalized, web_j_idx_normalized, mode_onehot_seq, mode_onehot_par)


def encode_action(web_i: int, web_j: int, mode: str, n_webs: int) -> np.ndarray:
    """Encode a composition action as a fixed-size vector."""
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = web_i / max(n_webs, 1)
    action[1] = web_j / max(n_webs, 1)
    if mode == "sequential":
        action[2] = 1.0
    elif mode in ("parallel", "parallel_stitch"):
        action[3] = 1.0
    else:
        action[2] = 0.5
        action[3] = 0.5
    return action


# ---------------------------------------------------------------------------
# Simple MLP (pure NumPy)
# ---------------------------------------------------------------------------


class NumpyMLP:
    """A simple multi-layer perceptron using only NumPy.

    Uses ReLU activations for hidden layers and no activation on the output.
    Supports forward pass and gradient-based updates via simple backprop.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rng = rng or np.random.default_rng(42)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialization
            std = math.sqrt(2.0 / fan_in)
            w = self.rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x can be (d,) or (batch, d)."""
        h = x.astype(np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:  # ReLU for hidden layers
                h = np.maximum(h, 0)
        return h

    def forward_with_intermediates(self, x: np.ndarray) -> list[np.ndarray]:
        """Forward pass returning all layer activations for backprop."""
        activations = [x.astype(np.float32)]
        h = activations[0]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            pre_act = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(pre_act, 0)
            else:
                h = pre_act
            activations.append(h)
        return activations

    def parameters_flat(self) -> np.ndarray:
        """Return all parameters as a single flat vector."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def load_parameters_flat(self, flat: np.ndarray) -> None:
        """Load parameters from a flat vector."""
        offset = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            self.weights[i] = flat[offset:offset + w_size].reshape(
                self.weights[i].shape
            ).astype(np.float32)
            offset += w_size
            b_size = self.biases[i].size
            self.biases[i] = flat[offset:offset + b_size].astype(np.float32)
            offset += b_size

    def copy(self) -> "NumpyMLP":
        """Create a deep copy."""
        new = NumpyMLP.__new__(NumpyMLP)
        new.rng = self.rng
        new.weights = [w.copy() for w in self.weights]
        new.biases = [b.copy() for b in self.biases]
        return new


def _sgd_update(
    mlp: NumpyMLP,
    x: np.ndarray,
    target: np.ndarray,
    lr: float = 0.001,
    max_grad_norm: float = 1.0,
) -> float:
    """One step of SGD on MSE loss with gradient clipping. Returns the loss value."""
    # Forward with intermediates
    acts = mlp.forward_with_intermediates(x)
    pred = acts[-1]

    # Check for NaN/Inf
    if not np.isfinite(pred).all():
        return float("nan")

    # MSE loss
    diff = pred - target
    loss = float(np.mean(diff ** 2))
    if not np.isfinite(loss):
        return float("nan")

    # Backprop through layers
    grad = 2.0 * diff / max(diff.size, 1)

    for i in reversed(range(len(mlp.weights))):
        h_in = acts[i]
        if h_in.ndim == 1:
            h_in = h_in.reshape(1, -1)
        if grad.ndim == 1:
            grad = grad.reshape(1, -1)

        # Clip gradient to prevent explosion
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / (grad_norm + 1e-8))

        # Gradient for weights and biases
        dw = h_in.T @ grad
        db = grad.sum(axis=0)

        # Clip weight gradients too
        dw_norm = np.linalg.norm(dw)
        if dw_norm > max_grad_norm:
            dw = dw * (max_grad_norm / (dw_norm + 1e-8))

        # Update
        mlp.weights[i] -= lr * dw
        mlp.biases[i] -= lr * db

        # Gradient for input
        grad = grad @ mlp.weights[i].T

        # ReLU backward (for hidden layers only)
        if i > 0:
            mask = (acts[i] > 0).astype(np.float32)
            if mask.ndim == 1:
                mask = mask.reshape(1, -1)
            grad = grad * mask

    return loss


# ---------------------------------------------------------------------------
# BYOL-Explore Agent
# ---------------------------------------------------------------------------

LATENT_DIM = 32
HIDDEN_DIM = 64


@dataclass
class ExplorationResult:
    """Result of a single exploration step."""
    web_i: int
    web_j: int
    mode: str
    intrinsic_reward: float
    graph: Any = None  # pyzx Graph
    features: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM))


class BYOLExploreAgent:
    """BYOL-Explore agent for curiosity-driven ZX-diagram composition.

    The agent maintains:
    - An online encoder that maps graph features to a latent space
    - A predictor that predicts the next latent state from current state + action
    - A target encoder (EMA of online) that provides stable prediction targets
    - An action-value estimator that tracks which compositions are most surprising

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    ema_tau : float
        EMA coefficient for target network update (0.01 = slow update).
    lr : float
        Learning rate for online network updates.
    epsilon : float
        Exploration rate for epsilon-greedy action selection.
    """

    def __init__(
        self,
        seed: int = 42,
        ema_tau: float = 0.01,
        lr: float = 0.001,
        epsilon: float = 0.2,
    ) -> None:
        self.rng_np = np.random.default_rng(seed)
        self.rng = random.Random(seed)
        self.ema_tau = ema_tau
        self.lr = lr
        self.epsilon = epsilon

        # Online encoder: features → latent
        self.online_encoder = NumpyMLP(
            [FEATURE_DIM, HIDDEN_DIM, HIDDEN_DIM, LATENT_DIM],
            rng=self.rng_np,
        )

        # Predictor: (latent + action) → predicted next latent
        self.predictor = NumpyMLP(
            [LATENT_DIM + ACTION_DIM, HIDDEN_DIM, HIDDEN_DIM, LATENT_DIM],
            rng=self.rng_np,
        )

        # Target encoder: EMA copy of online encoder
        self.target_encoder = self.online_encoder.copy()

        # Statistics for normalising features
        self._feature_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self._feature_std = np.ones(FEATURE_DIM, dtype=np.float32)
        self._n_observations = 0

        # Replay buffer for experience
        self._replay: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._max_replay = 10000

        # Tracking
        self.total_steps = 0
        self.total_reward = 0.0
        self.reward_history: list[float] = []

    def _normalise_features(self, features: np.ndarray) -> np.ndarray:
        """Normalise features using running statistics."""
        return (features - self._feature_mean) / (self._feature_std + 1e-8)

    def _update_feature_stats(self, features: np.ndarray) -> None:
        """Update running mean/std of features (Welford's algorithm)."""
        self._n_observations += 1
        n = self._n_observations
        delta = features - self._feature_mean
        self._feature_mean += delta / n
        if n > 1:
            # Running variance update
            self._feature_std = np.sqrt(
                ((n - 2) / (n - 1)) * self._feature_std ** 2
                + delta ** 2 / n
            )

    def _update_target_network(self) -> None:
        """Update target encoder via exponential moving average."""
        tau = self.ema_tau
        for i in range(len(self.target_encoder.weights)):
            self.target_encoder.weights[i] = (
                (1 - tau) * self.target_encoder.weights[i]
                + tau * self.online_encoder.weights[i]
            )
            self.target_encoder.biases[i] = (
                (1 - tau) * self.target_encoder.biases[i]
                + tau * self.online_encoder.biases[i]
            )

    def compute_intrinsic_reward(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
        next_features: np.ndarray,
    ) -> float:
        """Compute the BYOL-Explore intrinsic reward.

        reward = ||predictor(encoder(x_t), a) - target_encoder(x_{t+1})||²

        High reward = the predictor couldn't anticipate this outcome = novel.
        """
        # Encode current state
        z_t = self.online_encoder.forward(
            self._normalise_features(current_features).reshape(1, -1)
        )

        # Predict next latent
        pred_input = np.concatenate([z_t.flatten(), action]).reshape(1, -1)
        z_pred = self.predictor.forward(pred_input)

        # Target: encode actual next state (stop gradient = no update)
        z_target = self.target_encoder.forward(
            self._normalise_features(next_features).reshape(1, -1)
        )

        # Prediction error = intrinsic reward
        error = np.mean((z_pred - z_target) ** 2)
        if not np.isfinite(error):
            return 0.0
        return float(np.clip(error, 0.0, 100.0))

    def train_step(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
        next_features: np.ndarray,
    ) -> float:
        """Train the online encoder and predictor on one transition.

        Returns the prediction loss.
        """
        # Add to replay buffer
        self._replay.append((
            current_features.copy(),
            action.copy(),
            next_features.copy(),
        ))
        if len(self._replay) > self._max_replay:
            self._replay.pop(0)

        # Sample a mini-batch from replay
        batch_size = min(16, len(self._replay))
        indices = self.rng.sample(range(len(self._replay)), batch_size)

        total_loss = 0.0
        for idx in indices:
            curr_f, act, next_f = self._replay[idx]

            # Forward: encode current, predict next
            curr_norm = self._normalise_features(curr_f).reshape(1, -1)
            z_t = self.online_encoder.forward(curr_norm)
            pred_input = np.concatenate([z_t.flatten(), act]).reshape(1, -1)

            # Target (detached)
            next_norm = self._normalise_features(next_f).reshape(1, -1)
            z_target = self.target_encoder.forward(next_norm).flatten()

            # Update predictor
            loss = _sgd_update(self.predictor, pred_input, z_target.reshape(1, -1), self.lr)
            total_loss += loss

            # Update online encoder (via chain rule approximation:
            # the encoder should produce representations that make
            # prediction easier)
            _sgd_update(self.online_encoder, curr_norm, z_target.reshape(1, -1), self.lr * 0.1)

        # Update target network (EMA)
        self._update_target_network()

        return total_loss / batch_size

    def select_action(
        self,
        current_features: np.ndarray,
        candidate_actions: list[tuple[int, int, str]],
        webs_features: list[np.ndarray],
        n_webs: int,
    ) -> tuple[int, int, str, float]:
        """Select the most curiosity-driven action.

        With probability epsilon, picks a random action (exploration).
        Otherwise, picks the action with the highest predicted intrinsic reward.

        Returns (web_i, web_j, mode, predicted_reward).
        """
        if not candidate_actions:
            raise ValueError("No candidate actions to select from")

        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            choice = self.rng.choice(candidate_actions)
            return (*choice, 0.0)

        # Score each candidate action by predicted surprise
        best_score = -float("inf")
        best_action = candidate_actions[0]

        # Encode current state once
        curr_norm = self._normalise_features(current_features).reshape(1, -1)
        z_t = self.online_encoder.forward(curr_norm)

        for web_i, web_j, mode in candidate_actions:
            action_vec = encode_action(web_i, web_j, mode, n_webs)
            pred_input = np.concatenate([z_t.flatten(), action_vec]).reshape(1, -1)
            z_pred = self.predictor.forward(pred_input)

            # Estimate surprise: magnitude of predicted change
            # (proxy for prediction error — larger predicted changes
            # indicate the model expects something unusual)
            predicted_magnitude = float(np.mean(z_pred ** 2))

            # Also factor in how different the pair's features are
            if web_i < len(webs_features) and web_j < len(webs_features):
                feat_diff = np.mean(
                    (webs_features[web_i] - webs_features[web_j]) ** 2
                )
                predicted_magnitude += 0.5 * feat_diff

            if predicted_magnitude > best_score:
                best_score = predicted_magnitude
                best_action = (web_i, web_j, mode)

        return (*best_action, best_score)

    def get_stats(self) -> dict[str, Any]:
        """Return training statistics."""
        recent = self.reward_history[-100:] if self.reward_history else [0.0]
        return {
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "mean_recent_reward": float(np.mean(recent)),
            "max_recent_reward": float(np.max(recent)) if recent else 0.0,
            "replay_size": len(self._replay),
            "n_observations": self._n_observations,
        }


# ---------------------------------------------------------------------------
# Curiosity-guided composition loop
# ---------------------------------------------------------------------------


def run_curiosity_exploration(
    webs: list,  # list[ZXWeb]
    stitcher: Any,  # Stitcher instance
    config: Any,  # ComposeConfig
    n_episodes: int = 5,
    steps_per_episode: int = 200,
    seed: int = 42,
) -> tuple[list, list[dict]]:
    """Run BYOL-Explore curiosity-driven composition.

    This replaces the brute-force enumeration in Stage 4 with a learned
    exploration strategy. The agent:

    1. Picks a random "current state" (a web's graph features)
    2. Selects an action (composition operation) maximising predicted surprise
    3. Executes the composition
    4. Computes intrinsic reward (prediction error)
    5. Trains the prediction networks
    6. Adds successful compositions as candidates

    Parameters
    ----------
    webs : list[ZXWeb]
        Available ZX-Webs for composition.
    stitcher : Stitcher
        The existing stitcher for executing compositions.
    config : ComposeConfig
        Composition parameters.
    n_episodes : int
        Number of exploration episodes.
    steps_per_episode : int
        Composition attempts per episode.
    seed : int
        Random seed.

    Returns
    -------
    candidates : list[CandidateAlgorithm]
        Discovered candidate algorithms.
    exploration_log : list[dict]
        Per-step exploration statistics.
    """
    from zx_webs.stage4_compose.candidate import CandidateAlgorithm

    agent = BYOLExploreAgent(seed=seed, epsilon=0.3, lr=0.0005)
    rng = random.Random(seed)

    max_candidates = config.max_candidates
    min_qubits = config.min_compose_qubits
    max_qubits = config.max_compose_qubits

    # Pre-compute features for all webs
    logger.info("BYOL-Explore: computing features for %d webs...", len(webs))
    web_features: list[np.ndarray] = []
    web_graphs: list[zx.Graph | None] = []

    for web in webs:
        try:
            g = web.to_pyzx_graph()
            feats = extract_graph_features(g)
            web_features.append(feats)
            web_graphs.append(g)
            agent._update_feature_stats(feats)
        except Exception:
            web_features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            web_graphs.append(None)

    n_webs = len(webs)
    if n_webs < 2:
        logger.warning("BYOL-Explore: need at least 2 webs, got %d", n_webs)
        return [], []

    # Valid web indices (those with successful graph construction)
    valid_indices = [i for i in range(n_webs) if web_graphs[i] is not None]
    if len(valid_indices) < 2:
        logger.warning("BYOL-Explore: not enough valid webs")
        return [], []

    candidates: list[CandidateAlgorithm] = []
    exploration_log: list[dict] = []
    cand_idx = 0
    modes = ["sequential", "parallel", "parallel_stitch", "phase_perturb"]

    # Track seen compositions to avoid duplicates
    seen_compositions: set[tuple[int, int, str]] = set()

    # Track reward statistics per episode for adaptive epsilon
    episode_rewards: list[list[float]] = []

    logger.info(
        "BYOL-Explore: starting %d episodes x %d steps "
        "(max %d candidates, %d webs)",
        n_episodes, steps_per_episode, max_candidates, len(valid_indices),
    )

    for episode in range(n_episodes):
        episode_reward_list: list[float] = []

        # Decay epsilon over episodes
        agent.epsilon = max(0.05, 0.3 * (1 - episode / max(n_episodes, 1)))

        for step in range(steps_per_episode):
            if len(candidates) >= max_candidates:
                break

            # Pick a random "current" web as starting state
            curr_idx = rng.choice(valid_indices)
            curr_features = web_features[curr_idx]

            # Generate candidate actions: sample a subset of possible pairs
            n_action_candidates = min(50, len(valid_indices))
            action_partner_indices = rng.sample(
                valid_indices, min(n_action_candidates, len(valid_indices))
            )
            candidate_actions: list[tuple[int, int, str]] = []
            for j_idx in action_partner_indices:
                if j_idx == curr_idx:
                    continue
                for mode in modes:
                    key = (min(curr_idx, j_idx), max(curr_idx, j_idx), mode)
                    if key not in seen_compositions:
                        candidate_actions.append((curr_idx, j_idx, mode))

            if not candidate_actions:
                continue

            # Select action using curiosity
            web_i, web_j, mode, predicted_reward = agent.select_action(
                curr_features, candidate_actions, web_features, n_webs,
            )

            # Execute the composition
            composed_graph = None
            comp_type = mode

            try:
                if mode == "sequential":
                    composed_graph = stitcher.compose_sequential(
                        webs[web_i], webs[web_j]
                    )
                elif mode == "parallel":
                    composed_graph = stitcher.compose_parallel(
                        webs[web_i], webs[web_j]
                    )
                elif mode == "parallel_stitch":
                    composed_graph = stitcher.compose_parallel_stitch(
                        webs[web_i], webs[web_j], n_hadamard_edges=1
                    )
                elif mode == "phase_perturb":
                    # First compose sequentially, then perturb
                    base_graph = stitcher.compose_sequential(
                        webs[web_i], webs[web_j]
                    )
                    if base_graph is not None:
                        composed_graph = stitcher.perturb_phases(base_graph)
                    comp_type = "byol_phase_perturb"
            except Exception as exc:
                logger.debug("Composition failed: %s", exc)
                composed_graph = None

            # Mark this composition as tried
            seen_key = (min(web_i, web_j), max(web_i, web_j), mode)
            seen_compositions.add(seen_key)

            # Compute features of result and intrinsic reward
            if composed_graph is not None:
                try:
                    next_features = extract_graph_features(composed_graph)
                except Exception:
                    next_features = np.zeros(FEATURE_DIM, dtype=np.float32)

                action_vec = encode_action(web_i, web_j, mode, n_webs)
                intrinsic_reward = agent.compute_intrinsic_reward(
                    curr_features, action_vec, next_features,
                )

                # Train the agent
                train_loss = agent.train_step(
                    curr_features, action_vec, next_features,
                )

                agent._update_feature_stats(next_features)
                agent.total_steps += 1
                agent.total_reward += intrinsic_reward
                agent.reward_history.append(intrinsic_reward)
                episode_reward_list.append(intrinsic_reward)

                # Check if this is a valid candidate
                n_qubits = len(composed_graph.inputs()) if composed_graph.inputs() else 0
                n_outputs = len(composed_graph.outputs()) if composed_graph.outputs() else 0

                if (
                    min_qubits <= n_qubits <= max_qubits
                    and n_qubits == n_outputs
                    and n_qubits > 0
                ):
                    n_spiders = sum(
                        1 for v in composed_graph.vertices()
                        if composed_graph.type(v) != _VT_BOUNDARY
                    )

                    # Collect source families
                    families: list[str] = []
                    seen_fam: set[str] = set()
                    for idx in [web_i, web_j]:
                        for fam in webs[idx].source_families:
                            if fam not in seen_fam:
                                families.append(fam)
                                seen_fam.add(fam)

                    is_cross = len(seen_fam) > 1

                    cand = CandidateAlgorithm(
                        candidate_id=f"byol_{cand_idx:04d}",
                        graph_json=composed_graph.to_json(),
                        component_web_ids=[
                            webs[web_i].web_id, webs[web_j].web_id,
                        ],
                        composition_type=f"byol_{comp_type}",
                        n_qubits=n_qubits,
                        n_spiders=n_spiders,
                        source_families=families,
                        is_cross_family=is_cross,
                    )
                    candidates.append(cand)
                    cand_idx += 1

                exploration_log.append({
                    "episode": episode,
                    "step": step,
                    "web_i": web_i,
                    "web_j": web_j,
                    "mode": mode,
                    "intrinsic_reward": intrinsic_reward,
                    "train_loss": train_loss,
                    "predicted_reward": predicted_reward,
                    "n_qubits": n_qubits if composed_graph else 0,
                    "valid_candidate": n_qubits > 0 and n_qubits == n_outputs,
                    "total_candidates": len(candidates),
                })
            else:
                # Failed composition — still informative
                action_vec = encode_action(web_i, web_j, mode, n_webs)
                # Use a zero feature vector for failed compositions
                dummy_features = np.zeros(FEATURE_DIM, dtype=np.float32)
                intrinsic_reward = agent.compute_intrinsic_reward(
                    curr_features, action_vec, dummy_features,
                )
                agent.train_step(curr_features, action_vec, dummy_features)
                agent.total_steps += 1
                episode_reward_list.append(0.0)

        episode_rewards.append(episode_reward_list)
        mean_ep_reward = (
            np.mean(episode_reward_list) if episode_reward_list else 0.0
        )
        logger.info(
            "BYOL-Explore episode %d/%d: %d candidates, "
            "mean_reward=%.4f, epsilon=%.3f",
            episode + 1, n_episodes, len(candidates),
            mean_ep_reward, agent.epsilon,
        )

    # Summary
    stats = agent.get_stats()
    logger.info(
        "BYOL-Explore complete: %d candidates from %d steps. "
        "Mean intrinsic reward: %.4f",
        len(candidates), stats["total_steps"], stats["mean_recent_reward"],
    )

    return candidates, exploration_log
