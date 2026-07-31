import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np


# Deterministic fallback estimated from 2,000 CrossDock training complexes.
DEFAULT_CROSSDOCK_FRAGMENT_HISTOGRAM = {
    1: 291,
    2: 337,
    3: 351,
    4: 321,
    5: 228,
    6: 192,
    7: 111,
    8: 68,
    9: 53,
    10: 23,
}


def attachment_constrained_tree_plans(
    fragment_count,
    edge_options,
    beam_width=128,
    max_plans=8,
):
    """Find high-probability spanning trees without reusing attachment atoms.

    Each edge option contains ``frag_i``, ``frag_j``, ``edge_probability`` and
    a list of attachment ``choices``.  A choice contains atom indices for both
    fragments plus an optional additive ``score``.  The search is deterministic
    and never uses docking scores.
    """
    fragment_count = int(fragment_count)
    if fragment_count <= 1:
        return [[]]
    beam_width = max(1, int(beam_width))
    max_plans = max(1, int(max_plans))

    normalized_edges = []
    for edge in edge_options:
        frag_i = int(edge["frag_i"])
        frag_j = int(edge["frag_j"])
        if frag_i == frag_j or not (0 <= frag_i < fragment_count and 0 <= frag_j < fragment_count):
            continue
        choices = []
        for choice in edge.get("choices", []):
            atom_i = int(choice["atom_i"])
            atom_j = int(choice["atom_j"])
            if atom_i == atom_j:
                continue
            copied = dict(choice)
            copied["atom_i"] = atom_i
            copied["atom_j"] = atom_j
            copied["score"] = float(choice.get("score", 0.0))
            choices.append(copied)
        if not choices:
            continue
        copied_edge = dict(edge)
        copied_edge["frag_i"] = frag_i
        copied_edge["frag_j"] = frag_j
        copied_edge["edge_probability"] = float(edge.get("edge_probability", 0.0))
        copied_edge["choices"] = sorted(
            choices,
            key=lambda item: (
                -float(item.get("score", 0.0)),
                int(item["atom_i"]),
                int(item["atom_j"]),
                int(item.get("bond_label", 1)),
            ),
        )
        normalized_edges.append(copied_edge)

    normalized_edges.sort(
        key=lambda edge: (
            -float(edge["edge_probability"]),
            int(edge["frag_i"]),
            int(edge["frag_j"]),
        )
    )
    if not normalized_edges:
        return []

    # A state grows one connected component from a deterministic choice of root.
    states = [
        {
            "score": 0.0,
            "connected": frozenset([root]),
            "used_atoms": frozenset(),
            "plan": tuple(),
        }
        for root in range(fragment_count)
    ]

    for _ in range(fragment_count - 1):
        expanded = []
        for state in states:
            connected = state["connected"]
            used_atoms = state["used_atoms"]
            for edge in normalized_edges:
                frag_i = edge["frag_i"]
                frag_j = edge["frag_j"]
                i_connected = frag_i in connected
                j_connected = frag_j in connected
                if i_connected == j_connected:
                    continue
                parent_frag, child_frag = (
                    (frag_i, frag_j) if i_connected else (frag_j, frag_i)
                )
                for choice in edge["choices"]:
                    atom_i = choice["atom_i"]
                    atom_j = choice["atom_j"]
                    if atom_i in used_atoms or atom_j in used_atoms:
                        continue
                    parent_atom, child_atom = (
                        (atom_i, atom_j) if i_connected else (atom_j, atom_i)
                    )
                    selected = dict(edge)
                    selected.pop("choices", None)
                    selected.update(choice)
                    selected.update({
                        "parent_frag": parent_frag,
                        "child_frag": child_frag,
                        "parent_atom": parent_atom,
                        "child_atom": child_atom,
                    })
                    edge_score = math.log(max(float(edge["edge_probability"]), 1e-12))
                    expanded.append({
                        "score": state["score"] + edge_score + float(choice.get("score", 0.0)),
                        "connected": frozenset((*connected, child_frag)),
                        "used_atoms": frozenset((*used_atoms, atom_i, atom_j)),
                        "plan": state["plan"] + (selected,),
                    })

        if not expanded:
            return []

        # Keep the best path for an equivalent connectivity/attachment state.
        best_by_state = {}
        for state in expanded:
            key = (tuple(sorted(state["connected"])), tuple(sorted(state["used_atoms"])))
            previous = best_by_state.get(key)
            if previous is None or state["score"] > previous["score"]:
                best_by_state[key] = state

        def state_key(state):
            signature = tuple(
                (
                    int(edge["parent_frag"]),
                    int(edge["child_frag"]),
                    int(edge["parent_atom"]),
                    int(edge["child_atom"]),
                    int(edge.get("bond_label", 1)),
                )
                for edge in state["plan"]
            )
            return (-float(state["score"]), signature)

        states = sorted(best_by_state.values(), key=state_key)[:beam_width]

    complete = [state for state in states if len(state["connected"]) == fragment_count]
    complete.sort(key=lambda state: (-float(state["score"]), tuple(sorted(state["used_atoms"]))))
    return [list(state["plan"]) for state in complete[:max_plans]]


def load_fragment_histogram(path=None, max_count=10):
    histogram = DEFAULT_CROSSDOCK_FRAGMENT_HISTOGRAM
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        histogram = payload.get("histogram", payload)
    cleaned = {
        int(count): int(weight)
        for count, weight in histogram.items()
        if 1 <= int(count) <= int(max_count) and int(weight) > 0
    }
    if not cleaned:
        raise ValueError("Fragment-count histogram contains no positive entries in the requested range")
    return dict(sorted(cleaned.items()))


class FragmentCountSampler:
    def __init__(self, mode="empirical", legacy_value="5", max_count=10, seed=2025, histogram_path=None):
        self.mode = mode
        self.max_count = int(max_count)
        self.rng = random.Random(int(seed))
        self.histogram = load_fragment_histogram(histogram_path, self.max_count) if mode == "empirical" else None

        values = [int(part.strip()) for part in str(legacy_value).split(",") if part.strip()]
        if not values:
            raise ValueError("Fragment count requires at least one integer")
        self.fixed = values[0]
        self.low = min(values)
        self.high = max(values)

    def sample(self, size=None):
        if size is not None:
            return [self.sample() for _ in range(int(size))]
        if self.mode == "empirical":
            counts = list(self.histogram)
            weights = [self.histogram[count] for count in counts]
            return int(self.rng.choices(counts, weights=weights, k=1)[0])
        if self.mode == "fixed":
            value = self.fixed
        elif self.mode == "range":
            value = self.rng.randint(self.low, self.high)
        else:
            raise ValueError(f"Unsupported fragment-count mode: {self.mode}")
        return max(1, min(int(value), self.max_count))


def fragment_count_histogram(dataset, limit=None):
    counts = Counter()
    total = len(dataset) if limit is None else min(len(dataset), int(limit))
    failures = 0
    for index in range(total):
        try:
            item = dataset[index]
            counts[int(item["positions"].shape[0])] += 1
        except Exception:
            failures += 1
    return dict(sorted(counts.items())), failures


def rotation_matrix_align_vectors(source, target, eps=1e-8):
    """Return a proper rotation matrix that aligns one 3D vector to another."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < eps or target_norm < eps:
        return np.eye(3, dtype=np.float64)

    a = source / source_norm
    b = target / target_norm
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < eps:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(a)))] = 1.0
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    axis = cross / sine
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    angle = math.atan2(sine, cosine)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def rotate_points_about_axis(points, origin, axis, angle_radians, eps=1e-8):
    """Rotate points around a 3D axis passing through origin."""
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < eps:
        return points.copy()
    axis = axis / norm
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3)
        + math.sin(float(angle_radians)) * skew
        + (1.0 - math.cos(float(angle_radians))) * (skew @ skew)
    )
    return (points - origin) @ rotation.T + origin
