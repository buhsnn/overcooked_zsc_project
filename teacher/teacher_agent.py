# teacher/teacher_agent.py

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils.layout_utils import (
    AVAILABLE_LAYOUTS,
    APPROX_OPTIMAL_RETURN,
    one_hot_layout,
    mutate_layout,
)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = x / max(temperature, 1e-8)
    x = x - np.max(x)
    exps = np.exp(x)
    return exps / np.maximum(np.sum(exps), 1e-8)


@dataclass
class LevelRecord:
    """Information stored for each layout in the buffer."""
    layout_name: str
    returns: List[float] = field(default_factory=list)
    regret: float = 0.0
    novelty: float = 0.0
    progress: float = 0.0
    score: float = 0.0   # Final composite score
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(len(AVAILABLE_LAYOUTS)))


class LevelBuffer:
    """Buffer of layouts + associated statistics."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.levels: Dict[str, LevelRecord] = {}

    def ensure_level(self, layout_name: str) -> LevelRecord:
        if layout_name not in self.levels:
            if len(self.levels) >= self.max_size:
                # Remove the oldest layout (simplified: first key)
                drop_key = next(iter(self.levels.keys()))
                del self.levels[drop_key]
            rec = LevelRecord(
                layout_name=layout_name,
                embedding=one_hot_layout(layout_name),
            )
            self.levels[layout_name] = rec
        return self.levels[layout_name]

    def update_return(self, layout_name: str, episode_return: float):
        rec = self.ensure_level(layout_name)
        rec.returns.append(float(episode_return))

    def all_records(self) -> List[LevelRecord]:
        return list(self.levels.values())


class TeacherAgent:
    """
    Simple teacher based on:
    - Regret
    - Novelty
    - Student Progress

    It manages:
    - a buffer of layouts
    - a composite score
    - a sample_layout() function to choose the next level
    """

    def __init__(
        self,
        buffer_size: int = 50,
        w_regret: float = 1.0,
        w_novelty: float = 0.5,
        w_progress: float = 0.5,
        temperature: float = 1.0,
    ):
        self.buffer = LevelBuffer(max_size=buffer_size)
        self.w_regret = w_regret
        self.w_novelty = w_novelty
        self.w_progress = w_progress
        self.temperature = temperature

        # Also keep a history for progress
        # {layout_name: last_return}
        self.last_return: Dict[str, Optional[float]] = {}

        # Initial filling of the buffer: put a few base layouts
        self._init_buffer()

    def _init_buffer(self):
        initial_layouts = AVAILABLE_LAYOUTS[:3]  # can be extended later
        for name in initial_layouts:
            rec = self.buffer.ensure_level(name)
            # Initialize with a small fake return (0) to start
            rec.returns.append(0.0)
            self.last_return[name] = 0.0

    # --------------------- metrics --------------------- #

    def _compute_regret(self, rec: LevelRecord) -> float:
        if not rec.returns:
            return 0.0
        avg_ret = np.mean(rec.returns)
        optimal = APPROX_OPTIMAL_RETURN.get(rec.layout_name, 200)
        return max(optimal - avg_ret, 0.0)

    def _compute_novelty(self, rec: LevelRecord, others: List[LevelRecord]) -> float:
        if not others:
            return 0.0
        dists = []
        for o in others:
            if o.layout_name == rec.layout_name:
                continue
            d = np.linalg.norm(rec.embedding - o.embedding)
            dists.append(d)
        if not dists:
            return 0.0
        return float(np.mean(dists))

    def _compute_progress(self, rec: LevelRecord) -> float:
        if len(rec.returns) < 2:
            return 0.0
        # Last − previous
        return float(abs(rec.returns[-1] - rec.returns[-2]))

    def _update_scores(self):
        recs = self.buffer.all_records()
        if not recs:
            return

        # Compute regret/novelty/progress for each layout
        regrets = []
        novelties = []
        progresses = []
        for rec in recs:
            rec.regret = self._compute_regret(rec)
            rec.novelty = self._compute_novelty(rec, recs)
            rec.progress = self._compute_progress(rec)

            regrets.append(rec.regret)
            novelties.append(rec.novelty)
            progresses.append(rec.progress)

        # Normalization (simple z-score to avoid very different scales)
        def normalize(xs):
            xs = np.array(xs, dtype=float)
            if np.all(xs == 0):
                return np.zeros_like(xs)
            mean = xs.mean()
            std = xs.std() if xs.std() > 1e-8 else 1.0
            return (xs - mean) / std

        n_regret = normalize(regrets)
        n_novelty = normalize(novelties)
        n_progress = normalize(progresses)

        for i, rec in enumerate(recs):
            rec.score = (
                self.w_regret * n_regret[i]
                + self.w_novelty * n_novelty[i]
                + self.w_progress * n_progress[i]
            )

    # --------------------- public API --------------------- #

    def sample_layout(self) -> str:
        """
        Choose a layout from the buffer via softmax over the score.
        If everything is empty at the beginning, pick one at random.
        """
        recs = self.buffer.all_records()
        if not recs:
            return random.choice(AVAILABLE_LAYOUTS)

        self._update_scores()
        scores = np.array([r.score for r in recs], dtype=float)
        probs = softmax(scores, temperature=self.temperature)
        idx = np.random.choice(len(recs), p=probs)
        return recs[idx].layout_name

    def update_after_episode(self, layout_name: str, episode_return: float):
        """Call this after training/evaluating the student on a layout."""
        self.buffer.update_return(layout_name, episode_return)

        # Update memory for progress (not mandatory, but useful if you want another progress style)
        prev = self.last_return.get(layout_name, None)
        self.last_return[layout_name] = episode_return

        # Option: create a mutated layout and add it to the buffer
        mutated = mutate_layout(layout_name)
        self.buffer.ensure_level(mutated)
