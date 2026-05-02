"""
causal_graph.py — Causal Learning Engine

Learns structured cause→effect relationships from experience:
  • TransitionRecord: snapshot of (state, action, outcome, reward, surprise)
  • CausalEdge: weighted directed edge encoding reliability of a transition
  • CausalGraph: graph of state→action→outcome with context-dependent reliability

Unlike the associative WorldModel in consciousness.py (which tracks valence deltas),
this module learns WHAT happens (state transitions) and HOW RELIABLY, enabling
strategic planning instead of purely reactive behaviour.

Integration:
  - task_executive.py writes TransitionRecords after each skill execution
  - consciousness.py queries causal predictions for goal evaluation
  - persistence.py serialises the graph to SQLite
"""

from __future__ import annotations

import hashlib
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# TransitionRecord — one observed state transition
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TransitionRecord:
    """Single observed transition: state + action → outcome."""

    tick: int
    state_signature: str  # compact descriptor of pre-state
    action_kind: str  # skill/action type executed
    action_args: Dict  # parameters of the action
    predicted_outcome: str  # what the system expected
    observed_outcome: str  # what actually happened
    reward: float  # emotional/goal reward signal [-1, 1]
    surprise: float  # prediction error magnitude [0, 1]
    success: bool  # did the action achieve its intent?
    involved_persons: List[str] = field(default_factory=list)
    context_tags: List[str] = field(default_factory=list)
    state_type: str = ""  # "goal" | "skill" | "" (legacy)

    def outcome_match(self) -> float:
        """How well did prediction match observation? 1.0 = perfect."""
        if self.predicted_outcome == self.observed_outcome:
            return 1.0
        if not self.predicted_outcome or not self.observed_outcome:
            return 0.0
        # Simple word overlap
        pred_words = set(self.predicted_outcome.lower().split())
        obs_words = set(self.observed_outcome.lower().split())
        if not pred_words:
            return 0.0
        return len(pred_words & obs_words) / len(pred_words | obs_words)

    def to_dict(self) -> Dict:
        return {
            "tick": self.tick,
            "state_signature": self.state_signature,
            "action_kind": self.action_kind,
            "action_args": self.action_args,
            "predicted_outcome": self.predicted_outcome,
            "observed_outcome": self.observed_outcome,
            "reward": self.reward,
            "surprise": self.surprise,
            "success": self.success,
            "involved_persons": self.involved_persons,
            "context_tags": self.context_tags,
            "state_type": self.state_type,
        }

    @staticmethod
    def from_dict(d: Dict) -> "TransitionRecord":
        return TransitionRecord(
            tick=d.get("tick", 0),
            state_signature=d.get("state_signature", ""),
            action_kind=d.get("action_kind", ""),
            action_args=d.get("action_args", {}),
            predicted_outcome=d.get("predicted_outcome", ""),
            observed_outcome=d.get("observed_outcome", ""),
            reward=d.get("reward", 0.0),
            surprise=d.get("surprise", 0.0),
            success=d.get("success", False),
            involved_persons=d.get("involved_persons", []),
            context_tags=d.get("context_tags", []),
            state_type=d.get("state_type", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CausalEdge — weighted directed edge in the causal graph
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CausalEdge:
    """Directed edge: cause (state+action) → effect (outcome)."""

    cause: str  # state_signature + action_kind composite key
    effect: str  # observed outcome category
    context_hash: str  # hash of context tags for scoped reliability
    reliability: float  # [0, 1] — how often this transition succeeds
    support_count: int  # number of observations supporting this edge
    last_seen_tick: int  # tick of most recent observation
    avg_reward: float = 0.0  # mean reward when this edge fires
    avg_surprise: float = 0.0  # mean surprise when this edge fires

    def update(
        self, success: bool, reward: float, surprise: float, tick: int, lr: float = 0.15
    ) -> None:
        """EMA update reliability and reward from new observation."""
        outcome = 1.0 if success else 0.0
        self.reliability = self.reliability * (1 - lr) + outcome * lr
        self.avg_reward = self.avg_reward * (1 - lr) + reward * lr
        self.avg_surprise = self.avg_surprise * (1 - lr) + surprise * lr
        self.support_count += 1
        self.last_seen_tick = tick

    def to_dict(self) -> Dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "context_hash": self.context_hash,
            "reliability": self.reliability,
            "support_count": self.support_count,
            "last_seen_tick": self.last_seen_tick,
            "avg_reward": self.avg_reward,
            "avg_surprise": self.avg_surprise,
        }

    @staticmethod
    def from_dict(d: Dict) -> "CausalEdge":
        return CausalEdge(
            cause=d.get("cause", ""),
            effect=d.get("effect", ""),
            context_hash=d.get("context_hash", ""),
            reliability=d.get("reliability", 0.5),
            support_count=d.get("support_count", 0),
            last_seen_tick=d.get("last_seen_tick", 0),
            avg_reward=d.get("avg_reward", 0.0),
            avg_surprise=d.get("avg_surprise", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CausalGraph — the full causal model
# ─────────────────────────────────────────────────────────────────────────────


class CausalGraph:
    """
    Directed graph of causal relationships learned from experience.

    Nodes are compound keys: (state_signature, action_kind).
    Edges point to observed outcomes with reliability scores.

    Used by:
      - Goal evaluation: predict success probability before committing
      - Counterfactual reasoning: compare real vs hypothetical edges
      - Postmortem analysis: trace which causal chain led to failure
    """

    MAX_EDGES = 50_000
    MAX_TRANSITIONS = 20_000
    DECAY_RATE = 0.9998  # slow decay per tick on reliability

    def __init__(self) -> None:
        self._edges: Dict[str, CausalEdge] = {}  # edge_key → CausalEdge
        self._transitions: Deque[TransitionRecord] = deque(maxlen=self.MAX_TRANSITIONS)
        self._action_success: Dict[str, List[bool]] = (
            {}
        )  # action_kind → recent outcomes

    @staticmethod
    def _edge_key(state_sig: str, action: str, context_hash: str) -> str:
        return f"{state_sig}|{action}|{context_hash}"

    @staticmethod
    def _context_hash(tags: List[str]) -> str:
        if not tags:
            return "default"
        combined = "|".join(sorted(tags))
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    def record_goal_transition(self, tr: TransitionRecord) -> None:
        """Convenience: mark as goal-tier and record."""
        tr.state_type = "goal"
        self.record_transition(tr)

    def record_skill_transition(self, tr: TransitionRecord) -> None:
        """Convenience: mark as skill-tier and record."""
        tr.state_type = "skill"
        self.record_transition(tr)

    def record_transition(self, tr: TransitionRecord) -> None:
        """Record an observed transition and update causal edges."""
        self._transitions.append(tr)

        # Update action success tracking
        if tr.action_kind not in self._action_success:
            self._action_success[tr.action_kind] = []
        outcomes = self._action_success[tr.action_kind]
        outcomes.append(tr.success)
        if len(outcomes) > 100:
            self._action_success[tr.action_kind] = outcomes[-100:]

        # Update or create causal edge
        ctx_hash = self._context_hash(tr.context_tags)
        key = self._edge_key(tr.state_signature, tr.action_kind, ctx_hash)

        if key in self._edges:
            self._edges[key].update(tr.success, tr.reward, tr.surprise, tr.tick)
        else:
            if len(self._edges) >= self.MAX_EDGES:
                # Evict lowest-support edge
                worst_key = min(self._edges, key=lambda k: self._edges[k].support_count)
                del self._edges[worst_key]
            self._edges[key] = CausalEdge(
                cause=f"{tr.state_signature}|{tr.action_kind}",
                effect=tr.observed_outcome[:80],
                context_hash=ctx_hash,
                reliability=1.0 if tr.success else 0.0,
                support_count=1,
                last_seen_tick=tr.tick,
                avg_reward=tr.reward,
                avg_surprise=tr.surprise,
            )

    def predict_success(
        self, state_sig: str, action: str, context_tags: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """
        Predict (reliability, confidence) for taking action in state with context.
        confidence = how much data backs the prediction [0, 1].
        """
        ctx_hash = self._context_hash(context_tags or [])
        key = self._edge_key(state_sig, action, ctx_hash)

        edge = self._edges.get(key)
        if edge is not None:
            confidence = min(1.0, edge.support_count / 20.0)
            return edge.reliability, confidence

        # Fallback: context-free prediction
        key_default = self._edge_key(state_sig, action, "default")
        edge = self._edges.get(key_default)
        if edge is not None:
            confidence = min(1.0, edge.support_count / 20.0) * 0.7
            return edge.reliability, confidence

        # No data at all — use action-level success rate
        outcomes = self._action_success.get(action, [])
        if outcomes:
            rate = sum(outcomes) / len(outcomes)
            confidence = min(1.0, len(outcomes) / 30.0) * 0.5
            return rate, confidence

        return 0.5, 0.0  # no information

    def predict_reward(
        self, state_sig: str, action: str, context_tags: Optional[List[str]] = None
    ) -> float:
        """Predict expected reward for action in state."""
        ctx_hash = self._context_hash(context_tags or [])
        key = self._edge_key(state_sig, action, ctx_hash)
        edge = self._edges.get(key)
        if edge is not None:
            return edge.avg_reward

        key_default = self._edge_key(state_sig, action, "default")
        edge = self._edges.get(key_default)
        if edge is not None:
            return edge.avg_reward
        return 0.0

    def best_action(
        self,
        state_sig: str,
        candidate_actions: List[str],
        context_tags: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """Choose the action with highest expected utility (reliability * reward)."""
        best_act = candidate_actions[0] if candidate_actions else ""
        best_score = -999.0

        for action in candidate_actions:
            rel, conf = self.predict_success(state_sig, action, context_tags)
            rew = self.predict_reward(state_sig, action, context_tags)
            score = rel * 0.6 + rew * 0.3 + conf * 0.1
            if score > best_score:
                best_score = score
                best_act = action

        return best_act, best_score

    def recent_transitions(self, n: int = 10) -> List[TransitionRecord]:
        return list(self._transitions)[-n:]

    def predict_goal_success(
        self, state_sig: str, action: str, context_tags: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """Predict success using only goal-tier edges, with skill-tier fallback."""
        # Try goal-level edges first
        rel, conf = self._predict_tier(state_sig, action, context_tags, "goal")
        if conf > 0.05:
            return rel, conf
        # Fallback: any-tier (backward compat)
        return self.predict_success(state_sig, action, context_tags)

    def predict_skill_success(
        self, state_sig: str, action: str, context_tags: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """Predict success using only skill-tier edges."""
        rel, conf = self._predict_tier(state_sig, action, context_tags, "skill")
        if conf > 0.05:
            return rel, conf
        return self.predict_success(state_sig, action, context_tags)

    def _predict_tier(
        self, state_sig: str, action: str, context_tags: Optional[List[str]], tier: str
    ) -> Tuple[float, float]:
        """Filter edges by state_type prefix in their cause field."""
        ctx_hash = self._context_hash(context_tags or [])
        key = self._edge_key(state_sig, action, ctx_hash)
        edge = self._edges.get(key)
        if edge is not None:
            # Check tier from the stored transition history
            if self._edge_matches_tier(key, tier):
                confidence = min(1.0, edge.support_count / 20.0)
                return edge.reliability, confidence
        # Fallback: default context
        key_default = self._edge_key(state_sig, action, "default")
        edge = self._edges.get(key_default)
        if edge is not None and self._edge_matches_tier(key_default, tier):
            confidence = min(1.0, edge.support_count / 20.0) * 0.7
            return edge.reliability, confidence
        return 0.5, 0.0

    def _edge_matches_tier(self, edge_key: str, tier: str) -> bool:
        """Check if an edge's cause field matches the expected tier prefix."""
        edge = self._edges.get(edge_key)
        if edge is None:
            return False
        cause = edge.cause
        if tier == "goal":
            return cause.startswith("goal:") or cause.startswith("g:")
        elif tier == "skill":
            return cause.startswith("policy:") or (
                not cause.startswith("goal:") and not cause.startswith("g:")
            )
        return True  # no filter

    def aggregate_skill_to_goal(self, min_observations: int = 5) -> None:
        """Periodically derive goal-level heuristics from accumulated skill data.

        For each unique goal intent that appears in skill-tier transitions,
        compute aggregate success rate / reward and write a synthetic
        goal-tier edge so that _evaluate_goal benefits from skill experience.
        """
        # Collect skill-tier transitions grouped by goal intent
        from collections import defaultdict

        goal_stats: Dict[str, List[Tuple[bool, float]]] = defaultdict(list)
        for tr in self._transitions:
            if tr.state_type != "skill":
                continue
            # Extract goal intent from state_signature  "policy:<intent>:<person>"
            parts = tr.state_signature.split(":")
            if len(parts) >= 2:
                intent = parts[1]
                goal_stats[intent].append((tr.success, tr.reward))

        for intent, data in goal_stats.items():
            if len(data) < min_observations:
                continue
            agg_success = sum(1 for ok, _ in data if ok) / len(data)
            agg_reward = sum(r for _, r in data) / len(data)
            # Build synthetic goal-tier edge
            synth_sig = f"goal:{intent}"
            synth_key = self._edge_key(synth_sig, intent, "aggregated")
            if synth_key in self._edges:
                e = self._edges[synth_key]
                lr = 0.2
                e.reliability = e.reliability * (1 - lr) + agg_success * lr
                e.avg_reward = e.avg_reward * (1 - lr) + agg_reward * lr
                e.support_count += len(data)
            else:
                self._edges[synth_key] = CausalEdge(
                    cause=f"{synth_sig}|{intent}",
                    effect="aggregated_from_skills",
                    context_hash="aggregated",
                    reliability=agg_success,
                    support_count=len(data),
                    last_seen_tick=data[-1][1] if data else 0,
                    avg_reward=agg_reward,
                )

    def action_reliability(self, action: str) -> float:
        """Overall reliability for an action across all states."""
        outcomes = self._action_success.get(action, [])
        if not outcomes:
            return 0.5
        return sum(outcomes) / len(outcomes)

    def decay(self) -> None:
        """Slowly decay unused edges."""
        to_remove = []
        for key, edge in self._edges.items():
            edge.reliability *= self.DECAY_RATE
            if edge.reliability < 0.01 and edge.support_count < 3:
                to_remove.append(key)
        for key in to_remove:
            del self._edges[key]

    def summarise(self, top_n: int = 5) -> List[str]:
        """Return human-readable summaries of strongest causal edges."""
        ranked = sorted(
            self._edges.values(),
            key=lambda e: e.reliability * e.support_count,
            reverse=True,
        )
        out = []
        for edge in ranked[:top_n]:
            out.append(
                f"{edge.cause} → {edge.effect} "
                f"(rel={edge.reliability:.2f}, n={edge.support_count}, "
                f"rew={edge.avg_reward:+.2f})"
            )
        return out

    def to_dict(self) -> Dict:
        return {
            "edges": {k: e.to_dict() for k, e in self._edges.items()},
            "transitions": [t.to_dict() for t in list(self._transitions)[-500:]],
            "action_success": {k: v[-50:] for k, v in self._action_success.items()},
        }

    def from_dict(self, data: Dict) -> None:
        for key, ed in data.get("edges", {}).items():
            self._edges[key] = CausalEdge.from_dict(ed)
        for td in data.get("transitions", []):
            self._transitions.append(TransitionRecord.from_dict(td))
        for action, outcomes in data.get("action_success", {}).items():
            self._action_success[action] = outcomes
