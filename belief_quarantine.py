"""
belief_quarantine.py — Evidence-Based Belief Management

Extends the epistemic model with:
  • BeliefVersion: timestamped snapshots of a belief's confidence + evidence
  • BeliefQuarantine: holds beliefs below a confidence threshold in quarantine
    until sufficient evidence promotes or discards them
  • ConflictDetector: finds logical tensions between beliefs

Integration:
  - consciousness.py: BeliefStore gains quarantine and conflict detection
  - persistence.py: serialises quarantine state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# BeliefVersion — immutable snapshot of a belief at a point in time
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BeliefVersion:
    tick: int
    confidence: float  # [0, 1]
    source: str  # "observation", "inference", "testimony", "memory"
    evidence_summary: str  # brief text describing supporting evidence
    evidence_count: int
    contradictions: int


# ─────────────────────────────────────────────────────────────────────────────
# QuarantinedBelief — belief under review
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QuarantinedBelief:
    topic: str
    claim: str
    initial_tick: int
    versions: List[BeliefVersion] = field(default_factory=list)
    status: str = "quarantined"  # quarantined, promoted, discarded
    resolution_tick: int = 0
    resolution_reason: str = ""

    @property
    def latest_confidence(self) -> float:
        if self.versions:
            return self.versions[-1].confidence
        return 0.0

    @property
    def evidence_count(self) -> int:
        if self.versions:
            return self.versions[-1].evidence_count
        return 0

    def add_evidence(
        self,
        tick: int,
        confidence: float,
        source: str,
        evidence_summary: str,
        evidence_count: int,
        contradictions: int,
    ) -> None:
        v = BeliefVersion(
            tick, confidence, source, evidence_summary, evidence_count, contradictions
        )
        self.versions.append(v)
        if len(self.versions) > 20:
            self.versions = self.versions[-20:]

    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "claim": self.claim[:120],
            "initial_tick": self.initial_tick,
            "versions": [
                {
                    "tick": v.tick,
                    "confidence": v.confidence,
                    "source": v.source,
                    "evidence_summary": v.evidence_summary[:80],
                    "evidence_count": v.evidence_count,
                    "contradictions": v.contradictions,
                }
                for v in self.versions[-10:]
            ],
            "status": self.status,
            "resolution_tick": self.resolution_tick,
            "resolution_reason": self.resolution_reason,
        }

    @staticmethod
    def from_dict(d: Dict) -> "QuarantinedBelief":
        qb = QuarantinedBelief(
            topic=d.get("topic", ""),
            claim=d.get("claim", ""),
            initial_tick=d.get("initial_tick", 0),
            status=d.get("status", "quarantined"),
            resolution_tick=d.get("resolution_tick", 0),
            resolution_reason=d.get("resolution_reason", ""),
        )
        for vd in d.get("versions", []):
            qb.versions.append(
                BeliefVersion(
                    tick=vd["tick"],
                    confidence=vd["confidence"],
                    source=vd.get("source", "unknown"),
                    evidence_summary=vd.get("evidence_summary", ""),
                    evidence_count=vd.get("evidence_count", 0),
                    contradictions=vd.get("contradictions", 0),
                )
            )
        return qb


# ─────────────────────────────────────────────────────────────────────────────
# ConflictRecord — detected tension between two beliefs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConflictRecord:
    topic_a: str
    topic_b: str
    description: str
    severity: float  # [0, 1]
    detected_tick: int
    resolved: bool = False
    resolution: str = ""

    def to_dict(self) -> Dict:
        return {
            "topic_a": self.topic_a,
            "topic_b": self.topic_b,
            "description": self.description[:120],
            "severity": self.severity,
            "detected_tick": self.detected_tick,
            "resolved": self.resolved,
            "resolution": self.resolution[:80],
        }

    @staticmethod
    def from_dict(d: Dict) -> "ConflictRecord":
        return ConflictRecord(
            topic_a=d.get("topic_a", ""),
            topic_b=d.get("topic_b", ""),
            description=d.get("description", ""),
            severity=d.get("severity", 0.0),
            detected_tick=d.get("detected_tick", 0),
            resolved=d.get("resolved", False),
            resolution=d.get("resolution", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# BeliefQuarantine — evidence-gated belief management
# ─────────────────────────────────────────────────────────────────────────────


class BeliefQuarantine:
    """
    Manages beliefs that have insufficient evidence.

    Beliefs below PROMOTE_THRESHOLD stay quarantined.
    When evidence accumulates above PROMOTE_THRESHOLD they are promoted.
    When contradictions dominate they are discarded.
    """

    PROMOTE_THRESHOLD = 0.65
    DISCARD_THRESHOLD = 0.15
    MAX_QUARANTINED = 200
    MAX_CONFLICTS = 50
    CHECK_INTERVAL = 500  # ticks between reviews

    def __init__(self) -> None:
        self._quarantine: Dict[str, QuarantinedBelief] = {}
        self._conflicts: List[ConflictRecord] = []
        self._last_review_tick: int = 0

    # ── Ingestion ───────────────────────────────────────────

    def submit(
        self,
        tick: int,
        topic: str,
        claim: str,
        confidence: float,
        source: str,
        evidence_summary: str = "",
    ) -> str:
        """
        Submit a belief for quarantine evaluation.
        Returns: "promoted", "quarantined", "discarded".
        """
        if confidence >= self.PROMOTE_THRESHOLD:
            return "promoted"  # skip quarantine entirely

        if confidence <= self.DISCARD_THRESHOLD:
            return "discarded"

        key = topic
        if key not in self._quarantine:
            if len(self._quarantine) >= self.MAX_QUARANTINED:
                self._evict_oldest()
            self._quarantine[key] = QuarantinedBelief(
                topic=topic, claim=claim, initial_tick=tick
            )

        qb = self._quarantine[key]
        if qb.status != "quarantined":
            return qb.status  # already resolved

        qb.add_evidence(
            tick, confidence, source, evidence_summary, qb.evidence_count + 1, 0
        )
        return "quarantined"

    def add_evidence(
        self,
        topic: str,
        tick: int,
        confidence: float,
        source: str,
        evidence_summary: str = "",
        is_contradiction: bool = False,
    ) -> None:
        """Add evidence for or against a quarantined belief."""
        qb = self._quarantine.get(topic)
        if qb is None or qb.status != "quarantined":
            return
        prev = qb.versions[-1] if qb.versions else None
        e_count = (prev.evidence_count + 1) if prev else 1
        c_count = (
            (prev.contradictions + 1)
            if is_contradiction
            else (prev.contradictions if prev else 0)
        )
        qb.add_evidence(tick, confidence, source, evidence_summary, e_count, c_count)

    # ── Periodic review ─────────────────────────────────────

    def review(self, tick: int) -> List[Tuple[str, str]]:
        """
        Periodic review of quarantined beliefs.
        Returns list of (topic, new_status) for any status changes.
        """
        if tick - self._last_review_tick < self.CHECK_INTERVAL:
            return []
        self._last_review_tick = tick

        changes: List[Tuple[str, str]] = []
        for key, qb in list(self._quarantine.items()):
            if qb.status != "quarantined":
                continue
            conf = qb.latest_confidence
            if conf >= self.PROMOTE_THRESHOLD:
                qb.status = "promoted"
                qb.resolution_tick = tick
                qb.resolution_reason = f"confidence reached {conf:.2f}"
                changes.append((key, "promoted"))
            elif conf <= self.DISCARD_THRESHOLD:
                qb.status = "discarded"
                qb.resolution_tick = tick
                qb.resolution_reason = f"confidence dropped to {conf:.2f}"
                changes.append((key, "discarded"))
            elif qb.versions:
                latest = qb.versions[-1]
                if latest.contradictions > latest.evidence_count * 0.6:
                    qb.status = "discarded"
                    qb.resolution_tick = tick
                    qb.resolution_reason = "too many contradictions"
                    changes.append((key, "discarded"))
        return changes

    # ── Conflict detection ──────────────────────────────────

    def register_conflict(
        self, tick: int, topic_a: str, topic_b: str, description: str, severity: float
    ) -> None:
        """Register a detected conflict between two beliefs."""
        self._conflicts.append(
            ConflictRecord(
                topic_a=topic_a,
                topic_b=topic_b,
                description=description,
                severity=severity,
                detected_tick=tick,
            )
        )
        if len(self._conflicts) > self.MAX_CONFLICTS:
            self._conflicts = self._conflicts[-self.MAX_CONFLICTS :]

    def unresolved_conflicts(self) -> List[ConflictRecord]:
        return [c for c in self._conflicts if not c.resolved]

    def resolve_conflict(self, topic_a: str, topic_b: str, resolution: str) -> None:
        for c in self._conflicts:
            if c.topic_a == topic_a and c.topic_b == topic_b and not c.resolved:
                c.resolved = True
                c.resolution = resolution
                break

    # ── Queries ──────────────────────────────────────────────

    def quarantined_count(self) -> int:
        return sum(1 for qb in self._quarantine.values() if qb.status == "quarantined")

    def summary(self) -> str:
        q = self.quarantined_count()
        p = sum(1 for qb in self._quarantine.values() if qb.status == "promoted")
        d = sum(1 for qb in self._quarantine.values() if qb.status == "discarded")
        uc = len(self.unresolved_conflicts())
        return f"Beliefs: {q} quarantined, {p} promoted, {d} discarded, {uc} conflicts"

    # ── Eviction ─────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        resolved = [k for k, v in self._quarantine.items() if v.status != "quarantined"]
        if resolved:
            del self._quarantine[resolved[0]]
        else:
            oldest = min(
                self._quarantine, key=lambda k: self._quarantine[k].initial_tick
            )
            del self._quarantine[oldest]

    # ── Serialisation ────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "quarantine": {
                k: v.to_dict() for k, v in list(self._quarantine.items())[:200]
            },
            "conflicts": [c.to_dict() for c in self._conflicts[-50:]],
            "last_review_tick": self._last_review_tick,
        }

    def from_dict(self, data: Dict) -> None:
        self._quarantine.clear()
        for k, v in data.get("quarantine", {}).items():
            self._quarantine[k] = QuarantinedBelief.from_dict(v)
        self._conflicts = [
            ConflictRecord.from_dict(c) for c in data.get("conflicts", [])
        ]
        self._last_review_tick = data.get("last_review_tick", 0)
