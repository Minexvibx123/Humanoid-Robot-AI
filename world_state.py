"""
world_state.py — Structured World Model (Schicht A output)

Maintains a persistent, typed representation of the external world:
  • Tracked persons (ID, position, face, gesture, speaking, distance)
  • Detected objects (label, position, confidence, age)
  • Spatial zones (interaction distances, personal space)
  • Temporal persistence (objects/persons decay if not re-detected)
  • Predictive expectations (distance/speaking/gesture dynamics per entity)
  • Prediction error signals (surprise when reality deviates from model)

Replaces raw string injection into the neural pathway with structured
data that feeds consciousness, executive, emotion, and episodic memory.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from telemetry_bus import (
    EVENT_FACE_DETECTED,
    EVENT_GESTURE_DETECTED,
    EVENT_OBJECT_DETECTED,
    EVENT_PERSON_LOST,
    EVENT_PERSON_SEEN,
    EVENT_SPEAKER_ACTIVE,
    SensorEvent,
)

# ─────────────────────────────────────────────────────────────
# Prediction model per entity
# ─────────────────────────────────────────────────────────────


@dataclass
class EntityPrediction:
    """EMA-based expectation for a single tracked entity."""

    expected_distance_cm: float = 999.0
    expected_center_x: float = 0.5
    expected_center_y: float = 0.5
    expected_speaking: float = 0.0  # probability [0,1]
    expected_gesture: str = "none"
    velocity_cm: float = 0.0  # distance change per tick (EMA)
    velocity_x: float = 0.0  # pixel velocity x
    velocity_y: float = 0.0  # pixel velocity y
    # Accumulated prediction errors
    distance_error: float = 0.0
    position_error: float = 0.0
    speaking_error: float = 0.0
    presence_error: float = 0.0  # entity disappeared unexpectedly
    total_surprise: float = 0.0  # composite surprise [0,1]
    _update_count: int = 0

    _EMA_FAST: float = 0.25  # fast adaptation
    _EMA_SLOW: float = 0.08  # slow expectations

    def update_from_observation(
        self, distance_cm: float, cx: float, cy: float, speaking: bool, gesture: str
    ) -> float:
        """Update predictions from new observation. Returns surprise [0,1]."""
        self._update_count += 1
        alpha = self._EMA_FAST if self._update_count < 10 else self._EMA_SLOW

        # Distance prediction error
        self.distance_error = abs(distance_cm - self.expected_distance_cm)
        d_vel = distance_cm - self.expected_distance_cm
        self.velocity_cm = self.velocity_cm * 0.85 + d_vel * 0.15

        # Position prediction error
        dx = cx - self.expected_center_x
        dy = cy - self.expected_center_y
        self.position_error = math.sqrt(dx * dx + dy * dy)
        self.velocity_x = self.velocity_x * 0.85 + dx * 0.15
        self.velocity_y = self.velocity_y * 0.85 + dy * 0.15

        # Speaking prediction error
        sp_val = 1.0 if speaking else 0.0
        self.speaking_error = abs(sp_val - self.expected_speaking)

        # Update expectations (EMA toward observation)
        self.expected_distance_cm = (
            self.expected_distance_cm * (1 - alpha)
            + (distance_cm + self.velocity_cm) * alpha
        )
        self.expected_center_x = (
            self.expected_center_x * (1 - alpha) + (cx + self.velocity_x) * alpha
        )
        self.expected_center_y = (
            self.expected_center_y * (1 - alpha) + (cy + self.velocity_y) * alpha
        )
        self.expected_speaking = self.expected_speaking * (1 - alpha) + sp_val * alpha
        self.expected_gesture = gesture

        # Composite surprise
        norm_dist = min(1.0, self.distance_error / 80.0)
        norm_pos = min(1.0, self.position_error / 0.15)
        self.total_surprise = (
            norm_dist * 0.4
            + norm_pos * 0.3
            + self.speaking_error * 0.2
            + self.presence_error * 0.1
        )
        self.presence_error *= 0.9  # decay presence error
        return self.total_surprise

    def mark_disappeared(self) -> None:
        """Entity was expected but not observed."""
        self.presence_error = min(1.0, self.presence_error + 0.5)
        self.total_surprise = min(1.0, self.total_surprise + 0.3)

    def predict_next(self) -> Tuple[float, float, float]:
        """Return (predicted_distance, predicted_cx, predicted_cy)."""
        return (
            max(10.0, self.expected_distance_cm + self.velocity_cm),
            max(0.0, min(1.0, self.expected_center_x + self.velocity_x)),
            max(0.0, min(1.0, self.expected_center_y + self.velocity_y)),
        )


# ─────────────────────────────────────────────────────────────
# Tracked entities
# ─────────────────────────────────────────────────────────────


@dataclass
class TrackedPerson:
    """Persistent representation of one observed human."""

    person_id: str = "person_0"
    center_x: float = 0.5  # normalised image coords [0,1]
    center_y: float = 0.5
    area: float = 0.0  # bounding box area (normalised)
    distance_cm: float = 999.0  # estimated distance
    gesture: str = "none"
    speaking: bool = False
    face_visible: bool = False
    last_seen_tick: int = 0
    first_seen_tick: int = 0
    interaction_count: int = 0  # how many interactions with this person

    # ── Perception fields added for Domain H ──────────────────
    # Gaze direction relative to robot: "toward" | "away" | "unknown"
    gaze_direction: str = "unknown"
    # Prosodic affect inferred from audio features: "calm"|"excited"|"tense"|"sad"|"unknown"
    speech_affect: str = "unknown"
    # Normalised speech energy EMA [0,1]; updated by sensors.py AudioEncoder
    speech_energy: float = 0.0
    # Speaking rate variability EMA [0,1]; high = animated / low = flat
    speech_tempo_var: float = 0.0

    @property
    def engagement_score(self) -> float:
        """How engaged this person appears [0,1]."""
        s = 0.3 if self.face_visible else 0.0
        s += 0.3 if self.speaking else 0.0
        s += 0.2 if self.gesture not in ("none", "") else 0.0
        s += min(0.2, self.area * 2.0)
        return min(1.0, s)

    @property
    def zone(self) -> str:
        if self.distance_cm < 30:
            return "intimate"
        if self.distance_cm < 60:
            return "personal"
        if self.distance_cm < 120:
            return "social"
        return "public"


@dataclass
class TrackedObject:
    """Persistent representation of a detected object."""

    label: str = "unknown"
    center_x: float = 0.5
    center_y: float = 0.5
    area: float = 0.0
    confidence: float = 0.0
    last_seen_tick: int = 0
    first_seen_tick: int = 0


# ─────────────────────────────────────────────────────────────
# Interaction zone summary
# ─────────────────────────────────────────────────────────────


@dataclass
class InteractionZone:
    """Summary of the immediate social/physical environment."""

    nearest_person_distance_cm: float = 999.0
    nearest_person_id: str = "none"
    n_persons_visible: int = 0
    n_objects_visible: int = 0
    dominant_gesture: str = "none"
    anyone_speaking: bool = False
    zone_label: str = "public"  # intimate/personal/social/public


# ─────────────────────────────────────────────────────────────
# World predicates (for state-based planning)
# ─────────────────────────────────────────────────────────────


@dataclass
class WorldPredicates:
    """Boolean/scalar predicates about the world state.
    Used by the state-space planner to match preconditions and effects."""

    person_visible: bool = False
    person_attentive: bool = False
    person_speaking: bool = False
    hand_free_left: bool = True
    hand_free_right: bool = True
    holds_object_left: bool = False
    holds_object_right: bool = False
    distance_safe: bool = True
    object_reachable: bool = False
    head_aligned: bool = False
    body_idle: bool = True
    gesture_active: bool = False
    robot_speaking: bool = False
    greeting_done: bool = False
    # Scalar context
    nearest_person_cm: float = 999.0
    engagement_level: float = 0.0


# ─────────────────────────────────────────────────────────────
# World State
# ─────────────────────────────────────────────────────────────


@dataclass
class WorldPredictionSummary:
    """Aggregate prediction error summary for consciousness consumption."""

    mean_surprise: float = 0.0
    max_surprise: float = 0.0
    max_surprise_entity: str = ""
    unexpected_appearances: int = 0
    unexpected_departures: int = 0
    speaking_switches: int = 0
    semantic_labels: List[str] = field(default_factory=list)

    @property
    def is_surprising(self) -> bool:
        return self.max_surprise > 0.3


class WorldState:
    """
    Persistent, structured model of the observable world.
    Updated each tick from telemetry events.
    Consumed by executive, consciousness, emotion, social manager.
    """

    PERSON_TIMEOUT_TICKS = 60  # ticks before a person is considered lost
    OBJECT_TIMEOUT_TICKS = 120  # ticks before an object decays

    def __init__(self) -> None:
        self.persons: Dict[str, TrackedPerson] = {}
        self.objects: Dict[str, TrackedObject] = {}
        self.zone = InteractionZone()
        self.predicates = WorldPredicates()
        self._next_person_id: int = 0
        # Prediction layer
        self._predictions: Dict[str, EntityPrediction] = {}
        self.prediction_summary = WorldPredictionSummary()
        self._prev_speaker_id: Optional[str] = None
        # Reference resolution: tracks last-mentioned entity for pronoun resolution
        self._last_mentioned_entity: Optional[str] = None

    # ── Event processing ─────────────────────────────────────

    def process_event(self, event: SensorEvent) -> None:
        """Integrate a single sensor event into world state."""
        if event.kind in (EVENT_PERSON_SEEN, EVENT_FACE_DETECTED):
            self._update_person(event)
        elif event.kind == EVENT_PERSON_LOST:
            pass  # handled by decay
        elif event.kind == EVENT_GESTURE_DETECTED:
            self._update_gesture(event)
        elif event.kind == EVENT_SPEAKER_ACTIVE:
            self._update_speaker(event)
        elif event.kind == EVENT_OBJECT_DETECTED:
            self._update_object(event)

    def process_events(self, events: List[SensorEvent]) -> None:
        for e in events:
            self.process_event(e)

    # ── Tick (decay + zone update) ───────────────────────────

    def tick(self, current_tick: int) -> InteractionZone:
        """Decay old entities, compute prediction errors, recompute zone."""
        # ── 1. Compute prediction errors BEFORE decay ────────
        self._compute_prediction_errors(current_tick)

        # ── 2. Decay persons ─────────────────────────────────
        lost = [
            pid
            for pid, p in self.persons.items()
            if (current_tick - p.last_seen_tick) > self.PERSON_TIMEOUT_TICKS
        ]
        for pid in lost:
            del self.persons[pid]
            # keep prediction for quick surprise if they reappear
            if pid in self._predictions:
                self._predictions[pid].mark_disappeared()

        # Decay objects
        lost_obj = [
            oid
            for oid, o in self.objects.items()
            if (current_tick - o.last_seen_tick) > self.OBJECT_TIMEOUT_TICKS
        ]
        for oid in lost_obj:
            del self.objects[oid]

        # Clean stale predictions (long gone)
        stale = [
            k
            for k in self._predictions
            if k not in self.persons
            and k not in self.objects
            and self._predictions[k]._update_count > 0
            and self._predictions[k].total_surprise < 0.05
        ]
        for k in stale:
            del self._predictions[k]

        # Recompute zone
        self.zone.n_persons_visible = len(self.persons)
        self.zone.n_objects_visible = len(self.objects)
        self.zone.anyone_speaking = any(p.speaking for p in self.persons.values())
        gestures = [
            p.gesture for p in self.persons.values() if p.gesture not in ("none", "")
        ]
        self.zone.dominant_gesture = gestures[0] if gestures else "none"

        if self.persons:
            nearest = min(self.persons.values(), key=lambda p: p.distance_cm)
            self.zone.nearest_person_distance_cm = nearest.distance_cm
            self.zone.nearest_person_id = nearest.person_id
            self.zone.zone_label = nearest.zone
        else:
            self.zone.nearest_person_distance_cm = 999.0
            self.zone.nearest_person_id = "none"
            self.zone.zone_label = "public"

        # Recompute planning predicates
        self._compute_predicates()

        return self.zone

    def _compute_predicates(self, body=None) -> None:
        """Update boolean predicates from current world + zone state."""
        p = self.predicates
        p.person_visible = self.zone.n_persons_visible > 0
        p.person_speaking = self.zone.anyone_speaking
        p.gesture_active = self.zone.dominant_gesture not in ("none", "")
        p.nearest_person_cm = self.zone.nearest_person_distance_cm
        p.distance_safe = self.zone.nearest_person_distance_cm > 30.0

        # Person attentive = visible + engaged (face visible + close)
        engaged = self.most_engaged_person()
        if engaged:
            p.person_attentive = engaged.engagement_score > 0.4
            p.engagement_level = engaged.engagement_score
        else:
            p.person_attentive = False
            p.engagement_level = 0.0

        # Object reachable = any object in near zone
        p.object_reachable = any(o.confidence > 0.3 for o in self.objects.values())

        # Body state predicates are set externally by brain tick
        # (hand_free, holds_object, head_aligned, body_idle set from body_schema)

    def update_body_predicates(self, body) -> None:
        """Update predicates that depend on body state (called from brain tick)."""
        p = self.predicates
        if hasattr(body, "summary"):
            s = body.summary
            p.hand_free_left = s.left_hand_open
            p.hand_free_right = s.right_hand_open
            p.holds_object_left = not s.left_hand_open
            p.holds_object_right = not s.right_hand_open
            p.head_aligned = s.overall_error < 0.1
            p.body_idle = s.body_pose == "idle"

    # ── Queries ──────────────────────────────────────────────

    def nearest_person(self) -> Optional[TrackedPerson]:
        if not self.persons:
            return None
        return min(self.persons.values(), key=lambda p: p.distance_cm)

    def speaking_person(self) -> Optional[TrackedPerson]:
        for p in self.persons.values():
            if p.speaking:
                return p
        return None

    def most_engaged_person(self) -> Optional[TrackedPerson]:
        if not self.persons:
            return None
        return max(self.persons.values(), key=lambda p: p.engagement_score)

    # ── Reference resolution & salience ──────────────────────

    # Pronoun/demonstrative clusters (DE + EN)
    _PERSON_REFS = frozenset(
        {
            "er", "ihn", "ihm", "she", "her", "he", "him",
            "sie", "ihm", "the person", "die person",
        }
    )
    _OBJECT_REFS = frozenset(
        {
            "das", "dies", "dieses", "jenes", "it", "this", "that",
            "the thing", "das ding", "das objekt",
        }
    )
    _SPATIAL_REFS = frozenset(
        {"dort", "da", "here", "there", "drüben", "over there", "dahinter"}
    )
    _PLURAL_REFS = frozenset({"sie alle", "die", "they", "them", "those", "diese"})

    def compute_salience(
        self,
        current_tick: int,
        topic_tokens: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Score all currently tracked entities by salience [0, 1].

        Salience factors:
          • recency   — how recently seen (decays over ticks)
          • proximity — closer = more salient
          • speaking  — actively speaking person is maximally salient
          • gesture   — active gesture raises salience
          • topic     — entity label overlaps with current topic tokens
        """
        scores: Dict[str, float] = {}
        topic_set = set(t.lower() for t in (topic_tokens or []) if len(t) > 2)

        for pid, p in self.persons.items():
            age = current_tick - p.last_seen_tick
            recency = max(0.0, 1.0 - age / max(self.PERSON_TIMEOUT_TICKS, 1))
            proximity = max(0.0, 1.0 - p.distance_cm / 300.0)
            speaking_bonus = 0.4 if p.speaking else 0.0
            gesture_bonus = 0.15 if p.gesture not in ("none", "") else 0.0
            topic_bonus = 0.2 if (topic_set and any(
                t in str(pid).lower() or t in str(getattr(p, "name", "")).lower()
                for t in topic_set
            )) else 0.0
            engagement_bonus = p.engagement_score * 0.1
            score = (
                recency * 0.3
                + proximity * 0.25
                + speaking_bonus
                + gesture_bonus
                + topic_bonus
                + engagement_bonus
            )
            scores[f"person:{pid}"] = min(1.0, score)

        for oid, obj in self.objects.items():
            age = current_tick - obj.last_seen_tick
            recency = max(0.0, 1.0 - age / max(self.OBJECT_TIMEOUT_TICKS, 1))
            proximity = max(0.0, 1.0 - (getattr(obj, "center_x", 0.5) * 200) / 300.0)
            conf_bonus = obj.confidence * 0.2
            topic_bonus = 0.3 if (topic_set and any(
                t in obj.label.lower() for t in topic_set
            )) else 0.0
            score = recency * 0.4 + proximity * 0.2 + conf_bonus + topic_bonus
            scores[f"object:{oid}"] = min(1.0, score)

        # Boost last-mentioned entity
        _lm = getattr(self, "_last_mentioned_entity", None)
        if _lm and _lm in scores:
            scores[_lm] = min(1.0, scores[_lm] + 0.25)

        return scores

    def most_salient_entity(
        self,
        current_tick: int,
        entity_type: str = "any",
        topic_tokens: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Return the key of the most salient entity of the given type.

        entity_type: "person" | "object" | "any"
        Returns a key like "person:person_0" or "object:cup", or None.
        """
        scores = self.compute_salience(current_tick, topic_tokens)
        if not scores:
            return None
        if entity_type == "person":
            filtered = {k: v for k, v in scores.items() if k.startswith("person:")}
        elif entity_type == "object":
            filtered = {k: v for k, v in scores.items() if k.startswith("object:")}
        else:
            filtered = scores
        if not filtered:
            return None
        return max(filtered, key=lambda k: filtered[k])

    def resolve_reference(
        self,
        phrase: str,
        current_tick: int,
        topic_tokens: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Resolve a pronoun or demonstrative phrase to an entity key.

        Returns a key like "person:person_0" or "object:cup", or None
        if no plausible referent is found.

        Call note_mentioned() after using the result so future references
        can chain correctly ("er" → "the person I just mentioned").
        """
        ph = phrase.strip().lower()
        tokens = set(ph.split())

        if tokens & self._PERSON_REFS:
            return self.most_salient_entity(current_tick, "person", topic_tokens)
        if tokens & self._OBJECT_REFS:
            return self.most_salient_entity(current_tick, "object", topic_tokens)
        if tokens & self._PLURAL_REFS:
            # Return the most salient entity of any type
            return self.most_salient_entity(current_tick, "any", topic_tokens)
        if tokens & self._SPATIAL_REFS:
            # Spatial reference: nearest entity in any zone
            return self.most_salient_entity(current_tick, "any", topic_tokens)

        # Not a standard pronoun: try label-based lookup
        for oid in self.objects:
            if ph in oid.lower() or oid.lower() in ph:
                return f"object:{oid}"
        for pid in self.persons:
            pname = getattr(self.persons[pid], "name", "") or ""
            if ph in str(pid).lower() or (pname and ph in pname.lower()):
                return f"person:{pid}"

        return None

    def note_mentioned(self, entity_key: str) -> None:
        """Record that this entity was recently referenced in discourse.

        Used by dialogue_manager to boost salience of just-mentioned
        entities for subsequent pronoun resolution.
        """
        self._last_mentioned_entity = entity_key

    def entity_label(self, entity_key: str) -> str:
        """Return a human-readable label for an entity key."""
        if entity_key.startswith("person:"):
            pid = entity_key[len("person:"):]
            p = self.persons.get(pid)
            if p is not None:
                name = getattr(p, "name", None)
                return name if name else f"person {pid}"
            return pid
        if entity_key.startswith("object:"):
            oid = entity_key[len("object:"):]
            return self.objects[oid].label if oid in self.objects else oid
        return entity_key

    def describe(self) -> str:
        z = self.zone
        persons = (
            ", ".join(
                f"{p.person_id}({p.zone},{p.gesture})" for p in self.persons.values()
            )
            or "none"
        )
        objects = ", ".join(f"{o.label}" for o in self.objects.values()) or "none"
        return (
            f"world zone={z.zone_label} persons={z.n_persons_visible}[{persons}] "
            f"objects={z.n_objects_visible}[{objects}] "
            f"nearest={z.nearest_person_id}@{z.nearest_person_distance_cm:.0f}cm "
            f"speaking={int(z.anyone_speaking)} gesture={z.dominant_gesture}"
        )

    # ── Internal updates ─────────────────────────────────────

    def _update_person(self, event: SensorEvent) -> None:
        cx = float(event.data.get("center_x", 0.5))
        cy = float(event.data.get("center_y", 0.5))
        area = float(event.data.get("width", 0.1)) * float(
            event.data.get("height", 0.1)
        )

        # Match to existing person by position proximity
        best_id = None
        best_dist = 0.15  # max distance to match
        for pid, p in self.persons.items():
            d = ((cx - p.center_x) ** 2 + (cy - p.center_y) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id = pid

        if best_id is None:
            best_id = f"person_{self._next_person_id}"
            self._next_person_id += 1
            self.persons[best_id] = TrackedPerson(
                person_id=best_id,
                first_seen_tick=event.tick,
            )

        p = self.persons[best_id]
        p.center_x = cx
        p.center_y = cy
        p.area = area
        p.last_seen_tick = event.tick
        p.face_visible = event.kind == EVENT_FACE_DETECTED or p.face_visible
        # Rough distance estimate from face area (calibrated for InMoov camera)
        if area > 0.001:
            p.distance_cm = min(500.0, max(15.0, 35.0 / (area**0.5)))

    def _update_gesture(self, event: SensorEvent) -> None:
        gesture = str(event.data.get("gesture", "none"))
        # Apply to nearest person or most recent
        if self.persons:
            nearest = min(self.persons.values(), key=lambda p: p.distance_cm)
            nearest.gesture = gesture
            nearest.last_seen_tick = event.tick

    def _update_speaker(self, event: SensorEvent) -> None:
        # Mark nearest face-visible person as speaking
        candidates = [p for p in self.persons.values() if p.face_visible]
        if not candidates:
            candidates = list(self.persons.values())
        if candidates:
            nearest = min(candidates, key=lambda p: p.distance_cm)
            # Reset all speaking flags first
            for p in self.persons.values():
                p.speaking = False
            nearest.speaking = True
            nearest.interaction_count += 1
            nearest.last_seen_tick = event.tick

    def _update_object(self, event: SensorEvent) -> None:
        label = str(event.data.get("label", "unknown"))
        cx = float(event.data.get("center_x", 0.5))
        cy = float(event.data.get("center_y", 0.5))

        if label not in self.objects:
            self.objects[label] = TrackedObject(
                label=label,
                first_seen_tick=event.tick,
            )
        obj = self.objects[label]
        obj.center_x = cx
        obj.center_y = cy
        obj.confidence = float(event.data.get("score", 0.5))
        obj.last_seen_tick = event.tick

    # ── Prediction engine ────────────────────────────────────

    def _compute_prediction_errors(self, current_tick: int) -> None:
        """Compare current observations against predictions, build summary."""
        surprises: List[Tuple[str, float]] = []
        labels: List[str] = []
        appearances = 0
        departures = 0
        speak_switches = 0

        # Person predictions
        for pid, p in self.persons.items():
            if pid not in self._predictions:
                self._predictions[pid] = EntityPrediction(
                    expected_distance_cm=p.distance_cm,
                    expected_center_x=p.center_x,
                    expected_center_y=p.center_y,
                    expected_speaking=1.0 if p.speaking else 0.0,
                )
                appearances += 1
                labels.append("new_person_appeared")
                surprises.append((pid, 0.4))
            else:
                s = self._predictions[pid].update_from_observation(
                    p.distance_cm, p.center_x, p.center_y, p.speaking, p.gesture
                )
                surprises.append((pid, s))
                if s > 0.3:
                    if self._predictions[pid].distance_error > 30:
                        labels.append("unexpected_movement")
                    if self._predictions[pid].speaking_error > 0.5:
                        labels.append("speaking_state_change")
                    if self._predictions[pid].position_error > 0.1:
                        labels.append("position_jump")

                # 1.3 — Temporal consistency: flag teleportation jumps as unreliable
                # If distance changes >150 cm in one tick, physics says this is impossible
                # → push distance back toward expected to dampen the jump
                _dist_jump = self._predictions[pid].distance_error
                if _dist_jump > 150.0 and self._predictions[pid]._update_count > 3:
                    labels.append("teleportation_anomaly")
                    # Damp the distance jump: blend toward expected value
                    p.distance_cm = (
                        self._predictions[pid].expected_distance_cm * 0.7
                        + p.distance_cm * 0.3
                    )

        # Detect unexpected departures (predicted present but gone)
        for pid, pred in self._predictions.items():
            if pid not in self.persons:
                if pred._update_count > 5 and pred.total_surprise < 0.8:
                    pred.mark_disappeared()
                    departures += 1
                    labels.append("unexpected_departure")
                    surprises.append((pid, pred.total_surprise))

        # Speaker switch detection
        current_speaker = None
        for p in self.persons.values():
            if p.speaking:
                current_speaker = p.person_id
                break
        if (
            current_speaker != self._prev_speaker_id
            and self._prev_speaker_id is not None
        ):
            speak_switches += 1
            if current_speaker is not None:
                labels.append("speaker_switch")
        self._prev_speaker_id = current_speaker

        # Build summary
        s = self.prediction_summary
        if surprises:
            s.mean_surprise = sum(v for _, v in surprises) / len(surprises)
            best = max(surprises, key=lambda x: x[1])
            s.max_surprise = best[1]
            s.max_surprise_entity = best[0]
        else:
            s.mean_surprise = 0.0
            s.max_surprise = 0.0
            s.max_surprise_entity = ""
        s.unexpected_appearances = appearances
        s.unexpected_departures = departures
        s.speaking_switches = speak_switches
        s.semantic_labels = labels[:5]  # cap to prevent runaway

    def describe_predictions(self) -> str:
        """Compact one-line prediction summary for consciousness."""
        s = self.prediction_summary
        if s.max_surprise < 0.05:
            return "predictions_ok"
        parts = [f"surprise={s.max_surprise:.2f}@{s.max_surprise_entity}"]
        if s.semantic_labels:
            parts.append("labels=" + ",".join(s.semantic_labels))
        if s.unexpected_appearances:
            parts.append(f"+{s.unexpected_appearances}new")
        if s.unexpected_departures:
            parts.append(f"-{s.unexpected_departures}gone")
        return " ".join(parts)

    # ── Serialization for persistence ────────────────────────

    def to_dict(self, current_tick: int = 0) -> Dict:
        """Snapshot world state for persistence. Only saves entities
        that are fresh enough to be meaningful on reload."""
        freshness_limit = 300  # max ticks old for inclusion
        persons_data = {}
        for pid, p in self.persons.items():
            if current_tick and (current_tick - p.last_seen_tick) > freshness_limit:
                continue
            persons_data[pid] = {
                "person_id": p.person_id,
                "center_x": round(p.center_x, 4),
                "center_y": round(p.center_y, 4),
                "distance_cm": round(p.distance_cm, 1),
                "gesture": p.gesture,
                "speaking": p.speaking,
                "face_visible": p.face_visible,
                "last_seen_tick": p.last_seen_tick,
                "first_seen_tick": p.first_seen_tick,
                "interaction_count": p.interaction_count,
            }
        objects_data = {}
        for oid, o in self.objects.items():
            if current_tick and (current_tick - o.last_seen_tick) > freshness_limit:
                continue
            objects_data[oid] = {
                "label": o.label,
                "center_x": round(o.center_x, 4),
                "center_y": round(o.center_y, 4),
                "confidence": round(o.confidence, 3),
                "last_seen_tick": o.last_seen_tick,
                "first_seen_tick": o.first_seen_tick,
            }
        ps = self.prediction_summary
        # Serialize entity predictions (freshness-filtered)
        predictions_data = {}
        for eid, ep in self._predictions.items():
            if current_tick and ep._update_count < 3:
                continue  # too few observations to persist
            predictions_data[eid] = {
                "expected_distance_cm": round(ep.expected_distance_cm, 2),
                "expected_center_x": round(ep.expected_center_x, 4),
                "expected_center_y": round(ep.expected_center_y, 4),
                "expected_speaking": round(ep.expected_speaking, 3),
                "expected_gesture": ep.expected_gesture,
                "velocity_cm": round(ep.velocity_cm, 3),
                "velocity_x": round(ep.velocity_x, 4),
                "velocity_y": round(ep.velocity_y, 4),
                "_update_count": ep._update_count,
            }
        return {
            "persons": persons_data,
            "objects": objects_data,
            "prediction_summary": {
                "mean_surprise": round(ps.mean_surprise, 4),
                "max_surprise": round(ps.max_surprise, 4),
                "max_surprise_entity": ps.max_surprise_entity,
            },
            "next_person_id": self._next_person_id,
            "predictions": predictions_data,
            "prev_speaker_id": self._prev_speaker_id,
        }

    def from_dict(self, data: Dict, current_tick: int = 0) -> None:
        """Restore world state from persistence snapshot.
        Only re-activates entries within freshness window."""
        freshness_limit = 600  # generous on load — sensor will refresh soon
        self.persons.clear()
        for pid, pd in data.get("persons", {}).items():
            age = (current_tick - pd.get("last_seen_tick", 0)) if current_tick else 0
            if age > freshness_limit:
                continue
            self.persons[pid] = TrackedPerson(
                person_id=pd.get("person_id", pid),
                center_x=pd.get("center_x", 0.5),
                center_y=pd.get("center_y", 0.5),
                distance_cm=pd.get("distance_cm", 999.0),
                gesture=pd.get("gesture", "none"),
                speaking=pd.get("speaking", False),
                face_visible=pd.get("face_visible", False),
                last_seen_tick=pd.get("last_seen_tick", 0),
                first_seen_tick=pd.get("first_seen_tick", 0),
                interaction_count=pd.get("interaction_count", 0),
            )
        self.objects.clear()
        for oid, od in data.get("objects", {}).items():
            age = (current_tick - od.get("last_seen_tick", 0)) if current_tick else 0
            if age > freshness_limit:
                continue
            self.objects[oid] = TrackedObject(
                label=od.get("label", "unknown"),
                center_x=od.get("center_x", 0.5),
                center_y=od.get("center_y", 0.5),
                confidence=od.get("confidence", 0.0),
                last_seen_tick=od.get("last_seen_tick", 0),
                first_seen_tick=od.get("first_seen_tick", 0),
            )
        ps_data = data.get("prediction_summary", {})
        self.prediction_summary.mean_surprise = ps_data.get("mean_surprise", 0.0)
        self.prediction_summary.max_surprise = ps_data.get("max_surprise", 0.0)
        self.prediction_summary.max_surprise_entity = ps_data.get(
            "max_surprise_entity", ""
        )
        self._next_person_id = data.get("next_person_id", len(self.persons))
        # Restore entity predictions with soft EMA re-initialisation
        self._predictions.clear()
        for eid, pd_pred in data.get("predictions", {}).items():
            ep = EntityPrediction(
                expected_distance_cm=pd_pred.get("expected_distance_cm", 999.0),
                expected_center_x=pd_pred.get("expected_center_x", 0.5),
                expected_center_y=pd_pred.get("expected_center_y", 0.5),
                expected_speaking=pd_pred.get("expected_speaking", 0.0),
                expected_gesture=pd_pred.get("expected_gesture", "none"),
                velocity_cm=pd_pred.get("velocity_cm", 0.0),
                velocity_x=pd_pred.get("velocity_x", 0.0),
                velocity_y=pd_pred.get("velocity_y", 0.0),
            )
            # Use fast EMA initially so sensor data quickly overrides stale state
            ep._update_count = max(1, pd_pred.get("_update_count", 1) // 2)
            self._predictions[eid] = ep
        self._prev_speaker_id = data.get("prev_speaker_id")
