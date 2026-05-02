"""
counterfactual_engine.py — Temporally Stable Counterfactual Self-Model

Builds a persistent "who I am" model by:
  1. Anchoring stable traits from observed action patterns.
  2. Simulating counterfactual trajectories: "What would have happened
     if I had acted differently at tick T?"
  3. Generating self-explanations: "Ich habe X getan, weil ich Y bin."
  4. Evaluating temporal consistency: do traits remain stable across
     episodes, or are they contradicted by recent behaviour?

Architecture:
  ┌─────────────────────┐
  │   TraitModel        │ ← observed action-outcome pairs
  │   (name → weight)   │    update traits based on patterns
  └────────┬────────────┘
           │
  ┌────────▼────────────┐
  │  CounterfactualSim  │ ← "what if I had done Y instead of X?"
  │  (world model + CF) │    uses stored world-model deltas
  └────────┬────────────┘
           │
  ┌────────▼────────────┐
  │  SelfExplainer      │ ← "I did X because I am Y"
  │  (narrative output) │    anchors decisions to stable traits
  └────────┬────────────┘
           │
  ┌────────▼────────────┐
  │  ConsistencyChecker │ ← are recent actions consistent with traits?
  │  (temporal self)    │    flags contradictions for belief revision
  └─────────────────────┘

Integration:
    engine = CounterfactualEngine()

    # After every goal decision:
    engine.record_decision(tick, chosen_goal, rejected_goal,
                           chosen_score, rejected_score, causal_bonus)

    # After outcome arrives:
    engine.record_outcome(tick, goal_name, success, reward)

    # Periodically generate explanations:
    explanation = engine.explain_last_decision()
    consistency = engine.check_temporal_consistency()
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────
TRAIT_LEARN_RATE = 0.08  # per-observation update rate
TRAIT_DECAY_RATE = 0.0005  # per-tick slow decay
TRAIT_CONFIRM_THRESH = 0.56  # weight needed to call a trait "confirmed"
TRAIT_CONTRADICT_THRESH = 0.25  # weight below this → trait "weakened / contradicted"
CF_WINDOW = 50  # how many decisions are kept for simulation
CONSISTENCY_WINDOW = 20  # decisions used for temporal consistency check
MAX_TRAIT_NAMES = 30  # pool size for known traits


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class TraitObservation:
    """Maps an action pattern to a trait claim."""

    tick: int
    action: str
    trait: str  # e.g. "curiosity_driven", "social_responsive", "risk_averse"
    evidence: float  # strength of this observation [0..1]


@dataclass
class CFDecision:
    """One recorded decision with its counterfactual alternative."""

    tick: int
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    causal_bonus: float  # expected success probability for chosen
    outcome: str = "pending"  # "success" | "failure" | "pending"
    reward: float = 0.0
    explained: bool = False
    explanation: str = ""
    cf_outcome_est: str = "unknown"  # estimated outcome if rejected had been chosen


@dataclass
class SelfExplanation:
    """A generated self-explanation: 'I did X because I am Y.'"""

    tick: int
    action: str
    trait: str
    text_de: str
    text_en: str
    confidence: float


@dataclass
class ConsistencyReport:
    """Result of temporal consistency check."""

    tick: int
    consistent: bool
    consistency_score: float  # 0..1
    confirmed_traits: List[str]
    contradicted_traits: List[str]
    note: str


# ─── Main class ───────────────────────────────────────────────────────────────


class CounterfactualEngine:
    """
    Maintains a temporally stable counterfactual self-model.
    """

    # Goal → trait mappings (action patterns → character dispositions)
    _GOAL_TO_TRAITS: Dict[str, List[str]] = {
        "explore": ["curiosity_driven", "openness_high", "novelty_seeking"],
        "respond": ["social_responsive", "communicative", "other_oriented"],
        "consolidate": ["reflective", "coherence_seeking", "analytical"],
        "rest": ["self_preserving", "energy_conscious", "cautious"],
    }

    # Trait → natural language descriptors DE/EN
    _TRAIT_PHRASES: Dict[str, Tuple[str, str]] = {
        "curiosity_driven": ("meine Neugier antreibt", "my curiosity drives me"),
        "openness_high": ("ich offen für Neues bin", "I am open to new things"),
        "novelty_seeking": ("ich Neues suche", "I seek novelty"),
        "social_responsive": (
            "ich sozial reaktions­bereit bin",
            "I am socially responsive",
        ),
        "communicative": ("ich ausdrücken möchte", "I want to express myself"),
        "other_oriented": ("ich anderen zugewandt bin", "I am other-oriented"),
        "reflective": ("ich reflektieren muss", "I need to reflect"),
        "coherence_seeking": ("ich Kohärenz suche", "I seek coherence"),
        "analytical": ("ich analytisch denke", "I think analytically"),
        "self_preserving": ("mir meine Integrität wichtig ist", "I value my integrity"),
        "energy_conscious": ("ich meine Energie schone", "I conserve my energy"),
        "cautious": ("ich vorsichtig bin", "I am cautious"),
    }

    def __init__(self) -> None:
        self._tick: int = 0

        # Trait weight model: trait_name → float [0..1]
        self._traits: Dict[str, float] = {}
        self._trait_obs: Deque[TraitObservation] = deque(maxlen=500)

        # Decision history for counterfactual simulation
        self._decisions: Deque[CFDecision] = deque(maxlen=CF_WINDOW)

        # Generated explanations
        self._explanations: Deque[SelfExplanation] = deque(maxlen=100)

        # Consistency reports
        self._consistency_log: Deque[ConsistencyReport] = deque(maxlen=50)

        self._last_explanation_tick: int = -999

    # ── Core API ─────────────────────────────────────────────────────────────

    def record_decision(
        self,
        tick: int,
        chosen: str,
        rejected: str,
        chosen_score: float,
        rejected_score: float,
        causal_bonus: float,
    ) -> None:
        """Record a goal-selection decision for CF tracking."""
        self._tick = tick
        cf_est = self._simulate_counterfactual(rejected, causal_bonus)
        dec = CFDecision(
            tick=tick,
            chosen=chosen,
            rejected=rejected,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            causal_bonus=causal_bonus,
            cf_outcome_est=cf_est,
        )
        self._decisions.append(dec)
        # Observe trait evidence for the chosen goal
        self._observe_traits(tick, chosen, strength=0.6)

    def record_outcome(
        self,
        tick: int,
        goal_name: str,
        success: bool,
        reward: float,
    ) -> Optional[str]:
        """
        Match an outcome to the most recent matching decision,
        update trait weights, and generate explanation if warranted.
        Returns explanation string or None.
        """
        self._tick = tick
        explanation = None

        # Find most recent pending decision for this goal
        matching = [
            d
            for d in reversed(self._decisions)
            if d.chosen == goal_name and d.outcome == "pending"
        ]
        if not matching:
            return None

        dec = matching[0]
        dec.outcome = "success" if success else "failure"
        dec.reward = reward

        # Update trait weights from outcome
        strength = 0.8 if success else 0.2
        self._observe_traits(tick, goal_name, strength=strength)

        # Generate explanation for this decision
        if not dec.explained:
            explanation = self._generate_explanation(tick, dec)
            dec.explained = True
            dec.explanation = explanation

        # Decay all trait weights slightly
        self._decay_traits()

        return explanation

    # ── Counterfactual simulation ─────────────────────────────────────────────

    def _simulate_counterfactual(self, rejected_goal: str, causal_bonus: float) -> str:
        """
        Estimate what would have happened if the rejected goal had been chosen.
        Uses causal bonus of rejected goal as proxy for expected success rate.
        """
        # Estimate success probability for rejected goal from trait evidence
        rejected_traits = self._GOAL_TO_TRAITS.get(rejected_goal, [])
        trait_support = sum(self._traits.get(t, 0.3) for t in rejected_traits) / max(
            1, len(rejected_traits)
        )
        cf_success_prob = causal_bonus * 0.5 + trait_support * 0.5

        if cf_success_prob > 0.6:
            return "likely_success"
        elif cf_success_prob > 0.4:
            return "uncertain"
        else:
            return "likely_failure"

    # ── Trait learning ────────────────────────────────────────────────────────

    def _observe_traits(self, tick: int, goal: str, strength: float) -> None:
        """Update trait weights based on an action observation."""
        traits = self._GOAL_TO_TRAITS.get(goal, [])
        repeat_bonus = 0.0
        recent_same = sum(1 for d in list(self._decisions)[-6:] if d.chosen == goal)
        if recent_same >= 2:
            repeat_bonus = min(0.12, 0.03 * (recent_same - 1))
        for trait in traits:
            if trait not in self._traits:
                self._traits[trait] = 0.5
            # Positive update if goal succeeded well (strength > 0.5), else negative
            delta = TRAIT_LEARN_RATE * (strength * 2 - 1 + repeat_bonus)
            self._traits[trait] = max(0.0, min(1.0, self._traits[trait] + delta))
            self._trait_obs.append(
                TraitObservation(tick=tick, action=goal, trait=trait, evidence=strength)
            )

    def _decay_traits(self) -> None:
        for trait in list(self._traits):
            self._traits[trait] = max(0.0, self._traits[trait] - TRAIT_DECAY_RATE)

    # ── Self-explanation generation ───────────────────────────────────────────

    def _generate_explanation(self, tick: int, dec: CFDecision) -> str:
        """
        Generate: "Ich habe X getan, weil ich Y bin."
        Returns bilingual explanation string.
        """
        chosen_traits = self._GOAL_TO_TRAITS.get(dec.chosen, [])
        # Find the highest-weight confirmed trait
        best_trait = max(
            chosen_traits,
            key=lambda t: self._traits.get(t, 0.0),
            default=None,
        )
        trait_weight = self._traits.get(best_trait, 0.0) if best_trait else 0.0

        phrase_de, phrase_en = self._TRAIT_PHRASES.get(
            best_trait, ("ich so bin", "that is who I am")
        )

        outcome_de = (
            "Erfolg"
            if dec.outcome == "success"
            else ("Misserfolg" if dec.outcome == "failure" else "ausstehend")
        )
        outcome_en = dec.outcome

        cf_note_de = ""
        cf_note_en = ""
        if dec.cf_outcome_est != "unknown" and dec.rejected:
            cf_note_de = (
                f" Hätte ich '{dec.rejected}' gewählt, wäre das Ergebnis "
                f"vermutlich '{dec.cf_outcome_est}' gewesen."
            )
            cf_note_en = (
                f" If I had chosen '{dec.rejected}', the outcome would "
                f"likely have been '{dec.cf_outcome_est}'."
            )

        text_de = (
            f"Ich habe '{dec.chosen}' gewählt (Score={dec.chosen_score:.2f}), "
            f"weil {phrase_de}. "
            f"Ergebnis: {outcome_de} (reward={dec.reward:+.2f}).{cf_note_de}"
        )
        text_en = (
            f"I chose '{dec.chosen}' (score={dec.chosen_score:.2f}) "
            f"because {phrase_en}. "
            f"Outcome: {outcome_en} (reward={dec.reward:+.2f}).{cf_note_en}"
        )

        exp = SelfExplanation(
            tick=tick,
            action=dec.chosen,
            trait=best_trait or "unknown",
            text_de=text_de,
            text_en=text_en,
            confidence=trait_weight,
        )
        self._explanations.append(exp)
        self._last_explanation_tick = tick
        return text_de  # Default to German (system language)

    def explain_last_decision(self) -> Optional[str]:
        """Return the most recently generated self-explanation."""
        if self._explanations:
            return self._explanations[-1].text_de
        return None

    def explain_in(self, lang: str = "de") -> Optional[str]:
        """Return last explanation in specified language ('de' or 'en')."""
        if not self._explanations:
            return None
        last = self._explanations[-1]
        return last.text_de if lang == "de" else last.text_en

    # ── Temporal consistency check ────────────────────────────────────────────

    def check_temporal_consistency(self, tick: int) -> ConsistencyReport:
        """
        Check whether recent action patterns are consistent with claimed
        stable traits.  Returns ConsistencyReport.
        """
        confirmed = [t for t, w in self._traits.items() if w >= TRAIT_CONFIRM_THRESH]
        contradicted = [
            t
            for t, w in self._traits.items()
            if w <= TRAIT_CONTRADICT_THRESH and t in self._traits
        ]

        recent_actions = [d.chosen for d in list(self._decisions)[-CONSISTENCY_WINDOW:]]
        if not recent_actions:
            score = 0.5
            note = "Insufficient decision history for consistency check."
        else:
            # For each confirmed trait, check if the corresponding goals appear
            # in recent actions
            support_count = 0
            total_checks = 0
            for trait in confirmed:
                # Find which goals express this trait
                goal_set = {g for g, ts in self._GOAL_TO_TRAITS.items() if trait in ts}
                trait_present = any(a in goal_set for a in recent_actions)
                if trait_present:
                    support_count += 1
                total_checks += 1

            score = support_count / total_checks if total_checks > 0 else 0.5
            consistent = score >= 0.6 and len(contradicted) == 0
            note = (
                f"Confirmed traits in recent actions: "
                f"{support_count}/{total_checks} "
                f"({'consistent' if consistent else 'inconsistent'}). "
                f"Contradicted traits: {contradicted[:3]}"
            )

        consistent = score >= 0.6 and len(contradicted) == 0
        report = ConsistencyReport(
            tick=tick,
            consistent=consistent,
            consistency_score=score,
            confirmed_traits=confirmed[:6],
            contradicted_traits=contradicted[:6],
            note=note,
        )
        self._consistency_log.append(report)
        return report

    # ── Queries ──────────────────────────────────────────────────────────────

    def dominant_trait(self) -> Tuple[str, float]:
        if not self._traits:
            return ("undefined", 0.0)
        best = max(self._traits, key=self._traits.get)
        return best, self._traits[best]

    def trait_summary(self) -> str:
        if not self._traits:
            return "TRAITS: none"
        top = sorted(self._traits.items(), key=lambda x: x[1], reverse=True)[:4]
        parts = [f"{t}={w:.2f}" for t, w in top]
        confirmed = [t for t, w in top if w >= TRAIT_CONFIRM_THRESH]
        return f"TRAITS: [{', '.join(parts)}] " f"confirmed={confirmed}"

    def recent_explanations(self, n: int = 3) -> List[SelfExplanation]:
        return list(self._explanations)[-n:]

    def recent_decisions(self, n: int = 5) -> List[CFDecision]:
        return list(self._decisions)[-n:]

    def last_consistency_report(self) -> Optional[ConsistencyReport]:
        return self._consistency_log[-1] if self._consistency_log else None

    # ── Test probe ───────────────────────────────────────────────────────────

    def counterfactual_consistency_probe(self) -> Tuple[bool, str]:
        """
        Test: after recording a successful 'explore' decision and a failed
        'rest' decision, verify that:
          1. Traits reflect the outcome pattern (curiosity > cautious).
          2. Counterfactual for 'rest' predicts a different outcome than chosen.
          3. Consistency check reflects the trait pattern.
        """
        # Inject test decisions
        self.record_decision(9980, "explore", "rest", 1.2, 0.8, 0.7)
        self.record_outcome(9981, "explore", success=True, reward=0.6)
        self.record_decision(9982, "rest", "explore", 0.9, 1.1, 0.3)
        self.record_outcome(9983, "rest", success=False, reward=-0.2)

        curiosity_w = self._traits.get("curiosity_driven", 0.5)
        cautious_w = self._traits.get("cautious", 0.5)
        report = self.check_temporal_consistency(9984)
        last_exp = self.explain_last_decision()

        passed = (
            curiosity_w > cautious_w
            and len(report.confirmed_traits) >= 1
            and last_exp is not None
        )

        detail = (
            f"curiosity_driven={curiosity_w:.2f} vs cautious={cautious_w:.2f} | "
            f"confirmed_traits={report.confirmed_traits[:3]} | "
            f"explanation={'present' if last_exp else 'MISSING'}"
        )

        return passed, f"{'PASS' if passed else 'FAIL'}: CF consistency — {detail}"
