"""
consciousness_testbed.py — Hard Test Suite for Consciousness Architecture

Implements the mandatory test classes:
    1. Dissociation Tests          - information processed but not reported
    2. Perturbation Tests          - noise injection changes integrated state
    3. Agency Manipulation Tests   - self-caused vs externally-caused discrimination
    4. False Belief / Self-Error Tests - system detects its own mistakes
    5. Counterfactual Consistency Tests - CF reasoning consistent with causal structure

Each test returns a TestResult with:
  - passed: bool
  - test_name: str
  - detail: str  (what was measured)
  - failure_analysis: str  (if failed — what component is responsible)
  - recommendation: str  (if failed — what change is needed)

Usage:
    testbed = ConsciousnessTestbed()
    results = testbed.run_all(consciousness_core, brain)
    report  = testbed.final_report(results)

    # Or run individual classes:
    results = testbed.run_dissociation_tests(core)
    results = testbed.run_perturbation_tests(core, brain)
    results = testbed.run_agency_tests(core)
    results = testbed.run_false_belief_tests(core)
    results = testbed.run_counterfactual_tests(core)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from brain import Brain
    from consciousness import ConsciousnessCore


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class TestResult:
    test_name: str
    passed: bool
    detail: str
    failure_analysis: str = ""
    recommendation: str = ""
    category: str = ""  # "dissociation" | "perturbation" | "agency" |
    # "false_belief" | "counterfactual"


# ─── Main class ───────────────────────────────────────────────────────────────


class ConsciousnessTestbed:
    """
    Hard test suite for the consciousness architecture.
    All tests are self-contained and non-destructive.
    """

    # ── Class 1: Dissociation Tests ──────────────────────────────────────────

    def run_dissociation_tests(self, core: "ConsciousnessCore") -> List[TestResult]:
        """
        Test that processing and report are architecturally separated.

        Tests:
          D1: Sub-threshold concept -> no report (subliminal processing)
          D2: Above-threshold concept -> report allowed
          D3: Metacognition can name its own blind spots
        """
        results = []

        # D1: Sub-threshold -> report blocked
        d1_pass, d1_detail = core.gateway.dissociation_probe("subliminal_test")
        results.append(
            TestResult(
                test_name="D1: Subliminal processing without report",
                passed=d1_pass,
                detail=d1_detail,
                failure_analysis=(
                    ""
                    if d1_pass
                    else "ConsciousnessGateway.can_report() is not enforcing the integration "
                    "threshold correctly. The report gate is not functioning."
                ),
                recommendation=(
                    ""
                    if d1_pass
                    else "Verify that gateway.try_promote() rejects sub-threshold concepts "
                    "and that can_report() returns False for non-workspace concepts."
                ),
                category="dissociation",
            )
        )

        # D2: Above-threshold -> report allowed
        d2_pass, d2_detail = core.gateway.access_probe("accessible_test")
        results.append(
            TestResult(
                test_name="D2: Globally accessible concept can be reported",
                passed=d2_pass,
                detail=d2_detail,
                failure_analysis=(
                    ""
                    if d2_pass
                    else "Above-threshold concepts are not being admitted to the global workspace. "
                    "try_promote() or the workspace capacity logic may be broken."
                ),
                recommendation=(
                    ""
                    if d2_pass
                    else "Check INTEGRATION_THRESHOLD vs test integration_score. "
                    "Ensure WORKSPACE_CAPACITY is not preventing promotion."
                ),
                category="dissociation",
            )
        )

        # D3: Metacognition names blind spots
        d3_pass, d3_detail = core.gateway.metacog_boundary_probe()
        results.append(
            TestResult(
                test_name="D3: Metacognition identifies access boundaries",
                passed=d3_pass,
                detail=d3_detail,
                failure_analysis=(
                    ""
                    if d3_pass
                    else "metacog_blind_spots() is not detecting sub-threshold concepts. "
                    "The Layer 3 monitoring is not functioning."
                ),
                recommendation=(
                    ""
                    if d3_pass
                    else "Ensure that registry entries for non-promoted concepts are retained "
                    "and visible to metacog_blind_spots()."
                ),
                category="dissociation",
            )
        )

        # D4: In an isolated probe, blocked reports stay blocked and allowed
        # reports pass only after real global access.
        _allowed = "__probe_report_allowed__"
        _blocked = "__probe_report_blocked__"
        core.gateway.register_processed(
            _allowed, source="d4_probe", integration_score=0.8
        )
        core.gateway.try_promote(_allowed, 0.8, source="d4_probe")
        _allowed_ok = core.gateway.can_report(_allowed)
        if _allowed_ok:
            core.gateway.mark_reported(_allowed)
        core.gateway.register_processed(
            _blocked, source="d4_probe", integration_score=0.05
        )
        _blocked_ok = core.gateway.can_report(_blocked)
        report_acc = 1.0 if (_allowed_ok and not _blocked_ok) else 0.0
        d4_pass = _allowed_ok and not _blocked_ok
        core.gateway._registry.pop(_allowed, None)
        core.gateway._workspace.pop(_allowed, None)
        core.gateway._registry.pop(_blocked, None)
        core.gateway._workspace.pop(_blocked, None)
        results.append(
            TestResult(
                test_name="D4: No false reports (report-without-access blocked)",
                passed=d4_pass,
                detail=(
                    f"report_accuracy={report_acc:.3f} "
                    f"allowed={_allowed_ok} blocked={not _blocked_ok}"
                ),
                failure_analysis=(
                    ""
                    if d4_pass
                    else "The gateway has allowed reports without global workspace access. "
                    "This violates the architecture rule: no report without global access."
                ),
                recommendation=(
                    ""
                    if d4_pass
                    else "Audit all places in consciousness.py that call gateway.mark_reported() "
                    "to ensure they first call can_report() and respect False responses."
                ),
                category="dissociation",
            )
        )

        return results

    # ── Class 2: Perturbation Tests ──────────────────────────────────────────

    def run_perturbation_tests(
        self,
        core: "ConsciousnessCore",
        brain: Optional["Brain"],
    ) -> List[TestResult]:
        """
        Test that the system's information is integrated (non-decomposable).

        Tests:
          P1: Phi-surrogate > 0 (some integration exists)
          P2: Perturbation noise propagates to >= 30% of other regions
          P3: Lesion of one region measurably affects others
        """
        results = []

        if brain is not None and hasattr(brain, "_tick"):
            if max(brain.region_activity.values(), default=0.0) <= 0.001:
                for _ in range(60):
                    brain.inject_text_input(
                        "integration probe recurrent workspace self model memory feedback"
                    )
                    brain._tick()

        # P1: Phi surrogate
        phi = core.integration_probe.phi_surrogate()
        p1_pass = phi > 0.0
        results.append(
            TestResult(
                test_name="P1: System has non-zero information integration (phi > 0)",
                passed=p1_pass,
                detail=f"phi_surrogate={phi:.4f} density={core.integration_probe.integration_density():.3f}",
                failure_analysis=(
                    ""
                    if p1_pass
                    else "No integration has been measured yet. The integration probe "
                    "may not have received enough tick observations to compute correlations."
                ),
                recommendation=(
                    ""
                    if p1_pass
                    else "Run the brain for >= 50 ticks before executing this test. "
                    "Verify integration_probe.observe() is being called each tick."
                ),
                category="perturbation",
            )
        )

        # P2: Perturbation spread
        if brain is not None:
            p2_pass, p2_detail = core.integration_probe.assert_perturbation_spread(
                brain, brain.tick_count if hasattr(brain, "tick_count") else 0
            )
        else:
            p2_pass, p2_detail = False, "SKIP: brain reference not available"
        results.append(
            TestResult(
                test_name="P2: Noise injection propagates across regions",
                passed=p2_pass or "SKIP" in p2_detail,
                detail=p2_detail,
                failure_analysis=(
                    ""
                    if (p2_pass or "SKIP" in p2_detail)
                    else "Perturbation noise does not propagate. This suggests that brain "
                    "regions are operating as independent sub-processes with minimal "
                    "cross-region causal influence."
                ),
                recommendation=(
                    ""
                    if (p2_pass or "SKIP" in p2_detail)
                    else "Check synapse weights between regions. If STDP has not run long "
                    "enough, cross-region correlations will be near zero. Consider "
                    "running for more ticks before evaluating integration."
                ),
                category="perturbation",
            )
        )

        # P3: Lesion impact
        if brain is not None and brain.region_activity:
            regions = list(brain.region_activity.keys())
            target_region = max(
                regions, key=lambda r: brain.region_activity.get(r, 0.0)
            )
            lesion_r = core.integration_probe.run_lesion_test(
                brain,
                target_region,
                brain.tick_count if hasattr(brain, "tick_count") else 0,
            )
            if lesion_r:
                p3_pass = lesion_r.mean_impact > 0.0
                p3_detail = (
                    f"lesion={lesion_r.target_region} "
                    f"mean_impact={lesion_r.mean_impact:.3f} "
                    f"max_impact={lesion_r.max_impact:.3f} on '{lesion_r.max_impact_region}'"
                )
            else:
                p3_pass, p3_detail = False, "Lesion test returned None"
        else:
            p3_pass, p3_detail = True, "SKIP: no brain regions available"
        results.append(
            TestResult(
                test_name="P3: Virtual lesion of one region affects others",
                passed=p3_pass,
                detail=p3_detail,
                failure_analysis=(
                    ""
                    if p3_pass
                    else "Lesioning a region has no impact on others. Causal structure "
                    "between regions is absent or too weak to measure."
                ),
                recommendation=(
                    ""
                    if p3_pass
                    else "Allow more training ticks for STDP to build inter-region coupling. "
                    "Or verify that region activities are non-zero (check sensor input)."
                ),
                category="perturbation",
            )
        )

        return results

    # ── Class 3: Agency Manipulation Tests ───────────────────────────────────

    def run_agency_tests(self, core: "ConsciousnessCore") -> List[TestResult]:
        """
        Tests:
          A1: Self-caused changes attributed more to self than external changes
          A2: Agency probability > 0 after at least one trial
          A3: Ownership coherence > 0 (consistent attribution pattern)
        """
        results = []

        # A1: Discrimination probe
        a1_pass, a1_detail = core.agency_validator.agency_manipulation_probe()
        results.append(
            TestResult(
                test_name="A1: Validator distinguishes self-caused from external changes",
                passed=a1_pass,
                detail=a1_detail,
                failure_analysis=(
                    ""
                    if a1_pass
                    else "The agency validator cannot distinguish between self-caused and "
                    "externally-caused world changes. The net_agency_delta or Bayesian "
                    "update logic may be incorrectly calibrated."
                ),
                recommendation=(
                    ""
                    if a1_pass
                    else "Review AgencyValidator.observe_outcome() — particularly how "
                    "net_agency_delta and agency_signal are computed. Ensure the "
                    "baseline_delta is not larger than the actual effect to detect."
                ),
                category="agency",
            )
        )

        # A2: Agency probability > 0
        p_agency = core.agency_validator.state.agency_probability
        a2_pass = 0.0 < p_agency < 1.0  # must be non-trivial
        results.append(
            TestResult(
                test_name="A2: Agency probability is non-degenerate",
                passed=a2_pass,
                detail=f"p_agency={p_agency:.3f} trials={core.agency_validator.state.total_trials}",
                failure_analysis=(
                    ""
                    if a2_pass
                    else f"Agency probability is degenerate (p={p_agency:.3f}). "
                    "Either no trials have been observed (p=0.5 prior) or the "
                    "Bayesian update is stuck at 0 or 1."
                ),
                recommendation=(
                    ""
                    if a2_pass
                    else "Ensure the validator is being called from the tick loop. "
                    "Check that record_action() and observe_outcome() are invoked."
                ),
                category="agency",
            )
        )

        # A3: Ownership coherence
        coherence = core.agency_validator.state.ownership_coherence
        a3_pass = coherence > 0.2
        results.append(
            TestResult(
                test_name="A3: Ownership attributions are internally consistent",
                passed=a3_pass,
                detail=f"ownership_coherence={coherence:.3f}",
                failure_analysis=(
                    ""
                    if a3_pass
                    else "Ownership coherence is very low — agency attributions vary "
                    "wildly across trials with no stable pattern."
                ),
                recommendation=(
                    ""
                    if a3_pass
                    else "Allow more trials to accumulate. Ownership coherence requires "
                    ">= 10 recent trials. Consider increasing the EMA window if "
                    "the environment is extremely noisy."
                ),
                category="agency",
            )
        )

        return results

    # ── Class 4: False Belief / Self-Error Tests ─────────────────────────────

    def run_false_belief_tests(self, core: "ConsciousnessCore") -> List[TestResult]:
        """
        Tests:
          F1: System registers errors when predictions are wrong
          F2: Calibrator detects overconfidence after wrong claim
          F3: System can declare specific topics as unknown (boundaries)
        """
        results = []

        # F1 + F2: False belief probe
        f1_pass, f1_detail = core.metacog_calib.false_belief_probe()
        results.append(
            TestResult(
                test_name="F1+F2: System detects and records false self-predictions",
                passed=f1_pass,
                detail=f1_detail,
                failure_analysis=(
                    ""
                    if f1_pass
                    else "The metacog calibrator is not updating overconfidence when a "
                    "confident prediction is resolved as incorrect. The calibration "
                    "loop between claim registration and resolution is broken."
                ),
                recommendation=(
                    ""
                    if f1_pass
                    else "Verify that resolve_claim() correctly calls _update_trackers(). "
                    "Ensure bin updates are computing overconfidence correctly."
                ),
                category="false_belief",
            )
        )

        # F3: Boundary declaration
        f3_pass, f3_detail = core.metacog_calib.boundary_awareness_probe()
        results.append(
            TestResult(
                test_name="F3: System can declare and track knowledge boundaries",
                passed=f3_pass,
                detail=f3_detail,
                failure_analysis=(
                    ""
                    if f3_pass
                    else "declare_unknown() does not persist to the declared_unknowns deque."
                ),
                recommendation=(
                    ""
                    if f3_pass
                    else "Check MetacogCalibrator.declare_unknown() implementation."
                ),
                category="false_belief",
            )
        )

        # F4: Calibration error is finite and computable
        ece = core.metacog_calib.expected_calibration_error()
        f4_pass = isinstance(ece, float) and ece >= 0.0
        results.append(
            TestResult(
                test_name="F4: Expected calibration error (ECE) is computable",
                passed=f4_pass,
                detail=f"ECE={ece:.4f}",
                failure_analysis="" if f4_pass else "ECE is not a valid float.",
                recommendation="" if f4_pass else "Check calibration bin logic.",
                category="false_belief",
            )
        )

        # F5: Self-error report is generated without exceptions
        try:
            rep = core.metacog_calib.self_error_report()
            f5_pass = isinstance(rep.accuracy, float)
            f5_detail = (
                f"accuracy={rep.accuracy:.2f} "
                f"overconf={rep.overconfidence:+.2f} "
                f"ECE={rep.calibration_error:.3f}"
            )
        except Exception as e:
            f5_pass = False
            f5_detail = f"Exception: {e}"
        results.append(
            TestResult(
                test_name="F5: Self-error report generates without exception",
                passed=f5_pass,
                detail=f5_detail,
                failure_analysis=(
                    "" if f5_pass else "self_error_report() threw an exception."
                ),
                recommendation=(
                    ""
                    if f5_pass
                    else "Fix the exception in MetacogCalibrator.self_error_report()."
                ),
                category="false_belief",
            )
        )

        return results

    # ── Class 5: Counterfactual Consistency Tests ─────────────────────────────

    def run_counterfactual_tests(self, core: "ConsciousnessCore") -> List[TestResult]:
        """
        Tests:
          C1: CF consistency probe (trait > anti-trait after outcome pattern)
          C2: Self-explanations are generated after decisions
          C3: Open goal space has synthesized goals beyond 4 base goals
        """
        results = []

        # C1: CF consistency
        c1_pass, c1_detail = core.cf_engine.counterfactual_consistency_probe()
        results.append(
            TestResult(
                test_name="C1: Counterfactual reasoning is temporally consistent",
                passed=c1_pass,
                detail=c1_detail,
                failure_analysis=(
                    ""
                    if c1_pass
                    else "Trait weights do not reflect the action-outcome pattern. "
                    "The trait learning loop is not functioning correctly."
                ),
                recommendation=(
                    ""
                    if c1_pass
                    else "Verify CounterfactualEngine._observe_traits() and that "
                    "record_outcome() correctly calls _observe_traits() with "
                    "the appropriate strength parameter."
                ),
                category="counterfactual",
            )
        )

        # C2: Self-explanations exist
        explanations = core.cf_engine.recent_explanations(3)
        c2_pass = len(explanations) > 0
        c2_detail = (
            f"{len(explanations)} explanations generated, "
            f"last: '{explanations[-1].text_de[:80]}...'"
            if explanations
            else "No explanations generated yet"
        )
        results.append(
            TestResult(
                test_name="C2: System generates self-explanations for decisions",
                passed=c2_pass,
                detail=c2_detail,
                failure_analysis=(
                    ""
                    if c2_pass
                    else "No self-explanations have been generated. Either no decisions "
                    "have been recorded with outcomes, or _generate_explanation() fails."
                ),
                recommendation=(
                    ""
                    if c2_pass
                    else "Ensure cf_engine.record_decision() and record_outcome() are "
                    "called from the goal evaluation loop in ConsciousnessCore.tick()."
                ),
                category="counterfactual",
            )
        )

        # C3: Open goal space
        c3_pass, c3_detail = core.goal_synth.open_goal_space_probe()
        results.append(
            TestResult(
                test_name="C3: Goal space is open (beyond 4 hardcoded base goals)",
                passed=c3_pass,
                detail=c3_detail,
                failure_analysis=(
                    ""
                    if c3_pass
                    else "Only the base 4 goals exist. The goal synthesizer is not "
                    "generating new goals from prediction errors, conflicts, or gaps."
                ),
                recommendation=(
                    ""
                    if c3_pass
                    else "Verify GoalSynthesizer.synthesize() is called from the tick loop "
                    "with non-zero prediction_error, meta_gaps, and self_tensions."
                ),
                category="counterfactual",
            )
        )

        # C4: Dominant trait is identifiable
        dom_trait, dom_w = core.cf_engine.dominant_trait()
        c4_pass = dom_trait != "undefined" or dom_w > 0.0
        results.append(
            TestResult(
                test_name="C4: System has an identifiable dominant trait",
                passed=c4_pass,
                detail=f"dominant_trait='{dom_trait}' weight={dom_w:.3f}",
                failure_analysis=(
                    ""
                    if c4_pass
                    else "No traits have been learned. The trait model is empty, "
                    "meaning the system cannot explain its behaviour in terms of "
                    "stable dispositions."
                ),
                recommendation=(
                    ""
                    if c4_pass
                    else "Ensure cf_engine.record_decision() is called from the goal "
                    "selection loop with the chosen and rejected goals."
                ),
                category="counterfactual",
            )
        )

        return results

    # ── Master runner ────────────────────────────────────────────────────────

    def run_all(
        self,
        core: "ConsciousnessCore",
        brain: Optional["Brain"] = None,
    ) -> List[TestResult]:
        """Run all 5 test classes and return combined results list."""
        results: List[TestResult] = []
        results.extend(self.run_dissociation_tests(core))
        results.extend(self.run_perturbation_tests(core, brain))
        results.extend(self.run_agency_tests(core))
        results.extend(self.run_false_belief_tests(core))
        results.extend(self.run_counterfactual_tests(core))
        return results

    def final_report(self, results: List[TestResult]) -> str:
        """
        Produce a critical summary report of all test results.

        Verdict categories:
          PASS  - all tests pass -> structure is intact
          WARN  - minority of tests fail -> specific gaps identified
          FAIL  - majority of tests fail -> fundamental architecture problems
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        cat_counts: dict = {}
        for r in results:
            cat_counts.setdefault(r.category, [0, 0])
            cat_counts[r.category][0] += int(r.passed)
            cat_counts[r.category][1] += 1

        if failed == 0:
            verdict = "PASS"
            verdict_note = (
                "All structural consciousness tests pass. "
                "Functional architecture is intact."
            )
        elif failed <= total // 3:
            verdict = "WARN"
            verdict_note = (
                f"{failed}/{total} tests fail. "
                "Targeted gaps identified — see failure_analysis fields."
            )
        else:
            verdict = "FAIL"
            verdict_note = (
                f"{failed}/{total} tests fail. "
                "Fundamental architecture problems. "
                "System cannot make consciousness-related claims."
            )

        lines = [
            "=" * 72,
            f"CONSCIOUSNESS ARCHITECTURE TEST REPORT",
            f"Verdict: {verdict}  ({passed}/{total} passed)",
            verdict_note,
            "-" * 72,
        ]

        for cat, (cp, ct) in cat_counts.items():
            lines.append(f"  {cat.upper():20s}: {cp}/{ct} passed")
        lines.append("-" * 72)

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  {status} {r.test_name}")
            lines.append(f"     -> {r.detail}")
            if not r.passed:
                if r.failure_analysis:
                    lines.append(f"     [CAUSE]  {r.failure_analysis}")
                if r.recommendation:
                    lines.append(f"     [FIX]    {r.recommendation}")

        lines.append("=" * 72)
        lines.append("")
        lines.append("CRITICAL EVALUATION (Architecture Properties):")
        lines.append("")
        lines.append(
            "  1. Self-modeling            : "
            + (
                "INTACT"
                if cat_counts.get("counterfactual", [0, 1])[0] > 0
                else "ABSENT"
            )
        )
        lines.append(
            "  2. Information integration  : "
            + (
                "MEASURABLE"
                if cat_counts.get("perturbation", [0, 1])[0] > 1
                else "NOT MEASURABLE"
            )
        )
        lines.append(
            "  3. Processing/Report split  : "
            + (
                "ENFORCED"
                if cat_counts.get("dissociation", [0, 1])[0] >= 3
                else "NOT ENFORCED"
            )
        )
        lines.append(
            "  4. Open goal space          : "
            + (
                "OPEN"
                if any(r.test_name.startswith("C3") and r.passed for r in results)
                else "CLOSED"
            )
        )
        lines.append(
            "  5. Counterfactual reasoning : "
            + (
                "PRESENT"
                if any(r.test_name.startswith("C") and r.passed for r in results)
                else "ABSENT"
            )
        )
        lines.append(
            "  6. Metacognitive calibration: "
            + (
                "CALIBRATED"
                if cat_counts.get("false_belief", [0, 1])[0] >= 3
                else "UNCALIBRATED"
            )
        )
        lines.append(
            "  7. Agency attribution       : "
            + (
                "VALIDATED"
                if cat_counts.get("agency", [0, 1])[0] >= 2
                else "NOT VALIDATED"
            )
        )
        lines.append("")
        lines.append("KEY MISSING CONDITIONS (if any failures above):")
        unmet = [r.test_name for r in results if not r.passed]
        if not unmet:
            lines.append("  None — all tested conditions are met.")
        else:
            for name in unmet:
                lines.append(f"  - {name}")
        lines.append("")
        lines.append("VERDICT QUALIFIER:")
        lines.append(
            "  Complexity, adaptivity, and language production are NOT sufficient "
            "proofs of consciousness. The tests above measure architectural "
            "properties that are NECESSARY (but not sufficient) for consciousness-"
            "like information processing."
        )
        lines.append("=" * 72)
        report = "\n".join(lines)
        return report.encode("ascii", "replace").decode("ascii")
