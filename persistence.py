"""
persistence.py — Brain State Persistence

Saves and loads the FULL brain state to brain_state.db (SQLite):
  • Synapse weights + delays (all inter- and intra-regional)
  • Simulation metadata (t, tick_count)
  • Amygdala valence
  • Emotion engine EMA state (per-region activity EMAs)
  • Consciousness: concepts, conclusions, stream, ignition_count
  • Episodic memory events
  • Self-model fields
  • Personality trait exposure weights
  • Communication drive state

On startup  : call load_brain(brain) to resume from last session.
On shutdown : call save_brain(brain) to preserve all learned state.
Autosave    : Brain calls save_brain() every AUTO_SAVE_INTERVAL ticks.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from brain import Brain

DB_PATH = "brain_state.db"
SYNAPSE_LAYOUT_VERSION = "spatial_v1"
SYNAPSE_LAYOUT_META_KEY = "synapse_layout_version"

_log = logging.getLogger("persistence")


# ─────────────────────────────────────────────────────────────────────────────
# Structured error journal — replaces silent except:pass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PersistenceIssue:
    """One issue encountered during save or load."""

    phase: str  # e.g. "save:beliefs", "load:episodic"
    severity: str = "warn"  # "warn" | "error" | "critical"
    message: str = ""
    exception: str = ""


@dataclass
class PersistenceReport:
    """Accumulated report for one save/load operation."""

    operation: str = ""  # "save" | "load"
    timestamp: float = 0.0
    ok_phases: List[str] = field(default_factory=list)
    issues: List[PersistenceIssue] = field(default_factory=list)

    def record_ok(self, phase: str) -> None:
        self.ok_phases.append(phase)

    def record_issue(self, phase: str, exc: Exception, severity: str = "warn") -> None:
        issue = PersistenceIssue(
            phase=phase,
            severity=severity,
            message=str(exc),
            exception=type(exc).__name__,
        )
        self.issues.append(issue)
        _log.warning(
            "persistence %s issue [%s] %s: %s",
            self.operation,
            phase,
            type(exc).__name__,
            exc,
        )

    @property
    def degraded(self) -> bool:
        return any(i.severity in ("error", "critical") for i in self.issues)

    def summary(self) -> str:
        n_ok = len(self.ok_phases)
        n_issues = len(self.issues)
        status = (
            "OK" if not self.issues else ("DEGRADED" if self.degraded else "PARTIAL")
        )
        parts = [f"{self.operation}: {status} ({n_ok} ok, {n_issues} issues)"]
        for issue in self.issues[:5]:
            parts.append(f"  [{issue.severity}] {issue.phase}: {issue.message[:80]}")
        return "\n".join(parts)


# Latest reports — accessible from GUI / telemetry
_last_save_report: PersistenceReport | None = None
_last_load_report: PersistenceReport | None = None


def last_save_report() -> PersistenceReport | None:
    return _last_save_report


def last_load_report() -> PersistenceReport | None:
    return _last_load_report


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS synapses (
    pre_nid  INTEGER NOT NULL,
    post_nid INTEGER NOT NULL,
    weight   REAL    NOT NULL,
    delay    REAL    NOT NULL,
    PRIMARY KEY (pre_nid, post_nid)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    val TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodic (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    tick     INTEGER NOT NULL,
    kind     TEXT    NOT NULL,
    content  TEXT    NOT NULL,
    emotion  TEXT    NOT NULL DEFAULT '',
    person_id INTEGER DEFAULT NULL,
    prediction TEXT  NOT NULL DEFAULT '',
    action   TEXT    NOT NULL DEFAULT '',
    observed_outcome TEXT NOT NULL DEFAULT '',
    causal_update TEXT NOT NULL DEFAULT '',
    social_person_ids TEXT NOT NULL DEFAULT '[]',
    phenomenal_vector TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS person_models (
    person_id       TEXT PRIMARY KEY,
    trust           REAL    NOT NULL DEFAULT 0.5,
    familiarity     REAL    NOT NULL DEFAULT 0.0,
    total_encounters INTEGER NOT NULL DEFAULT 0,
    total_words_heard INTEGER NOT NULL DEFAULT 0,
    total_words_spoken INTEGER NOT NULL DEFAULT 0,
    avg_valence     REAL    NOT NULL DEFAULT 0.0,
    last_encounter_tick INTEGER NOT NULL DEFAULT 0,
    relationship_type TEXT NOT NULL DEFAULT 'emergent_contact',
    relationship_confidence REAL NOT NULL DEFAULT 0.35,
    relationship_scores TEXT NOT NULL DEFAULT '{}',
    relationship_history TEXT NOT NULL DEFAULT '[]',
    preferences     TEXT    NOT NULL DEFAULT '{}',
    inferred_interests TEXT NOT NULL DEFAULT '[]',
    interaction_log TEXT    NOT NULL DEFAULT '[]'
);
"""


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(_DDL)
    # Migrate: add missing columns to episodic for pre-existing DBs
    try:
        cols = {row[1] for row in c.execute("PRAGMA table_info(episodic)")}
        if "person_id" not in cols:
            c.execute("ALTER TABLE episodic ADD COLUMN person_id INTEGER DEFAULT NULL")
        for col in ("prediction", "action", "observed_outcome", "causal_update"):
            if col not in cols:
                c.execute(
                    f"ALTER TABLE episodic ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                )
        if "social_person_ids" not in cols:
            c.execute(
                "ALTER TABLE episodic ADD COLUMN social_person_ids TEXT NOT NULL DEFAULT '[]'"
            )
        if "phenomenal_vector" not in cols:
            c.execute(
                "ALTER TABLE episodic ADD COLUMN phenomenal_vector TEXT NOT NULL DEFAULT ''"
            )
        c.commit()
    except Exception:
        pass
    # Migrate: person_models.person_id must be TEXT (values are strings like "person_0")
    try:
        pm_info = {
            row[1]: row[2].upper()
            for row in c.execute("PRAGMA table_info(person_models)")
        }
        if pm_info.get("person_id", "") == "INTEGER":
            c.execute("DROP TABLE person_models")
            c.executescript("""
                CREATE TABLE IF NOT EXISTS person_models (
                    person_id       TEXT PRIMARY KEY,
                    trust           REAL    NOT NULL DEFAULT 0.5,
                    familiarity     REAL    NOT NULL DEFAULT 0.0,
                    total_encounters INTEGER NOT NULL DEFAULT 0,
                    total_words_heard INTEGER NOT NULL DEFAULT 0,
                    total_words_spoken INTEGER NOT NULL DEFAULT 0,
                    avg_valence     REAL    NOT NULL DEFAULT 0.0,
                    last_encounter_tick INTEGER NOT NULL DEFAULT 0,
                    relationship_type TEXT NOT NULL DEFAULT 'emergent_contact',
                    relationship_confidence REAL NOT NULL DEFAULT 0.35,
                    relationship_scores TEXT NOT NULL DEFAULT '{}',
                    relationship_history TEXT NOT NULL DEFAULT '[]',
                    preferences     TEXT    NOT NULL DEFAULT '{}',
                    inferred_interests TEXT NOT NULL DEFAULT '[]',
                    interaction_log TEXT    NOT NULL DEFAULT '[]'
                );
            """)
            c.commit()
    except Exception:
        pass
    try:
        pm_cols = {row[1] for row in c.execute("PRAGMA table_info(person_models)")}
        if "relationship_type" not in pm_cols:
            c.execute(
                "ALTER TABLE person_models ADD COLUMN relationship_type TEXT NOT NULL DEFAULT 'emergent_contact'"
            )
        if "relationship_confidence" not in pm_cols:
            c.execute(
                "ALTER TABLE person_models ADD COLUMN relationship_confidence REAL NOT NULL DEFAULT 0.35"
            )
        if "relationship_scores" not in pm_cols:
            c.execute(
                "ALTER TABLE person_models ADD COLUMN relationship_scores TEXT NOT NULL DEFAULT '{}'"
            )
        if "relationship_history" not in pm_cols:
            c.execute(
                "ALTER TABLE person_models ADD COLUMN relationship_history TEXT NOT NULL DEFAULT '[]'"
            )
        c.commit()
    except Exception:
        pass
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────


def save_brain(brain: "Brain", db_path: str = DB_PATH) -> int:
    global _last_save_report
    report = PersistenceReport(operation="save", timestamp=time.time())
    conn = _conn(db_path)

    # ── 1. Synapses ───────────────────────────────────────────
    all_syn = list(brain._inter_synapses)
    for region in brain._all_regions:
        all_syn.extend(region.internal_synapses)

    conn.execute("DELETE FROM synapses")
    conn.executemany(
        "INSERT OR REPLACE INTO synapses VALUES (?,?,?,?)",
        [
            (s.pre.nid, s.post.nid, round(float(s.weight), 6), round(float(s.delay), 3))
            for s in all_syn
        ],
    )

    # ── 2. Simulation metadata ────────────────────────────────
    meta_rows = [
        ("sim_t", str(brain.t)),
        ("tick_count", str(brain.tick_count)),
        ("saved_at", str(time.time())),
        (SYNAPSE_LAYOUT_META_KEY, SYNAPSE_LAYOUT_VERSION),
    ]

    # ── 3. Amygdala valence ───────────────────────────────────
    meta_rows.append(("amygdala_valence", str(brain.amygdala.valence)))

    # ── 4. Emotion EMA state (per-region _act_ema dict) ──────
    try:
        ema_data = {k: v for k, v in brain._emotion_engine._act_ema.items()}
        meta_rows.append(("emotion_ema", json.dumps(ema_data)))
        report.record_ok("save:emotion_ema")
    except Exception as _exc:
        report.record_issue("save:emotion_ema", _exc)

    # ── 5. Consciousness core ─────────────────────────────────
    cs = brain._consciousness
    try:
        meta_rows.append(("concepts", json.dumps(list(cs._concepts))))
        meta_rows.append(("conclusions", json.dumps(list(cs._conclusions))))
        meta_rows.append(("stream", json.dumps(list(cs.stream)[-120:])))
        meta_rows.append(("ignition_count", str(cs._ignition_count)))
        meta_rows.append(("cs_tick", str(cs._tick)))
        meta_rows.append(("cs_goal", cs.state.goal))
        meta_rows.append(("prev_goal", getattr(cs, "_prev_goal", "")))
        meta_rows.append(("last_replay", str(cs._last_replay)))
        meta_rows.append(
            ("engrams", json.dumps({str(k): v for k, v in cs._engrams.items()}))
        )
        report.record_ok("save:consciousness_core")
    except Exception as _exc:
        report.record_issue("save:consciousness_core", _exc)

    # ── 6. Episodic memory (with causal fields) ───────────────────
    try:
        conn.execute("DELETE FROM episodic")
        conn.executemany(
            "INSERT INTO episodic (tick, kind, content, emotion, person_id,"
            " prediction, action, observed_outcome, causal_update,"
            " social_person_ids, phenomenal_vector)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    e.tick,
                    e.kind,
                    e.content,
                    e.emotion_snapshot,
                    e.social_person_id,
                    getattr(e, "prediction", ""),
                    getattr(e, "action", ""),
                    getattr(e, "observed_outcome", ""),
                    getattr(e, "causal_update", ""),
                    json.dumps(list(getattr(e, "social_person_ids", ()))),
                    (
                        json.dumps(getattr(e, "phenomenal_vector", None).tolist())
                        if getattr(e, "phenomenal_vector", None) is not None
                        else ""
                    ),
                )
                for e in cs.episodic._events
            ],
        )
        report.record_ok("save:episodic")
    except Exception as _exc:
        report.record_issue("save:episodic", _exc)

    # ── 7. Self-model ─────────────────────────────────────────
    try:
        sm = cs.self_model
        sm_data = {
            "name": sm.name,
            "birth_tick": sm.birth_tick,
            "total_spikes": sm.total_spikes,
            "concepts_learned": sm.concepts_learned,
            "ignitions_total": sm.ignitions_total,
            "known_strengths": sm.known_strengths,
            "known_gaps": sm.known_gaps,
        }
        meta_rows.append(("self_model", json.dumps(sm_data)))
        report.record_ok("save:self_model")
    except Exception as _exc:
        report.record_issue("save:self_model", _exc)

    # ── 8. Personality exposure ───────────────────────────────
    try:
        meta_rows.append(("personality_exposure", json.dumps(cs.personality._exposure)))
        meta_rows.append(
            ("personality_traits", json.dumps(cs.personality.active_traits))
        )
        report.record_ok("save:personality")
    except Exception as _exc:
        report.record_issue("save:personality", _exc)

    # ── 9. Meta-cognition familiarity + depth ─────────────────
    try:
        mc = cs.meta
        meta_rows.append(("meta_familiarity", json.dumps(mc._familiarity)))
        meta_rows.append(("meta_depth", json.dumps(mc._depth)))
        report.record_ok("save:meta_cognition")
    except Exception as _exc:
        report.record_issue("save:meta_cognition", _exc)

    # ── 10. Comm drive state ──────────────────────────────────
    try:
        cd = cs.comm_drive
        comm_data = {
            "drive": cd.drive,
            "_last_spoke": cd._last_spoke,
        }
        meta_rows.append(("comm_drive", json.dumps(comm_data)))
        report.record_ok("save:comm_drive")
    except Exception as _exc:
        report.record_issue("save:comm_drive", _exc)
    # ── 11. BeliefStore — propositional memory (explicit serialization) ─
    try:
        bs = cs.belief_store
        bs_serial: dict = {}
        for subj, rels in bs._beliefs.items():
            bs_serial[subj] = {}
            for rel, objs in rels.items():
                bs_serial[subj][rel] = {}
                for obj, entry in objs.items():
                    bs_serial[subj][rel][obj] = {
                        "confidence": entry.confidence,
                        "source": entry.source,
                        "first_tick": entry.first_tick,
                        "last_tick": entry.last_tick,
                        "evidence_count": entry.evidence_count,
                        "contradiction_count": entry.contradiction_count,
                        "epistemic_status": entry.epistemic_status.value,
                    }
        meta_rows.append(("belief_store", json.dumps(bs_serial)))
        # Quarantined beliefs
        quar_serial: dict = {}
        for (s, r, o), entry in bs._quarantine.items():
            key = f"{s}|{r}|{o}"
            quar_serial[key] = {
                "confidence": entry.confidence,
                "source": entry.source,
                "first_tick": entry.first_tick,
                "last_tick": entry.last_tick,
                "evidence_count": entry.evidence_count,
                "contradiction_count": entry.contradiction_count,
                "epistemic_status": entry.epistemic_status.value,
            }
        meta_rows.append(("belief_quarantine", json.dumps(quar_serial)))
        report.record_ok("save:beliefs")
    except Exception as _exc:
        report.record_issue("save:beliefs", _exc)

    # ── 12. LanguageProducer — language preference ────────────────
    try:
        meta_rows.append(("lang_pref", cs.lang._lang))
        report.record_ok("save:lang")
    except Exception as _exc:
        report.record_issue("save:lang", _exc)

    # ── 13. PersonModel persistence (social memory) ──────────────
    try:
        sm_obj = brain._social_manager
        conn.execute("DELETE FROM person_models")
        for pid, pm in sm_obj.person_models.items():
            conn.execute(
                "INSERT OR REPLACE INTO person_models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    pm.person_id,
                    pm.trust,
                    pm.familiarity,
                    pm.total_encounters,
                    pm.total_words_heard,
                    pm.total_words_spoken,
                    pm.avg_valence,
                    pm.last_encounter_tick,
                    getattr(pm, "relationship_type", "emergent_contact"),
                    getattr(pm, "relationship_confidence", 0.35),
                    json.dumps(getattr(pm, "relationship_scores", {})),
                    json.dumps(getattr(pm, "relationship_history", [])),
                    json.dumps(pm.preferences),
                    json.dumps(pm.inferred_interests),
                    json.dumps(pm.interaction_log[-50:]),
                ),
            )
        report.record_ok("save:person_models")
    except Exception as _exc:
        report.record_issue("save:person_models", _exc)

    # ── 14. Autobiographic chapters + guidelines + consistency ─────
    try:
        chapters = list(cs.autobiography._chapters)
        meta_rows.append(("auto_chapters", json.dumps(chapters[-20:])))
        meta_rows.append(("auto_identity_summary", cs.autobiography.identity_summary))
        # Guidelines with full metadata
        gl_data = [
            {
                "text": gl.text,
                "source": gl.source,
                "strength": gl.strength,
                "tick_born": gl.tick_born,
            }
            for gl in cs.autobiography._guidelines
        ]
        meta_rows.append(("auto_guidelines", json.dumps(gl_data)))
        # Identity consistency + recurring stats
        meta_rows.append(
            ("auto_identity_consistency", str(cs.autobiography._identity_consistency))
        )
        meta_rows.append(
            ("auto_recurring_goals", json.dumps(cs.autobiography._recurring_goals))
        )
        meta_rows.append(
            (
                "auto_recurring_emotions",
                json.dumps(cs.autobiography._recurring_emotions),
            )
        )
        meta_rows.append(
            ("auto_stable_topics", json.dumps(cs.autobiography._stable_topics))
        )
        meta_rows.append(
            (
                "auto_social_encounters",
                json.dumps(
                    {str(k): v for k, v in cs.autobiography._social_encounters.items()}
                ),
            )
        )
        meta_rows.append(
            (
                "auto_relationship_types",
                json.dumps(cs.autobiography._relationship_types),
            )
        )
        report.record_ok("save:autobiography")
    except Exception as _exc:
        report.record_issue("save:autobiography", _exc)

    # ── 16. World model — causal action-consequence entries ────────
    try:
        wm = cs.world_model
        wm_data = {f"{g}|{a}": e for (g, a), e in wm._entries.items()}
        meta_rows.append(("world_model", json.dumps(wm_data)))
        report.record_ok("save:world_model")
    except Exception as _exc:
        report.record_issue("save:world_model", _exc)

    # ── 17. Continuity monitor state ────────────────────────────
    try:
        cm = cs.continuity
        cm_data = {
            "memory_coherence": cm.memory_coherence,
            "agency_stability": cm.agency_stability,
            "value_stability": cm.value_stability,
        }
        meta_rows.append(("continuity_monitor", json.dumps(cm_data)))
        report.record_ok("save:continuity")
    except Exception as _exc:
        report.record_issue("save:continuity", _exc)

    # ── 18. Sandbox planner postmortem lessons ──────────────────
    try:
        sp = cs.sandbox_planner
        meta_rows.append(("sandbox_lessons", json.dumps(list(sp._postmortem_lessons))))
        report.record_ok("save:sandbox")
    except Exception as _exc:
        report.record_issue("save:sandbox", _exc)

    # ── 15. ExperienceAppraisal (learned emotion associations) ───
    try:
        ea = brain._emotion_engine.experience
        ea_data = {
            "associations": ea._associations,
            "observation_count": ea._observation_count,
        }
        meta_rows.append(("experience_appraisal", json.dumps(ea_data)))
        report.record_ok("save:experience")
    except Exception as _exc:
        report.record_issue("save:experience", _exc)

    # ── 19. Operative embodiment state ────────────────────────────
    # EmbodiedSelfState + RobotState + TaskFrame + SensorimotorForwardModel
    try:
        es = cs.embodied_self
        es_data = {
            "battery_estimate": es.battery_estimate,
            "motor_readiness": es.motor_readiness,
            "visual_contact": es.visual_contact,
            "proximity_alert": es.proximity_alert,
            "social_presence": es.social_presence,
            "posture": es.posture,
            "agency_window": es.agency_window,
            "focus_target": es.focus_target,
            "focus_x": es.focus_x,
            "focus_y": es.focus_y,
            "focus_size": es.focus_size,
        }
        meta_rows.append(("embodied_self", json.dumps(es_data)))
        report.record_ok("save:embodied_self")
    except Exception as _exc:
        report.record_issue("save:embodied_self", _exc)

    try:
        rs = cs.robot_state
        rs_data = {
            "head_yaw": rs.head_yaw,
            "head_pitch": rs.head_pitch,
            "gaze_target": rs.gaze_target,
            "torso_mode": rs.torso_mode,
            "left_arm_mode": rs.left_arm_mode,
            "right_arm_mode": rs.right_arm_mode,
            "left_gripper": rs.left_gripper,
            "right_gripper": rs.right_gripper,
            "locomotion_mode": rs.locomotion_mode,
            "interaction_zone": rs.interaction_zone,
            "engagement_level": rs.engagement_level,
            "imitation_readiness": rs.imitation_readiness,
            "target_x": rs.target_x,
            "target_y": rs.target_y,
        }
        meta_rows.append(("robot_state", json.dumps(rs_data)))
        report.record_ok("save:robot_state")
    except Exception as _exc:
        report.record_issue("save:robot_state", _exc)

    try:
        tf = cs.task_frame
        tf_data = {
            "active_task": tf.active_task,
            "operational_goal": tf.operational_goal,
            "current_step": tf.current_step,
            "subgoals": tf.subgoals[:5],
            "blockers": tf.blockers[:3],
            "progress": tf.progress,
            "confidence": tf.confidence,
            "last_result": tf.last_result,
            "mode": tf.mode,
            "plan_phase": tf.plan_phase,
        }
        meta_rows.append(("task_frame", json.dumps(tf_data)))
        report.record_ok("save:task_frame")
    except Exception as _exc:
        report.record_issue("save:task_frame", _exc)

    try:
        sm_fw = cs.sensorimotor
        sm_fw_data = {
            "models": {f"{act}|{ch}": m for (act, ch), m in sm_fw._models.items()},
            "prediction_error": sm_fw._prediction_error,
            "agency_confirmation": sm_fw._agency_confirmation,
        }
        meta_rows.append(("sensorimotor_model", json.dumps(sm_fw_data)))
        report.record_ok("save:sensorimotor")
    except Exception as _exc:
        report.record_issue("save:sensorimotor", _exc)

    # ── 20. Identity stats (identity_veto_count, identity_map) ────
    try:
        gs = cs.goal_system
        meta_rows.append(("goal_system_veto_count", str(getattr(gs, "_veto_count", 0))))
        report.record_ok("save:goal_system")
    except Exception as _exc:
        report.record_issue("save:goal_system", _exc)

    # ── 21. Causal graph ─────────────────────────────────────────
    try:
        meta_rows.append(("causal_graph", json.dumps(cs.causal_graph.to_dict())))
        report.record_ok("save:causal_graph")
    except Exception as _exc:
        report.record_issue("save:causal_graph", _exc)

    # ── 22. Value model ──────────────────────────────────────────
    try:
        meta_rows.append(("value_model", json.dumps(cs.value_model.to_dict())))
        report.record_ok("save:value_model")
    except Exception as _exc:
        report.record_issue("save:value_model", _exc)

    # ── 23. Identity arc ─────────────────────────────────────────
    try:
        meta_rows.append(("identity_arc", json.dumps(cs.identity_arc.to_dict())))
        report.record_ok("save:identity_arc")
    except Exception as _exc:
        report.record_issue("save:identity_arc", _exc)

    # ── 24. Narrative thread ─────────────────────────────────────
    try:
        meta_rows.append(
            ("narrative_thread", json.dumps(cs.narrative_thread.to_dict()))
        )
        report.record_ok("save:narrative_thread")
    except Exception as _exc:
        report.record_issue("save:narrative_thread", _exc)

    # ── 25. Theory of mind ───────────────────────────────────────
    try:
        meta_rows.append(("theory_of_mind", json.dumps(cs.theory_of_mind.to_dict())))
        report.record_ok("save:theory_of_mind")
    except Exception as _exc:
        report.record_issue("save:theory_of_mind", _exc)

    # ── 26. Belief quarantine module ──────────────────────────────
    try:
        meta_rows.append(
            ("belief_quarantine_module", json.dumps(cs.belief_quarantine.to_dict()))
        )
        report.record_ok("save:belief_quarantine_module")
    except Exception as _exc:
        report.record_issue("save:belief_quarantine_module", _exc)

    # ── 27. Attention controller ─────────────────────────────────
    try:
        meta_rows.append(("attention_ctrl", json.dumps(cs.attention_ctrl.to_dict())))
        report.record_ok("save:attention_ctrl")
    except Exception as _exc:
        report.record_issue("save:attention_ctrl", _exc)

    # ── 28. Long-horizon goals ───────────────────────────────────
    try:
        meta_rows.append(("long_horizon", json.dumps(cs.long_horizon.to_dict())))
        report.record_ok("save:long_horizon")
    except Exception as _exc:
        report.record_issue("save:long_horizon", _exc)

    # ── 29. Concept salience map ─────────────────────────────────
    try:
        meta_rows.append(("concept_salience", json.dumps(cs._concept_salience)))
        report.record_ok("save:concept_salience")
    except Exception as _exc:
        report.record_issue("save:concept_salience", _exc)

    # ── 30. Active scene anchors ─────────────────────────────────
    try:
        _as = cs.active_scene
        scene_data = {
            "focus_person": _as.focus_person,
            "focus_object": _as.focus_object,
            "focus_zone": _as.focus_zone,
            "active_relations": _as.active_relations,
            "scene_confidence": _as.scene_confidence,
            "last_update_tick": _as.last_update_tick,
        }
        meta_rows.append(("active_scene", json.dumps(scene_data)))
        report.record_ok("save:active_scene")
    except Exception as _exc:
        report.record_issue("save:active_scene", _exc)

    # ── 31. Unified self-state ───────────────────────────────────
    try:
        meta_rows.append(("unified_self", json.dumps(cs.unified_self.to_dict())))
        report.record_ok("save:unified_self")
    except Exception as _exc:
        report.record_issue("save:unified_self", _exc)

    # ── 32. Grounded semantic memory ─────────────────────────────
    try:
        meta_rows.append(("grounded_memory", json.dumps(cs.grounded_memory.to_dict())))
        report.record_ok("save:grounded_memory")
    except Exception as _exc:
        report.record_issue("save:grounded_memory", _exc)

    # ── 33. Phenomenal buffer ────────────────────────────────────
    try:
        meta_rows.append(
            ("phenomenal_buffer", json.dumps(cs.phenomenal_buffer.to_dict()))
        )
        report.record_ok("save:phenomenal_buffer")
    except Exception as _exc:
        report.record_issue("save:phenomenal_buffer", _exc)

    # ── 34. Learned world model (RSSM) ──────────────────────────
    try:
        meta_rows.append(("learned_world", json.dumps(cs.learned_world.to_dict())))
        report.record_ok("save:learned_world")
    except Exception as _exc:
        report.record_issue("save:learned_world", _exc)

    # ── 35. Model-based planner ──────────────────────────────────
    try:
        meta_rows.append(("model_planner", json.dumps(cs.model_planner.to_dict())))
        report.record_ok("save:model_planner")
    except Exception as _exc:
        report.record_issue("save:model_planner", _exc)

    # ── 36. Self-model full dynamic state ────────────────────────
    try:
        sm = cs.self_model
        sm_full = {
            "energy": sm.energy,
            "uncertainty": sm.uncertainty,
            "identity_stability": sm.identity_stability,
            "agency_score": sm.agency_score,
            "self_awareness_level": sm.self_awareness_level,
            "ownership_confidence": sm.ownership_confidence,
            "agency_confidence": sm.agency_confidence,
            "continuity_estimate": sm.continuity_estimate,
            "self_tensions": sm.self_tensions,
            "self_narrative": sm.self_narrative,
            "self_beliefs": sm.self_beliefs,
            "self_theses": [th.to_dict() for th in sm.self_theses.values()],
            "self_contradictions": sm.self_contradictions,
            "last_action_self_report": sm.last_action_self_report,
            "body_pose": sm.body_pose,
            "body_load": sm.body_load,
            "body_pain": sm.body_pain,
            "balance": sm.balance,
            "last_skill": sm.last_skill,
            "turn_state": sm.turn_state,
        }
        meta_rows.append(("self_model_dynamic", json.dumps(sm_full)))
        report.record_ok("save:self_model_dynamic")
    except Exception as _exc:
        report.record_issue("save:self_model_dynamic", _exc)

    # ── 37. Dialogue manager state (common ground per person) ─────
    try:
        dm = brain._dialogue_manager
        meta_rows.append(("dialogue_manager", json.dumps(dm.to_dict())))
        report.record_ok("save:dialogue_manager")
    except Exception as _exc:
        report.record_issue("save:dialogue_manager", _exc)

    # ── 38. World state snapshot (persons, objects, predictions) ─────
    try:
        ws = brain._world_state
        _ws_data = ws.to_dict(current_tick=brain.tick_count)
        meta_rows.append(("world_state", json.dumps(_ws_data)))
        report.record_ok("save:world_state")
    except Exception as _exc:
        report.record_issue("save:world_state", _exc)

    _last_save_report = report

    conn.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", meta_rows)
    conn.commit()
    conn.close()
    return len(all_syn)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────


def load_brain(brain: "Brain", db_path: str = DB_PATH) -> int:
    """
    Fully reconstruct brain topology and all learned state from a previous session.

    Strategy:
      1. Clear all randomly-initialised synapses from this session.
      2. Rebuild every saved synapse by looking up neurons by nid.
      3. Restore all consciousness/emotion/meta state.

    Returns the number of synapses restored.
    """
    global _last_load_report
    report = PersistenceReport(operation="load", timestamp=time.time())

    if not os.path.exists(db_path):
        return 0

    try:
        conn = sqlite3.connect(db_path)

        def _get(key: str, default: str = "") -> str:
            row = conn.execute("SELECT val FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else default

        rows = conn.execute(
            "SELECT pre_nid, post_nid, weight, delay FROM synapses"
        ).fetchall()
        saved_layout_version = _get(SYNAPSE_LAYOUT_META_KEY)

        restored = 0
        if not rows:
            # No synapses — but still restore consciousness/meta state below
            pass
        elif saved_layout_version != SYNAPSE_LAYOUT_VERSION:
            report.record_issue(
                "load:synapses",
                RuntimeError(
                    "Skipping persisted synapses due to incompatible layout version "
                    f"{saved_layout_version or '<missing>'} != {SYNAPSE_LAYOUT_VERSION}"
                ),
                severity="warn",
            )
        else:
            # ── Build nid lookup ──────────────────────────────────
            nid_to_neuron: dict = {}
            for region in brain._all_regions:
                for n in region.neurons:
                    nid_to_neuron[n.nid] = n

            # ── Tear down existing random topology ────────────────
            def _detach(syns):
                for syn in syns:
                    try:
                        syn.pre.efferents.remove(syn)
                    except ValueError:
                        pass
                    try:
                        syn.post.afferents.remove(syn)
                    except ValueError:
                        pass

            _detach(brain._inter_synapses)
            brain._inter_synapses.clear()
            for region in brain._all_regions:
                _detach(region.internal_synapses)
                region.internal_synapses.clear()

            # ── Rebuild saved topology ────────────────────────────
            from synapse import Synapse as _Synapse

            region_map = {r.name: r for r in brain._all_regions}

            for pre_nid, post_nid, weight, delay in rows:
                pre_n = nid_to_neuron.get(pre_nid)
                post_n = nid_to_neuron.get(post_nid)
                if pre_n is None or post_n is None:
                    continue
                syn = _Synapse(pre_n, post_n, weight=float(weight), delay=float(delay))
                if pre_n.region == post_n.region:
                    r = region_map.get(pre_n.region)
                    if r is not None:
                        r.internal_synapses.append(syn)
                else:
                    brain._inter_synapses.append(syn)
                restored += 1

        # ── Restore simulation time ───────────────────────────
        for k, attr, cast in [
            ("sim_t", "t", float),
            ("tick_count", "tick_count", int),
        ]:
            v = _get(k)
            if v:
                try:
                    setattr(brain, attr, cast(v))
                except (ValueError, TypeError):
                    pass

        # ── Restore amygdala valence ──────────────────────────
        v = _get("amygdala_valence")
        if v:
            try:
                brain.amygdala.valence = float(v)
            except Exception:
                pass

        # ── Restore emotion EMA ───────────────────────────────
        v = _get("emotion_ema")
        if v:
            try:
                brain._emotion_engine._act_ema.update(json.loads(v))
            except Exception:
                pass

        # ── Restore consciousness core ────────────────────────
        cs = brain._consciousness

        v = _get("concepts")
        if v:
            try:
                from consciousness import _CONCEPT_STOPWORDS

                for c in json.loads(v):
                    # Re-apply the stopword filter so old sessions can't
                    # permanently pollute concept memory with function words.
                    if c and len(c) > 4 and c.lower() not in _CONCEPT_STOPWORDS:
                        cs._concepts.append(c)
            except Exception:
                pass

        v = _get("conclusions")
        if v:
            try:
                for c in json.loads(v):
                    cs._conclusions.append(c)
            except Exception:
                pass

        v = _get("stream")
        if v:
            try:
                for s in json.loads(v):
                    cs.stream.append(s)
            except Exception:
                pass

        v = _get("ignition_count")
        if v:
            try:
                cs._ignition_count = int(v)
            except Exception:
                pass

        v = _get("cs_tick")
        if v:
            try:
                cs._tick = int(v)
            except Exception:
                pass

        v = _get("cs_goal")
        if v:
            cs.state.goal = v

        v = _get("prev_goal")
        if v:
            cs._prev_goal = v

        v = _get("last_replay")
        if v:
            try:
                cs._last_replay = int(v)
            except Exception:
                pass

        v = _get("engrams")
        if v:
            try:
                cs._engrams.update(
                    {int(k): int(cnt) for k, cnt in json.loads(v).items()}
                )
            except Exception:
                pass

        # ── Restore episodic memory (with causal fields) ─────
        try:
            import numpy as np

            from consciousness import EpisodicEvent

            ep_rows = conn.execute(
                "SELECT tick, kind, content, emotion, person_id,"
                " prediction, action, observed_outcome, causal_update,"
                " social_person_ids, phenomenal_vector"
                " FROM episodic ORDER BY id"
            ).fetchall()
            for row in ep_rows:
                _sp_ids = ()
                if len(row) > 9 and row[9]:
                    try:
                        _sp_ids = tuple(json.loads(row[9]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                _phenom = None
                if len(row) > 10 and row[10]:
                    try:
                        _phenom = np.array(json.loads(row[10]), dtype=np.float32)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                cs.episodic._events.append(
                    EpisodicEvent(
                        tick=row[0],
                        kind=row[1],
                        content=row[2],
                        emotion_snapshot=row[3],
                        social_person_id=row[4] if len(row) > 4 else None,
                        social_person_ids=_sp_ids,
                        prediction=row[5] if len(row) > 5 else "",
                        action=row[6] if len(row) > 6 else "",
                        observed_outcome=row[7] if len(row) > 7 else "",
                        causal_update=row[8] if len(row) > 8 else "",
                        phenomenal_vector=_phenom,
                    )
                )
        except Exception:
            pass

        # ── Restore self-model ────────────────────────────────
        v = _get("self_model")
        if v:
            try:
                sm_data = json.loads(v)
                sm = cs.self_model
                sm.name = sm_data.get("name", sm.name)
                sm.birth_tick = sm_data.get("birth_tick", sm.birth_tick)
                sm.total_spikes = sm_data.get("total_spikes", sm.total_spikes)
                sm.concepts_learned = sm_data.get(
                    "concepts_learned", sm.concepts_learned
                )
                sm.ignitions_total = sm_data.get("ignitions_total", sm.ignitions_total)
                sm.known_strengths = sm_data.get("known_strengths", sm.known_strengths)
                sm.known_gaps = sm_data.get("known_gaps", sm.known_gaps)
            except Exception:
                pass

        # ── Restore personality ───────────────────────────────
        v = _get("personality_exposure")
        if v:
            try:
                cs.personality._exposure.update(json.loads(v))
            except Exception:
                pass

        v = _get("personality_traits")
        if v:
            try:
                cs.personality.active_traits = json.loads(v)
            except Exception:
                pass

        # ── Restore meta-cognition ────────────────────────────
        v = _get("meta_familiarity")
        if v:
            try:
                cs.meta._familiarity.update(json.loads(v))
            except Exception:
                pass

        v = _get("meta_depth")
        if v:
            try:
                cs.meta._depth.update(json.loads(v))
            except Exception:
                pass

        # ── Restore comm drive ────────────────────────────────
        v = _get("comm_drive")
        if v:
            try:
                cd_data = json.loads(v)
                cs.comm_drive.drive = float(cd_data.get("drive", 0.0))
                cs.comm_drive._last_spoke = int(cd_data.get("_last_spoke", -400))
            except Exception:
                pass
        # ── Restore BeliefStore (explicit deserialization) ───────
        v = _get("belief_store")
        if v:
            try:
                from consciousness import BeliefEntry, EpistemicStatus

                raw = json.loads(v)
                for subj, rels in raw.items():
                    if subj not in cs.belief_store._beliefs:
                        cs.belief_store._beliefs[subj] = {}
                    for rel, objs in rels.items():
                        if rel not in cs.belief_store._beliefs[subj]:
                            cs.belief_store._beliefs[subj][rel] = {}
                        for obj, entry_data in objs.items():
                            if isinstance(entry_data, dict):
                                cs.belief_store._beliefs[subj][rel][obj] = BeliefEntry(
                                    confidence=entry_data.get("confidence", 0.5),
                                    source=entry_data.get("source", ""),
                                    first_tick=entry_data.get("first_tick", 0),
                                    last_tick=entry_data.get("last_tick", 0),
                                    evidence_count=entry_data.get("evidence_count", 1),
                                    contradiction_count=entry_data.get(
                                        "contradiction_count", 0
                                    ),
                                    epistemic_status=EpistemicStatus(
                                        entry_data.get("epistemic_status", "belief")
                                    ),
                                )
                cs.belief_store._total = sum(
                    len(objs)
                    for rels in cs.belief_store._beliefs.values()
                    for objs in rels.values()
                )
            except Exception:
                pass

        # ── Restore quarantined beliefs ─────────────────────────────
        v = _get("belief_quarantine")
        if v:
            try:
                from consciousness import BeliefEntry, EpistemicStatus

                raw = json.loads(v)
                for key, entry_data in raw.items():
                    parts = key.split("|", 2)
                    if len(parts) == 3:
                        s, r, o = parts
                        cs.belief_store._quarantine[(s, r, o)] = BeliefEntry(
                            confidence=entry_data.get("confidence", 0.5),
                            source=entry_data.get("source", ""),
                            first_tick=entry_data.get("first_tick", 0),
                            last_tick=entry_data.get("last_tick", 0),
                            evidence_count=entry_data.get("evidence_count", 1),
                            contradiction_count=entry_data.get(
                                "contradiction_count", 0
                            ),
                            epistemic_status=EpistemicStatus(
                                entry_data.get("epistemic_status", "belief")
                            ),
                        )
            except Exception:
                pass

        # ── Restore language preference ──────────────────────────────
        v = _get("lang_pref")
        if v and v in ("de", "en"):
            try:
                cs.lang._lang = v
            except Exception:
                pass

        # ── Restore PersonModel (social memory) ─────────────────────
        try:
            from social_manager import PersonModel as _PM

            pm_rows = conn.execute(
                "SELECT person_id, trust, familiarity, total_encounters, "
                "total_words_heard, total_words_spoken, avg_valence, "
                "last_encounter_tick, relationship_type, relationship_confidence, relationship_scores, relationship_history, preferences, inferred_interests, "
                "interaction_log FROM person_models"
            ).fetchall()
            sm_obj = brain._social_manager
            for row in pm_rows:
                pm = _PM(
                    person_id=row[0],
                    trust=row[1],
                    familiarity=row[2],
                    total_encounters=row[3],
                    total_words_heard=row[4],
                    total_words_spoken=row[5],
                    avg_valence=row[6],
                    last_encounter_tick=row[7],
                    relationship_type=row[8] or "emergent_contact",
                    relationship_confidence=row[9] if row[9] is not None else 0.35,
                    relationship_scores=json.loads(row[10] or "{}"),
                    relationship_history=json.loads(row[11] or "[]"),
                    preferences=json.loads(row[12]),
                    inferred_interests=json.loads(row[13]),
                    interaction_log=json.loads(row[14]),
                )
                sm_obj._person_models[pm.person_id] = pm
        except Exception:
            pass

        # ── Restore autobiographic chapters + guidelines + consistency ─
        v = _get("auto_chapters")
        if v:
            try:
                for ch in json.loads(v):
                    cs.autobiography._chapters.append(ch)
            except Exception:
                pass
        v = _get("auto_identity_summary")
        if v:
            try:
                cs.autobiography._identity_summary = v
            except Exception:
                pass
        # Restore guidelines with full metadata
        v = _get("auto_guidelines")
        if v:
            try:
                from consciousness import PersonalGuideline

                for gl_data in json.loads(v):
                    cs.autobiography._guidelines.append(
                        PersonalGuideline(
                            text=gl_data["text"],
                            source=gl_data.get("source", ""),
                            strength=gl_data.get("strength", 0.5),
                            tick_born=gl_data.get("tick_born", 0),
                        )
                    )
            except Exception:
                pass
        # Restore identity consistency
        v = _get("auto_identity_consistency")
        if v:
            try:
                cs.autobiography._identity_consistency = float(v)
            except Exception:
                pass
        # Restore recurring stats (for guideline derivation continuity)
        v = _get("auto_recurring_goals")
        if v:
            try:
                cs.autobiography._recurring_goals.update(json.loads(v))
            except Exception:
                pass
        v = _get("auto_recurring_emotions")
        if v:
            try:
                cs.autobiography._recurring_emotions.update(json.loads(v))
            except Exception:
                pass
        v = _get("auto_stable_topics")
        if v:
            try:
                cs.autobiography._stable_topics.update(json.loads(v))
            except Exception:
                pass
        v = _get("auto_social_encounters")
        if v:
            try:
                for k, n in json.loads(v).items():
                    cs.autobiography._social_encounters[int(k)] = n
            except Exception:
                pass
        v = _get("auto_relationship_types")
        if v:
            try:
                cs.autobiography._relationship_types.update(
                    {str(k): int(n) for k, n in json.loads(v).items()}
                )
            except Exception:
                pass

        # ── Restore world model ──────────────────────────────────
        v = _get("world_model")
        if v:
            try:
                raw = json.loads(v)
                for key, entry in raw.items():
                    parts = key.split("|", 1)
                    if len(parts) == 2:
                        cs.world_model._entries[(parts[0], parts[1])] = entry
            except Exception:
                pass

        # ── Restore continuity monitor ───────────────────────────
        v = _get("continuity_monitor")
        if v:
            try:
                cm_data = json.loads(v)
                cs.continuity.memory_coherence = cm_data.get("memory_coherence", 1.0)
                cs.continuity.agency_stability = cm_data.get("agency_stability", 1.0)
                cs.continuity.value_stability = cm_data.get("value_stability", 1.0)
            except Exception:
                pass

        # ── Restore sandbox planner lessons ───────────────────────
        v = _get("sandbox_lessons")
        if v:
            try:
                for lesson in json.loads(v):
                    cs.sandbox_planner._postmortem_lessons.append(lesson)
            except Exception:
                pass

        # ── Restore ExperienceAppraisal ──────────────────────────────
        v = _get("experience_appraisal")
        if v:
            try:
                ea_data = json.loads(v)
                ea = brain._emotion_engine.experience
                ea._associations.update(ea_data.get("associations", {}))
                ea._observation_count.update(
                    {k: int(c) for k, c in ea_data.get("observation_count", {}).items()}
                )
            except Exception:
                pass

        # ── Restore operative embodiment state ────────────────────
        v = _get("embodied_self")
        if v:
            try:
                es_data = json.loads(v)
                es = cs.embodied_self
                for k in (
                    "battery_estimate",
                    "motor_readiness",
                    "visual_contact",
                    "proximity_alert",
                    "social_presence",
                    "focus_x",
                    "focus_y",
                    "focus_size",
                ):
                    if k in es_data:
                        setattr(es, k, float(es_data[k]))
                for k in ("posture", "agency_window", "focus_target"):
                    if k in es_data:
                        setattr(es, k, str(es_data[k]))
            except Exception:
                pass

        v = _get("robot_state")
        if v:
            try:
                rs_data = json.loads(v)
                rs = cs.robot_state
                for k in (
                    "head_yaw",
                    "head_pitch",
                    "engagement_level",
                    "imitation_readiness",
                    "target_x",
                    "target_y",
                ):
                    if k in rs_data:
                        setattr(rs, k, float(rs_data[k]))
                for k in (
                    "gaze_target",
                    "torso_mode",
                    "left_arm_mode",
                    "right_arm_mode",
                    "left_gripper",
                    "right_gripper",
                    "locomotion_mode",
                    "interaction_zone",
                ):
                    if k in rs_data:
                        setattr(rs, k, str(rs_data[k]))
            except Exception:
                pass

        v = _get("task_frame")
        if v:
            try:
                tf_data = json.loads(v)
                tf = cs.task_frame
                for k in (
                    "active_task",
                    "operational_goal",
                    "current_step",
                    "last_result",
                    "mode",
                    "plan_phase",
                ):
                    if k in tf_data:
                        setattr(tf, k, str(tf_data[k]))
                for k in ("progress", "confidence"):
                    if k in tf_data:
                        setattr(tf, k, float(tf_data[k]))
                tf.subgoals = tf_data.get("subgoals", [])
                tf.blockers = tf_data.get("blockers", [])
            except Exception:
                pass

        v = _get("sensorimotor_model")
        if v:
            try:
                sm_data = json.loads(v)
                sm_fw = cs.sensorimotor
                for key, m in sm_data.get("models", {}).items():
                    parts = key.split("|", 1)
                    if len(parts) == 2:
                        sm_fw._models[(parts[0], parts[1])] = m
                sm_fw._prediction_error = float(sm_data.get("prediction_error", 0.0))
                sm_fw._agency_confirmation = float(
                    sm_data.get("agency_confirmation", 1.0)
                )
            except Exception:
                pass

        v = _get("goal_system_veto_count")
        if v:
            try:
                cs.goal_system._veto_count = int(v)
            except Exception:
                pass

        # ── Restore causal graph ─────────────────────────────────
        v = _get("causal_graph")
        if v:
            try:
                cs.causal_graph.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore value model ──────────────────────────────────
        v = _get("value_model")
        if v:
            try:
                cs.value_model.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore identity arc ─────────────────────────────────
        v = _get("identity_arc")
        if v:
            try:
                cs.identity_arc.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore narrative thread ─────────────────────────────
        v = _get("narrative_thread")
        if v:
            try:
                cs.narrative_thread.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore theory of mind ───────────────────────────────
        v = _get("theory_of_mind")
        if v:
            try:
                cs.theory_of_mind.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore belief quarantine module ─────────────────────
        v = _get("belief_quarantine_module")
        if v:
            try:
                cs.belief_quarantine.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore attention controller ─────────────────────────
        v = _get("attention_ctrl")
        if v:
            try:
                cs.attention_ctrl.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore long-horizon goals ───────────────────────────
        v = _get("long_horizon")
        if v:
            try:
                cs.long_horizon.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore concept salience ─────────────────────────────
        v = _get("concept_salience")
        if v:
            try:
                cs._concept_salience = json.loads(v)
            except Exception:
                pass

        # ── Restore active scene anchors ─────────────────────────
        v = _get("active_scene")
        if v:
            try:
                sd = json.loads(v)
                _as = cs.active_scene
                _as.focus_person = sd.get("focus_person", "")
                _as.focus_object = sd.get("focus_object", "")
                _as.focus_zone = sd.get("focus_zone", "public")
                _as.active_relations = sd.get("active_relations", [])
                _as.scene_confidence = sd.get("scene_confidence", 0.0)
                _as.last_update_tick = sd.get("last_update_tick", 0)
            except Exception:
                pass

        # ── Re-wire epistemic gate (not persisted, runtime ref only) ─────────
        cs.belief_store._epistemic_gate = cs.belief_quarantine
        # ── Re-wire grounded memory (not persisted, runtime ref only) ────
        cs.belief_store._grounded_memory = cs.grounded_memory

        # ── Restore unified self-state ───────────────────────────────
        v = _get("unified_self")
        if v:
            try:
                from consciousness import UnifiedSelfState

                cs.unified_self = UnifiedSelfState.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore grounded semantic memory ─────────────────────────
        v = _get("grounded_memory")
        if v:
            try:
                from consciousness import GroundedSemanticMemory

                cs.grounded_memory = GroundedSemanticMemory.from_dict(json.loads(v))
                cs.belief_store._grounded_memory = cs.grounded_memory
            except Exception:
                pass

        # ── Restore phenomenal buffer ────────────────────────────────
        v = _get("phenomenal_buffer")
        if v:
            try:
                from consciousness import PhenomenalBuffer

                cs.phenomenal_buffer = PhenomenalBuffer.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore learned world model (RSSM) ──────────────────────
        v = _get("learned_world")
        if v:
            try:
                from consciousness import LearnedWorldModel

                cs.learned_world = LearnedWorldModel.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore model-based planner ──────────────────────────────
        v = _get("model_planner")
        if v:
            try:
                from consciousness import ModelBasedPlanner

                cs.model_planner = ModelBasedPlanner.from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore self-model dynamic state ─────────────────────────
        v = _get("self_model_dynamic")
        if v:
            try:
                from consciousness import SelfThesis

                sd = json.loads(v)
                sm = cs.self_model
                sm.energy = sd.get("energy", sm.energy)
                sm.uncertainty = sd.get("uncertainty", sm.uncertainty)
                sm.identity_stability = sd.get(
                    "identity_stability", sm.identity_stability
                )
                sm.agency_score = sd.get("agency_score", sm.agency_score)
                sm.self_awareness_level = sd.get(
                    "self_awareness_level", sm.self_awareness_level
                )
                sm.ownership_confidence = sd.get(
                    "ownership_confidence", sm.ownership_confidence
                )
                sm.agency_confidence = sd.get("agency_confidence", sm.agency_confidence)
                sm.continuity_estimate = sd.get(
                    "continuity_estimate", sm.continuity_estimate
                )
                sm.self_tensions = list(sd.get("self_tensions", sm.self_tensions))
                sm.self_narrative = sd.get("self_narrative", sm.self_narrative)
                sm.self_beliefs = dict(sd.get("self_beliefs", sm.self_beliefs))
                sm.self_theses = {
                    th.get("domain", f"thesis_{idx}"): SelfThesis.from_dict(th)
                    for idx, th in enumerate(sd.get("self_theses", []))
                    if isinstance(th, dict)
                }
                sm.self_contradictions = list(
                    sd.get("self_contradictions", sm.self_contradictions)
                )
                sm.last_action_self_report = sd.get(
                    "last_action_self_report", sm.last_action_self_report
                )
                sm.body_pose = sd.get("body_pose", sm.body_pose)
                sm.body_load = sd.get("body_load", sm.body_load)
                sm.body_pain = sd.get("body_pain", sm.body_pain)
                sm.balance = sd.get("balance", sm.balance)
                sm.last_skill = sd.get("last_skill", sm.last_skill)
                sm.turn_state = sd.get("turn_state", sm.turn_state)
            except Exception:
                pass

        # ── Restore dialogue manager state ───────────────────────────
        v = _get("dialogue_manager")
        if v:
            try:
                brain._dialogue_manager.load_from_dict(json.loads(v))
            except Exception:
                pass

        # ── Restore world state snapshot ─────────────────────────────
        v = _get("world_state")
        if v:
            try:
                brain._world_state.from_dict(
                    json.loads(v), current_tick=brain.tick_count
                )
            except Exception:
                pass

        conn.close()
        _last_load_report = report
        return restored

    except Exception as _exc:
        report.record_issue("load:fatal", _exc, severity="critical")
        _last_load_report = report
        _log.error("load_brain fatal: %s", _exc)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────


def db_stats(db_path: str = DB_PATH) -> dict:
    """Return basic stats about the saved brain state."""
    if not os.path.exists(db_path):
        return {"exists": False}
    try:
        conn = sqlite3.connect(db_path)
        n_syn = conn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]
        n_ep = (
            conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
            if _table_exists(conn, "episodic")
            else 0
        )
        saved = conn.execute("SELECT val FROM meta WHERE key='saved_at'").fetchone()
        ticks = conn.execute("SELECT val FROM meta WHERE key='tick_count'").fetchone()
        conc = conn.execute("SELECT val FROM meta WHERE key='concepts'").fetchone()
        n_conc = len(json.loads(conc[0])) if conc else 0
        bs_row = conn.execute(
            "SELECT val FROM meta WHERE key='belief_store'"
        ).fetchone()
        n_beliefs = (
            sum(
                len(objs)
                for rels in json.loads(bs_row[0]).values()
                for objs in rels.values()
            )
            if bs_row
            else 0
        )
        lang_row = conn.execute("SELECT val FROM meta WHERE key='lang_pref'").fetchone()
        n_persons = (
            conn.execute("SELECT COUNT(*) FROM person_models").fetchone()[0]
            if _table_exists(conn, "person_models")
            else 0
        )
        conn.close()
        return {
            "exists": True,
            "synapses": n_syn,
            "episodic": n_ep,
            "concepts": n_conc,
            "beliefs": n_beliefs,
            "persons": n_persons,
            "lang_pref": lang_row[0] if lang_row else "de",
            "saved_at": float(saved[0]) if saved else 0.0,
            "tick_count": int(ticks[0]) if ticks else 0,
        }
    except Exception:
        return {"exists": True, "error": "unreadable"}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()[0]
        > 0
    )
