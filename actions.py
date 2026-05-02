"""
actions.py — AI Action Toolbelt
================================
Gives the AI the ability to act autonomously on the host PC.

The AI decides WHEN to act based on internal state (communication drive,
goal, curiosity) — not on a timer or random trigger.

Available tools
───────────────
  web_search(query)      — DuckDuckGo instant-answer search
  fetch_page(url)        — Fetch and summarise a webpage
  read_file(path)        — Read a file from the filesystem
  write_note(text)       — Append a thought/note to ai_notes.txt
  open_app(name)         — Open an application (browser, notepad, …)
  run_program(cmd, args) — Run a whitelisted program with arguments
  system_info()          — Query CPU/RAM/disk/time info
  list_files(path)       — List files in a directory

Safety model
────────────
  • SAFE operations  → execute immediately, just logged
  • CAUTION ops      → execute but logged prominently
  • DANGEROUS ops    → stored in pending_approvals; execute only after
                       brain.approve_action(action_id) is called

The AI builds its action queue from genuine internal signals:
  • goal == "explore"     → web search / fetch
  • meta-cognitive gap    → web search for the gap concept
  • high curiosity+concept → fetch related page
  • insight               → write note
  • goal == "respond"     → open app or speak
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger("actions")

# ─────────────────────────────────────────────────────────────────────────────
# Action dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Action:
    id: int
    kind: str  # "web_search" | "fetch_page" | "read_file" | ...
    args: Dict  # tool-specific arguments
    reason: str  # why the AI wants to do this
    safety: str  # "safe" | "caution" | "dangerous"
    status: str = "pending"  # "pending" | "done" | "approved" | "denied"
    result: str = ""  # raw output / error
    tick: int = 0  # brain tick when action was queued

    def describe(self) -> str:
        return f"[{self.kind}] {json.dumps(self.args)} ({self.safety})"


# ─────────────────────────────────────────────────────────────────────────────
# Safety classification
# ─────────────────────────────────────────────────────────────────────────────

# Programs on this list can be launched directly
_APP_WHITELIST = {
    "notepad",
    "notepad.exe",
    "explorer",
    "explorer.exe",
    "calc",
    "calc.exe",
    "mspaint",
    "mspaint.exe",
    "chrome",
    "chrome.exe",
    "firefox",
    "firefox.exe",
    "msedge",
    "msedge.exe",
    "code",
    "code.exe",  # VS Code
    "vlc",
    "vlc.exe",
}

# These command prefixes are always dangerous
_DANGEROUS_PREFIXES = (
    "rm ",
    "del ",
    "format ",
    "shutdown",
    "reboot",
    "reg ",
    "regedit",
    "powershell -enc",
    "cmd /c",
    "taskkill",
    "netsh",
    "attrib",
)


def _classify_safety(kind: str, args: Dict) -> str:
    if kind in (
        "web_search",
        "fetch_page",
        "system_info",
        "list_files",
        "look_at",
        "set_pose",
        "mirror_gesture",
        "track_person",
    ):
        return "safe"
    if kind == "read_file":
        # Reading outside the project folder = caution
        path = args.get("path", "")
        if os.path.abspath(path).startswith(os.path.abspath(".")):
            return "safe"
        return "caution"
    if kind == "write_note":
        return "safe"
    if kind == "open_app":
        app = args.get("app", "").lower()
        return "safe" if app in _APP_WHITELIST else "caution"
    if kind == "run_program":
        cmd = args.get("cmd", "").lower()
        if any(cmd.startswith(p) for p in _DANGEROUS_PREFIXES):
            return "dangerous"
        return "caution"
    return "caution"


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────


def _tool_web_search(args: Dict) -> str:
    """DuckDuckGo instant-answer API — no browser, no cookies."""
    import urllib.parse
    import urllib.request

    query = args.get("query", "")
    if not query:
        return "No query."
    url = (
        "https://api.duckduckgo.com/?q="
        + urllib.parse.quote(query)
        + "&format=json&no_html=1&skip_disambig=1"
    )
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        abstract = data.get("AbstractText", "")
        answer = data.get("Answer", "")
        topics = [t.get("Text", "") for t in data.get("RelatedTopics", [])[:3]]
        result = abstract or answer or "; ".join(t for t in topics if t)
        return result[:600] if result else "No direct answer found."
    except Exception as e:
        return f"Search error: {e}"


def _tool_fetch_page(args: Dict) -> str:
    """Download and extract plain text from a URL."""
    import urllib.request

    url = args.get("url", "")
    if not url:
        return "No URL."
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(30_000).decode("utf-8", errors="replace")
        # Strip HTML tags
        import re

        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:800]
    except Exception as e:
        return f"Fetch error: {e}"


def _tool_read_file(args: Dict) -> str:
    """Read up to 4KB from a local file."""
    path = args.get("path", "")
    if not path or not os.path.isfile(path):
        return f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(4096)
    except Exception as e:
        return f"Read error: {e}"


def _tool_write_note(args: Dict) -> str:
    """Append a note to ai_notes.txt in the project folder."""
    text = args.get("text", "")
    if not text:
        return "Nothing to write."
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {text}\n"
    try:
        with open("ai_notes.txt", "a", encoding="utf-8") as f:
            f.write(line)
        return f"Note saved ({len(text)} chars)."
    except Exception as e:
        return f"Write error: {e}"


def _tool_open_app(args: Dict) -> str:
    """Open an application by name."""
    app = args.get("app", "")
    if not app:
        return "No app specified."
    try:
        subprocess.Popen(
            app, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return f"Opened: {app}"
    except Exception as e:
        return f"Open error: {e}"


def _tool_run_program(args: Dict) -> str:
    """Run a program and capture its stdout (max 2s, 4KB)."""
    cmd = args.get("cmd", "")
    pargs = args.get("args", [])
    if not cmd:
        return "No command."
    try:
        result = subprocess.run(
            [cmd] + list(pargs),
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
        out = (result.stdout + result.stderr)[:2048]
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Timeout."
    except Exception as e:
        return f"Run error: {e}"


def _tool_system_info(_args: Dict) -> str:
    """Return basic system info without external dependencies."""
    import platform

    info = {
        "os": platform.system() + " " + platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": os.getcwd(),
    }
    try:
        import shutil

        total, used, free = shutil.disk_usage(".")
        info["disk_free_gb"] = round(free / 1e9, 1)
    except Exception:
        pass
    return json.dumps(info, ensure_ascii=False)


def _tool_list_files(args: Dict) -> str:
    """List files in a directory."""
    path = args.get("path", ".")
    try:
        entries = os.listdir(path)
        return "\n".join(entries[:80])
    except Exception as e:
        return f"Error: {e}"


def _tool_look_at(args: Dict) -> str:
    """Simulated robot gaze command for later InMoov integration."""
    target = args.get("target", "none")
    yaw = float(args.get("yaw", 0.0))
    pitch = float(args.get("pitch", 0.0))
    return f"ROBOT_CMD gaze target={target} yaw={yaw:+.2f} pitch={pitch:+.2f}"


def _tool_set_pose(args: Dict) -> str:
    """Simulated posture/arm command."""
    pose = args.get("pose", "idle")
    arms = args.get("arms", "parked")
    hands = args.get("hands", "open")
    return f"ROBOT_CMD pose={pose} arms={arms} hands={hands}"


def _tool_mirror_gesture(args: Dict) -> str:
    """Simulated gesture imitation command."""
    gesture = args.get("gesture", "neutral")
    intensity = float(args.get("intensity", 0.5))
    return f"ROBOT_CMD mirror gesture={gesture} intensity={intensity:.2f}"


def _tool_track_person(args: Dict) -> str:
    """Simulated person-tracking command."""
    mode = args.get("mode", "soft")
    zone = args.get("zone", "social")
    return f"ROBOT_CMD track_person mode={mode} zone={zone}"


_TOOLS: Dict[str, Callable] = {
    "web_search": _tool_web_search,
    "fetch_page": _tool_fetch_page,
    "read_file": _tool_read_file,
    "write_note": _tool_write_note,
    "open_app": _tool_open_app,
    "run_program": _tool_run_program,
    "system_info": _tool_system_info,
    "list_files": _tool_list_files,
    "look_at": _tool_look_at,
    "set_pose": _tool_set_pose,
    "mirror_gesture": _tool_mirror_gesture,
    "track_person": _tool_track_person,
}


# ─────────────────────────────────────────────────────────────────────────────
# Action Toolbelt — orchestrates action generation and execution
# ─────────────────────────────────────────────────────────────────────────────


class ActionToolbelt:
    """
    Receives internal-state signals from the consciousness tick and
    decides which tools to invoke.  Safe actions run immediately;
    dangerous ones are queued for approval.

    Call toolbelt.tick(brain, cs_state) once per consciousness tick.
    Results are injected back as text into the brain's sensory pathway
    so the AI 'reads' what it found.
    """

    _COOLDOWN = 300  # min ticks between action invocations
    _MAX_QUEUE = 20  # max pending actions

    def __init__(self) -> None:
        self._action_id: int = 0
        self._last_acted: int = -self._COOLDOWN
        self.history: Deque[Action] = deque(maxlen=200)
        self.pending_approvals: List[Action] = []  # dangerous, awaiting user
        self._last_result: str = ""
        self._brain: object = None  # set by Brain.__init__ after construction

    # ─────────────────────────────────────────────────────────

    def tick(self, tick_count: int, em, cs_state, cs_core) -> Optional[str]:
        """
        Evaluate internal state and optionally queue + run an action.
        Returns a result string if an action was executed, else None.
        """
        if tick_count - self._last_acted < self._COOLDOWN:
            return None

        action = self._decide(tick_count, em, cs_state, cs_core)
        if action is None:
            return None

        self.history.append(action)
        self._last_acted = tick_count

        if action.safety == "dangerous":
            self.pending_approvals.append(action)
            action.status = "pending_approval"
            logger.warning("ACTION APPROVAL REQUIRED: %s", action.describe())
            return f"[ACTION QUEUED — needs approval] {action.describe()}"

        # Execute immediately
        return self._execute(action)

    def approve_action(self, action_id: int) -> str:
        """User approves a pending dangerous action. Returns result."""
        for a in self.pending_approvals:
            if a.id == action_id:
                self.pending_approvals.remove(a)
                return self._execute(a)
        return f"No action with id={action_id} found."

    def deny_action(self, action_id: int) -> None:
        """User denies a pending action."""
        for a in self.pending_approvals:
            if a.id == action_id:
                a.status = "denied"
                self.pending_approvals.remove(a)
                return

    # ─────────────────────────────────────────────────────────

    def _new_id(self) -> int:
        self._action_id += 1
        return self._action_id

    def _mk_action(self, tick_count: int, kind: str, args: Dict, reason: str) -> Action:
        """Create a fully-classified Action in one place."""
        return Action(
            id=self._new_id(),
            tick=tick_count,
            kind=kind,
            args=args,
            reason=reason,
            safety=_classify_safety(kind, args),
        )

    def _task_first_action(
        self, tick_count: int, em, cs_state, cs_core
    ) -> Optional[Action]:
        """Prefer explicit task-frame driven actions over loose heuristics."""
        task = getattr(cs_core, "task_frame", None)
        body = getattr(cs_core, "body", None)
        embodied = getattr(cs_core, "embodied_self", None)
        robot = getattr(cs_core, "robot_state", None)
        if task is None:
            return None

        concepts = list(getattr(cs_core, "_concepts", []))
        conclusions = list(getattr(cs_core, "_conclusions", []))
        concept = concepts[-1] if concepts else ""
        blocker_set = set(task.blockers)
        urgency = body.homeostatic_urgency() if body is not None else 0.0
        learned_policy_action = getattr(task, "controller_policy_action", "")
        learned_policy_conf = float(
            getattr(task, "controller_policy_confidence", 0.0) or 0.0
        )

        # 0. Robot-first orientation: embodied attention before abstract tool use
        if (
            robot is not None
            and embodied is not None
            and embodied.social_presence > 0.5
        ):
            if (
                learned_policy_action
                and learned_policy_conf >= 0.58
                and "low_controller_success" not in blocker_set
            ):
                if (
                    learned_policy_action == "mirror_gesture"
                    and embodied.current_gesture
                ):
                    return self._mk_action(
                        tick_count,
                        "mirror_gesture",
                        {
                            "gesture": embodied.current_gesture,
                            "intensity": max(
                                robot.imitation_readiness, learned_policy_conf
                            ),
                        },
                        f"Reusing learned robot policy '{learned_policy_action}' with confidence {learned_policy_conf:.2f}",
                    )
                if learned_policy_action == "look_at" and robot.gaze_target == "person":
                    return self._mk_action(
                        tick_count,
                        "look_at",
                        {
                            "target": robot.gaze_target,
                            "yaw": robot.head_yaw,
                            "pitch": robot.head_pitch,
                        },
                        f"Reusing learned robot policy '{learned_policy_action}' with confidence {learned_policy_conf:.2f}",
                    )
                if learned_policy_action == "track_person":
                    mode = "firm" if learned_policy_conf >= 0.72 else "soft"
                    return self._mk_action(
                        tick_count,
                        "track_person",
                        {"mode": mode, "zone": robot.interaction_zone},
                        f"Reusing learned robot policy '{learned_policy_action}' with confidence {learned_policy_conf:.2f}",
                    )
                if learned_policy_action == "set_pose":
                    arms = "gesture_ready" if embodied.current_gesture else "stabilise"
                    return self._mk_action(
                        tick_count,
                        "set_pose",
                        {
                            "pose": "engaged_idle",
                            "arms": arms,
                            "hands": robot.left_gripper,
                        },
                        f"Reusing learned robot policy '{learned_policy_action}' with confidence {learned_policy_conf:.2f}",
                    )
            if embodied.current_gesture and robot.imitation_readiness > 0.55:
                return self._mk_action(
                    tick_count,
                    "mirror_gesture",
                    {
                        "gesture": embodied.current_gesture,
                        "intensity": robot.imitation_readiness,
                    },
                    f"Embodied imitation opportunity detected for gesture '{embodied.current_gesture}'",
                )
            if robot.gaze_target == "person":
                return self._mk_action(
                    tick_count,
                    "look_at",
                    {
                        "target": robot.gaze_target,
                        "yaw": robot.head_yaw,
                        "pitch": robot.head_pitch,
                    },
                    "Maintaining social gaze alignment with the currently perceived person",
                )
            if task.active_task in ("model_human_presence", "sustain_social_exchange"):
                return self._mk_action(
                    tick_count,
                    "track_person",
                    {"mode": "soft", "zone": robot.interaction_zone},
                    f"Maintaining person tracking while task '{task.active_task}' is active",
                )

        # 1. Low grounding during a social exchange -> gather information
        if (
            task.active_task == "sustain_social_exchange"
            and "low_semantic_grounding" in blocker_set
        ):
            query = concept or (
                embodied.last_user_utterance if embodied else "current interaction"
            )
            if query:
                return self._mk_action(
                    tick_count,
                    "web_search",
                    {"query": query},
                    f"Task '{task.active_task}' lacks grounding; collecting external context for '{query}'",
                )

        # 1b. Social imitation readiness low -> gather world/body context, not act outwardly yet
        if robot is not None and "low_imitation_readiness" in blocker_set:
            return self._mk_action(
                tick_count,
                "system_info",
                {},
                f"Robot state not ready for close interaction in zone '{robot.interaction_zone}'; sampling internal/external status first",
            )

        # 2. Expand world model -> search current focus concept
        if task.active_task == "expand_world_model" and concept and em.curiosity > 0.28:
            return self._mk_action(
                tick_count,
                "web_search",
                {"query": concept},
                f"Task '{task.active_task}' is active; probing concept '{concept}'",
            )

        # 3. Integrate recent experience -> write structured note
        if task.active_task == "integrate_recent_experience" and conclusions:
            note = (
                f"[Task={task.active_task} step={task.current_step}] "
                f"goal={task.operational_goal}; blocker={', '.join(task.blockers) or 'none'}; "
                f"latest={conclusions[-1][:140]}"
            )
            return self._mk_action(
                tick_count,
                "write_note",
                {"text": note},
                f"Consolidating recent experience for task '{task.active_task}'",
            )

        # 4. Preserve internal continuity / low energy -> lightweight self-observation
        if urgency > 0.72 or "low_energy" in blocker_set:
            return self._mk_action(
                tick_count,
                "set_pose",
                {"pose": "protective_idle", "arms": "parked", "hands": "open"},
                "High homeostatic urgency; shifting robot body into low-risk protective posture",
            )

        # 4b. Socially engaged robot with gesture present -> enrich context around imitation target
        if (
            robot is not None
            and embodied is not None
            and robot.imitation_readiness > 0.45
            and embodied.current_gesture
        ):
            return self._mk_action(
                tick_count,
                "web_search",
                {
                    "query": f"human gesture {embodied.current_gesture} meaning interaction"
                },
                f"Robot is socially engaged and sees gesture '{embodied.current_gesture}'; enriching imitation context",
            )

        # 5. User-addressed response but no current user text -> avoid noisy actions
        if task.active_task == "form_grounded_response" and task.needs_user_input:
            return None

        return None

    def _decide(self, tick_count: int, em, cs_state, cs_core) -> Optional["Action"]:
        """
        Choose an action based on internal state signals.
        Returns an Action, or None if no action is warranted.
        """
        task_action = self._task_first_action(tick_count, em, cs_state, cs_core)
        if task_action is not None:
            return task_action

        goal = cs_state.goal
        concepts = list(cs_core._concepts)
        concept = concepts[-1] if concepts else None
        gaps = cs_core.meta.gaps(2)

        # 1. Knowledge gap + curiosity  →  search for it
        if gaps and em.curiosity > 0.45:
            q = gaps[0]
            return self._mk_action(
                tick_count,
                "web_search",
                {"query": q},
                f"Knowledge gap detected: '{q}'",
            )

        # 2. Explore goal + fresh concept  →  search for it
        if goal == "explore" and concept and em.curiosity > 0.35:
            return self._mk_action(
                tick_count,
                "web_search",
                {"query": concept},
                f"Exploring concept '{concept}'",
            )

        # 3. Ignition + concept  →  deep-fetch a page
        if cs_state.ignition and concept:
            query = concept + " detailed explanation"
            url = (
                "https://en.wikipedia.org/w/index.php?search="
                + query.replace(" ", "+")
                + "&ns0=1"
            )
            return self._mk_action(
                tick_count,
                "fetch_page",
                {"url": url},
                f"Global ignition while thinking about '{concept}'",
            )

        # 4. High arousal + insight  →  write a note
        last_ins = cs_core.episodic.last_of_kind("insight")
        if last_ins and (tick_count - last_ins.tick) < 300 and em.arousal() > 0.50:
            return self._mk_action(
                tick_count,
                "write_note",
                {"text": f"[Insight at t={last_ins.tick}] {last_ins.content}"},
                "Recording a significant insight",
            )

        # 5. System curiosity (rare, every ~3000 ticks)
        if goal == "explore" and tick_count % 3000 < 5:
            return self._mk_action(
                tick_count,
                "system_info",
                {},
                "Periodic environment awareness",
            )

        return None

    def _execute(self, action: Action) -> str:
        tool = _TOOLS.get(action.kind)
        if tool is None:
            action.result = f"Unknown tool: {action.kind}"
            action.status = "done"
            return action.result
        try:
            action.result = tool(action.args)
        except Exception as e:
            action.result = f"Error: {e}"
        action.status = "done"
        self._last_result = action.result
        logger.info(
            "ACTION [%s] %s -> %s", action.kind, action.args, action.result[:80]
        )

        # ── Perception-action feedback loop ──────────────────────────────
        # Every action must change perception and update internal state.
        # Robot motor commands feed back through the sensory pathway so the
        # system observes the consequence of its own behaviour.
        _ROBOT_KINDS = {"look_at", "set_pose", "mirror_gesture", "track_person"}
        if action.kind in _ROBOT_KINDS and self._brain is not None:
            _br = self._brain
            # ── Point 3: Real perception-action loop ──────────────────────────
            # Route robot proprioceptive feedback ONLY through the sensory
            # pathway (sensory_w → thalamus → cortex).  No semantic shortcut:
            # the system must integrate the signal before comprehending it.
            # build a normalised 48-element proprioceptive activation vector
            # from the robot command args (angles → [0,1], strings → 0.5)
            try:
                _sw = getattr(_br, "sensory_w", None)
                if _sw is not None:
                    _n = len(_sw._exc_cache)
                    _vals = list(action.args.values())
                    _prop: list = []
                    for _v in _vals:
                        if isinstance(_v, (int, float)):
                            _prop.append(min(1.0, abs(float(_v)) / 180.0))
                        else:
                            _prop.append(0.5)
                    # extend or truncate to exactly _n elements
                    if len(_prop) < _n:
                        _prop += [0.0] * (_n - len(_prop))
                    else:
                        _prop = _prop[:_n]
                    _sw.inject([c * 18.0 for c in _prop])
            except Exception:
                pass
            # Update body schema with the motor command outcome
            try:
                _bs = getattr(_br, "_body_schema", None)
                if _bs is not None:
                    _bs.update_from_robot_command(action.kind, action.args)
            except Exception:
                pass

        return action.result

    # ─────────────────────────────────────────────────────────

    @property
    def recent_results(self) -> List[str]:
        return [
            f"[{a.kind}] {a.result[:60]}"
            for a in list(self.history)[-5:]
            if a.status == "done"
        ]
