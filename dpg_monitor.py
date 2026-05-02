from __future__ import annotations

import collections
import datetime
import queue
import signal
import sys
import threading
import time
from typing import Dict, List, Tuple

import dearpygui.dearpygui as dpg
import numpy as np

from brain import Brain
from persistence import db_stats

HISTORY = 180
DARK = (13, 13, 26)
PANEL = (18, 18, 42)
TEXT = (212, 212, 240)
GRID = (42, 42, 68)
ACCENT = (102, 170, 255)
REGION_NAMES = [
    "sensory_visual",
    "sensory_auditory",
    "sensory_web",
    "thalamus",
    "v1_visual",
    "a1_auditory",
    "association",
    "hippocampus",
    "amygdala",
    "prefrontal",
    "motor",
]
LABELS = {
    "sensory_visual": "Vision",
    "sensory_auditory": "Audio",
    "sensory_web": "Web",
    "thalamus": "Thalamus",
    "v1_visual": "V1",
    "a1_auditory": "A1",
    "association": "Assoc",
    "hippocampus": "Hipp",
    "amygdala": "Amy",
    "prefrontal": "PFC",
    "motor": "Motor",
}
KEY_REGIONS = {
    "association": (51, 153, 255),
    "hippocampus": (51, 204, 153),
    "amygdala": (255, 102, 68),
    "prefrontal": (255, 204, 0),
    "motor": (204, 102, 255),
}
EM_DIMS = [
    "joy",
    "stress",
    "curiosity",
    "calm",
    "sadness",
    "anger",
    "surprise",
    "fatigue",
]
EM_COLORS = {
    "joy": (255, 224, 102),
    "stress": (255, 119, 85),
    "curiosity": (68, 221, 255),
    "calm": (136, 204, 255),
    "sadness": (136, 136, 204),
    "anger": (255, 68, 68),
    "surprise": (255, 170, 68),
    "fatigue": (119, 153, 119),
}
HEAD_LABELS = {
    "head_yaw": "Head Yaw",
    "head_pitch": "Head Pitch",
    "neck_roll": "Neck Roll",
    "jaw": "Jaw",
    "eye_yaw": "Eye Yaw",
    "eye_pitch": "Eye Pitch",
    "left_upper_lid": "L Upper Lid",
    "left_lower_lid": "L Lower Lid",
    "right_upper_lid": "R Upper Lid",
    "right_lower_lid": "R Lower Lid",
}
ANAT_EDGES = [
    ("sensory_web", "association", 2.5),
    ("sensory_web", "hippocampus", 2.0),
    ("sensory_web", "amygdala", 2.0),
    ("sensory_visual", "thalamus", 1.5),
    ("sensory_auditory", "thalamus", 1.5),
    ("thalamus", "v1_visual", 2.0),
    ("thalamus", "a1_auditory", 2.0),
    ("v1_visual", "association", 1.5),
    ("a1_auditory", "association", 1.5),
    ("association", "hippocampus", 1.5),
    ("association", "amygdala", 1.5),
    ("hippocampus", "prefrontal", 1.5),
    ("amygdala", "prefrontal", 1.8),
    ("prefrontal", "motor", 1.8),
]
POS = {
    "sensory_visual": (-2.5, 1.5),
    "sensory_auditory": (-2.5, -1.5),
    "sensory_web": (-2.5, 0.0),
    "thalamus": (-1.5, 0.0),
    "v1_visual": (-0.8, 1.5),
    "a1_auditory": (-0.8, -1.5),
    "association": (0.0, 0.5),
    "hippocampus": (0.5, -0.5),
    "amygdala": (0.5, 1.5),
    "prefrontal": (1.5, 0.5),
    "motor": (2.5, 0.0),
}
SPATIAL_COLORS = {
    "sensory_visual": (120, 190, 255),
    "sensory_auditory": (120, 255, 190),
    "sensory_web": (180, 180, 255),
    "thalamus": (255, 215, 120),
    "v1_visual": (80, 170, 255),
    "a1_auditory": (80, 255, 170),
    "association": (70, 150, 255),
    "hippocampus": (70, 215, 155),
    "amygdala": (255, 110, 90),
    "prefrontal": (255, 210, 70),
    "motor": (205, 120, 255),
}

_INPUT_QUEUE: queue.Queue[str] = queue.Queue()


def input_thread() -> None:
    while True:
        try:
            line = sys.stdin.readline()
            if line:
                _INPUT_QUEUE.put(line.rstrip("\n"))
            elif line == "":
                _INPUT_QUEUE.put("!quit")
                break
        except Exception:
            _INPUT_QUEUE.put("!quit")
            break


class FastMonitor:
    def __init__(self, args=None) -> None:
        import argparse as _ap

        if args is None:
            # Fallback: derive from sys.argv for backwards-compat
            args = _ap.Namespace(
                nocam="--nocam" in sys.argv,
                nomic="--nomic" in sys.argv,
                noweb="--noweb" in sys.argv,
                web_interval=15.0,
                camera=0,
                lang="de",
            )
        print("Initialisiere Gehirn …")
        stats = db_stats()
        if stats.get("exists") and stats.get("synapses", 0) > 0:
            saved_ts = stats.get("saved_at", 0)
            saved_str = datetime.datetime.fromtimestamp(saved_ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(
                f"Vorherige Session: {stats['synapses']:,} Synapsen  "
                f"({stats.get('tick_count', 0):,} Ticks, {saved_str})"
            )
        self.brain = Brain(
            camera_index=getattr(args, "camera", 0),
            use_camera=not getattr(args, "nocam", False),
            use_microphone=not getattr(args, "nomic", False),
            use_web=not getattr(args, "noweb", False),
            web_fetch_interval=getattr(args, "web_interval", 15.0),
        )
        self.brain.start()
        # Apply language preference
        _lang = getattr(args, "lang", "de") or "de"
        self.brain._consciousness.lang._lang = _lang
        print("Fast GUI läuft …")

        self._reply_pending = False
        self._reply_time = 0.0
        self._stream_count = 0
        self._conversation: collections.deque[str] = collections.deque(maxlen=120)
        self._thoughts: collections.deque[str] = collections.deque(maxlen=500)
        self._act_hist = {
            name: collections.deque([0.0] * HISTORY, maxlen=HISTORY)
            for name in REGION_NAMES
        }
        self._wc_hist = collections.deque([0.0] * HISTORY, maxlen=HISTORY)
        self._emotion_hist = {
            name: collections.deque([0.0] * HISTORY, maxlen=HISTORY) for name in EM_DIMS
        }
        self._prev_wsum = 0.0
        self._frame_times = collections.deque(maxlen=60)
        self._last_status = ""
        self._last_controls_sync = 0.0
        self._camera_width = 640
        self._camera_height = 360
        self._camera_rgba = np.zeros(
            (self._camera_height, self._camera_width, 4), dtype=np.float32
        )
        self._head_slider_tags = {}
        self._head_value_tags = {}
        self._head_cfg_tags = {}
        self._viewport_size = (0, 0)
        self._pending_user_text = ""
        self._social_event_log: collections.deque[str] = collections.deque(maxlen=200)
        self._spatial_debug_lines: collections.deque[str] = collections.deque(maxlen=40)

    def build(self) -> None:
        dpg.create_context()
        dpg.create_viewport(
            title="Neural Consciousness Fast Monitor", width=1780, height=1040
        )
        self._build_theme()
        self._build_textures()
        self._build_ui()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        self._sync_viewport_layout(force=True)

    def _build_theme(self) -> None:
        with dpg.theme(tag="global_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, DARK)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (26, 26, 48))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (36, 52, 88))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (42, 72, 118))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (34, 51, 85))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (52, 85, 150))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (70, 110, 190))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (28, 35, 60))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (42, 62, 104))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (52, 82, 135))
                dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT)
                dpg.add_theme_color(dpg.mvThemeCol_Border, GRID)
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (24, 24, 44))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (34, 44, 78))
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (42, 62, 104))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 5)
        dpg.bind_theme("global_theme")

    def _build_textures(self) -> None:
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self._camera_width,
                self._camera_height,
                self._camera_rgba.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag="camera_texture",
            )

    def _build_ui(self) -> None:
        with dpg.window(
            tag="main_window",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
        ):
            with dpg.group(horizontal=True):
                with dpg.child_window(
                    tag="left_pane", width=1140, autosize_y=True, border=False
                ):
                    dpg.add_text("Neural Consciousness Fast Monitor", color=ACCENT)
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        with dpg.child_window(
                            tag="status_panel", width=560, height=250, border=True
                        ):
                            dpg.add_text("Status")
                            dpg.add_separator()
                            dpg.add_text("", tag="status_text", wrap=540)
                        with dpg.child_window(
                            tag="region_panel", width=560, height=250, border=True
                        ):
                            dpg.add_text("Region Activity")
                            with dpg.plot(
                                tag="activity_plot",
                                height=210,
                                width=540,
                                no_menus=True,
                                no_box_select=True,
                                no_mouse_pos=True,
                            ):
                                x_axis = dpg.add_plot_axis(
                                    dpg.mvXAxis, no_tick_labels=True
                                )
                                y_axis = dpg.add_plot_axis(
                                    dpg.mvYAxis, label="activity"
                                )
                                dpg.set_axis_limits(y_axis, 0.0, 1.0)
                                dpg.add_bar_series(
                                    list(range(len(REGION_NAMES))),
                                    [0.0] * len(REGION_NAMES),
                                    weight=0.8,
                                    parent=y_axis,
                                    tag="activity_bars",
                                )
                    with dpg.group(horizontal=True):
                        with dpg.child_window(
                            tag="emotion_panel", width=370, height=250, border=True
                        ):
                            dpg.add_text("Emotion")
                            with dpg.plot(
                                tag="emotion_plot",
                                height=210,
                                width=350,
                                no_menus=True,
                                no_box_select=True,
                                no_mouse_pos=True,
                            ):
                                x_axis = dpg.add_plot_axis(
                                    dpg.mvXAxis, no_tick_labels=True
                                )
                                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="emotion")
                                dpg.set_axis_limits(y_axis, 0.0, 1.0)
                                dpg.add_bar_series(
                                    list(range(len(EM_DIMS))),
                                    [0.0] * len(EM_DIMS),
                                    weight=0.75,
                                    parent=y_axis,
                                    tag="emotion_bars",
                                )
                            dpg.add_text("", tag="emotion_text", wrap=350)
                        with dpg.child_window(
                            tag="weight_panel", width=370, height=250, border=True
                        ):
                            dpg.add_text("Synapse Weight Delta")
                            with dpg.plot(
                                tag="weight_plot",
                                height=210,
                                width=350,
                                no_menus=True,
                                no_box_select=True,
                                no_mouse_pos=True,
                            ):
                                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="history")
                                y_axis = dpg.add_plot_axis(
                                    dpg.mvYAxis, label="Δw", tag="weight_delta_y_axis"
                                )
                                dpg.set_axis_limits("weight_delta_y_axis", -0.2, 0.2)
                                dpg.add_line_series(
                                    list(range(HISTORY)),
                                    [0.0] * HISTORY,
                                    parent="weight_delta_y_axis",
                                    tag="weight_delta_line",
                                )
                            dpg.add_text("", tag="weight_delta_text", wrap=350)
                        with dpg.child_window(
                            tag="drives_panel", width=380, height=250, border=True
                        ):
                            dpg.add_text("Drives / Concepts")
                            dpg.add_separator()
                            dpg.add_input_text(
                                tag="drives_text",
                                multiline=True,
                                readonly=True,
                                width=360,
                                height=205,
                            )
                    with dpg.tab_bar(tag="main_tabs"):
                        with dpg.tab(label="Camera", tag="camera_tab"):
                            dpg.add_image("camera_texture", tag="camera_image")
                            dpg.add_text("", tag="camera_status", wrap=1080)
                        with dpg.tab(label="Anatomy", tag="anatomy_tab"):
                            dpg.add_drawlist(
                                width=1080, height=420, tag="anatomy_drawlist"
                            )
                        with dpg.tab(label="Spatial 3D", tag="spatial_tab"):
                            dpg.add_drawlist(
                                width=1080, height=300, tag="spatial_drawlist"
                            )
                            dpg.add_separator()
                            dpg.add_input_text(
                                tag="spatial_text",
                                multiline=True,
                                readonly=True,
                                width=1060,
                                height=105,
                            )
                        with dpg.tab(label="Synapses", tag="synapses_tab"):
                            with dpg.plot(
                                tag="synapses_plot",
                                height=420,
                                width=1080,
                                no_menus=True,
                                no_box_select=True,
                                no_mouse_pos=True,
                            ):
                                x_axis = dpg.add_plot_axis(
                                    dpg.mvXAxis, label="Pre Region"
                                )
                                y_axis = dpg.add_plot_axis(
                                    dpg.mvYAxis, label="Post Region"
                                )
                                dpg.set_axis_limits(
                                    x_axis, -0.5, len(REGION_NAMES) - 0.5
                                )
                                dpg.set_axis_limits(
                                    y_axis, -0.5, len(REGION_NAMES) - 0.5
                                )
                                dpg.add_scatter_series(
                                    [], [], parent=y_axis, tag="synapse_scatter"
                                )
                            dpg.add_text("", tag="synapse_text")
                        with dpg.tab(label="Timeline", tag="timeline_tab"):
                            with dpg.plot(
                                tag="timeline_plot",
                                height=420,
                                width=1080,
                                no_menus=True,
                                no_box_select=True,
                                no_mouse_pos=True,
                            ):
                                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="history")
                                y_axis = dpg.add_plot_axis(
                                    dpg.mvYAxis, label="activity"
                                )
                                dpg.set_axis_limits(y_axis, 0.0, 0.6)
                                xs = list(range(HISTORY))
                                for region, color in KEY_REGIONS.items():
                                    dpg.add_line_series(
                                        xs,
                                        [0.0] * HISTORY,
                                        parent=y_axis,
                                        tag=f"timeline_{region}",
                                        label=LABELS.get(region, region),
                                    )
                        with dpg.tab(label="Social", tag="social_tab"):
                            with dpg.child_window(
                                tag="social_persons_panel",
                                width=1060,
                                height=200,
                                border=True,
                            ):
                                dpg.add_text("Erkannte Personen", color=ACCENT)
                                dpg.add_separator()
                                dpg.add_input_text(
                                    tag="social_persons_text",
                                    multiline=True,
                                    readonly=True,
                                    width=1040,
                                    height=155,
                                )
                            with dpg.child_window(
                                tag="social_events_panel",
                                width=1060,
                                height=195,
                                border=True,
                            ):
                                dpg.add_text("Social Events / Gespräche", color=ACCENT)
                                dpg.add_separator()
                                dpg.add_input_text(
                                    tag="social_events_text",
                                    multiline=True,
                                    readonly=True,
                                    width=1040,
                                    height=150,
                                )
                        with dpg.tab(label="Episodik", tag="episodes_tab"):
                            dpg.add_input_text(
                                tag="episodes_text",
                                multiline=True,
                                readonly=True,
                                width=1060,
                                height=415,
                            )
                        with dpg.tab(label="Skills", tag="skills_tab"):
                            dpg.add_text("", tag="skills_stats_text", wrap=1060)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                with dpg.child_window(
                                    tag="skills_cost_panel",
                                    width=520,
                                    height=175,
                                    border=True,
                                ):
                                    dpg.add_text("Learned Costs", color=ACCENT)
                                    dpg.add_input_text(
                                        tag="skills_cost_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=140,
                                    )
                                with dpg.child_window(
                                    tag="skills_success_panel",
                                    width=520,
                                    height=175,
                                    border=True,
                                ):
                                    dpg.add_text("Contextual Success", color=ACCENT)
                                    dpg.add_input_text(
                                        tag="skills_success_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=140,
                                    )
                            dpg.add_separator()
                            dpg.add_text("Ziel-Historie", color=ACCENT)
                            dpg.add_input_text(
                                tag="skills_history_text",
                                multiline=True,
                                readonly=True,
                                width=1060,
                                height=170,
                            )
                        with dpg.tab(label="Konzepte", tag="concepts_tab"):
                            dpg.add_text("", tag="concepts_header_text", wrap=1060)
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                with dpg.child_window(
                                    tag="induced_panel",
                                    width=520,
                                    height=370,
                                    border=True,
                                ):
                                    dpg.add_text("Induzierte Konzepte", color=ACCENT)
                                    dpg.add_separator()
                                    dpg.add_input_text(
                                        tag="induced_concepts_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=325,
                                    )
                                with dpg.child_window(
                                    tag="concept_conf_panel",
                                    width=520,
                                    height=370,
                                    border=True,
                                ):
                                    dpg.add_text("Erfahrungs-Konfidenz", color=ACCENT)
                                    dpg.add_separator()
                                    with dpg.plot(
                                        tag="concept_conf_plot",
                                        height=255,
                                        width=500,
                                        no_menus=True,
                                        no_box_select=True,
                                        no_mouse_pos=True,
                                    ):
                                        dpg.add_plot_axis(
                                            dpg.mvXAxis,
                                            no_tick_labels=True,
                                            tag="concept_conf_x",
                                        )
                                        dpg.add_plot_axis(
                                            dpg.mvYAxis,
                                            label="conf",
                                            tag="concept_conf_y",
                                        )
                                        dpg.set_axis_limits("concept_conf_y", 0.0, 1.0)
                                        dpg.add_bar_series(
                                            [],
                                            [],
                                            weight=0.75,
                                            parent="concept_conf_y",
                                            tag="concept_conf_bars",
                                        )
                                    dpg.add_text("", tag="concept_conf_text", wrap=500)
                        with dpg.tab(label="Kognition", tag="kognition_tab"):
                            with dpg.group(horizontal=True):
                                with dpg.child_window(
                                    tag="kontinuitat_panel",
                                    width=500,
                                    height=300,
                                    border=True,
                                ):
                                    dpg.add_text("Selbst-Kontinuität", color=ACCENT)
                                    dpg.add_separator()
                                    with dpg.plot(
                                        tag="continuity_plot",
                                        height=150,
                                        width=480,
                                        no_menus=True,
                                        no_box_select=True,
                                        no_mouse_pos=True,
                                    ):
                                        dpg.add_plot_axis(
                                            dpg.mvXAxis,
                                            no_tick_labels=True,
                                            tag="cont_x",
                                        )
                                        dpg.add_plot_axis(
                                            dpg.mvYAxis, label="[0,1]", tag="cont_y"
                                        )
                                        dpg.set_axis_limits("cont_y", 0.0, 1.0)
                                        dpg.add_bar_series(
                                            [0, 1, 2],
                                            [1.0, 1.0, 1.0],
                                            weight=0.6,
                                            parent="cont_y",
                                            tag="continuity_bars",
                                        )
                                    dpg.add_text("", tag="continuity_text", wrap=480)
                                with dpg.child_window(
                                    tag="identity_panel",
                                    width=548,
                                    height=300,
                                    border=True,
                                ):
                                    dpg.add_text("Identität & Guidelines", color=ACCENT)
                                    dpg.add_separator()
                                    dpg.add_input_text(
                                        tag="identity_text",
                                        multiline=True,
                                        readonly=True,
                                        width=528,
                                        height=255,
                                    )
                            with dpg.child_window(
                                tag="kognition_bottom_panel",
                                width=1060,
                                height=165,
                                border=True,
                            ):
                                dpg.add_text(
                                    "Weltmodell · Simulation · Überzeugungen",
                                    color=ACCENT,
                                )
                                dpg.add_separator()
                                dpg.add_input_text(
                                    tag="kognition_bottom_text",
                                    multiline=True,
                                    readonly=True,
                                    width=1040,
                                    height=120,
                                )
                        with dpg.tab(label="Systemik", tag="systemik_tab"):
                            with dpg.group(horizontal=True):
                                with dpg.child_window(
                                    tag="arc_panel", width=520, height=320, border=True
                                ):
                                    dpg.add_text(
                                        "IdentityArc — Dimensionen", color=ACCENT
                                    )
                                    dpg.add_separator()
                                    dpg.add_input_text(
                                        tag="arc_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=275,
                                    )
                                with dpg.child_window(
                                    tag="valcausal_panel",
                                    width=520,
                                    height=320,
                                    border=True,
                                ):
                                    dpg.add_text(
                                        "ValueModel · CausalGraph", color=ACCENT
                                    )
                                    dpg.add_separator()
                                    dpg.add_input_text(
                                        tag="valcausal_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=275,
                                    )
                            with dpg.group(horizontal=True):
                                with dpg.child_window(
                                    tag="narrative_panel",
                                    width=520,
                                    height=230,
                                    border=True,
                                ):
                                    dpg.add_text(
                                        "NarrativeThread — Kapitel", color=ACCENT
                                    )
                                    dpg.add_separator()
                                    dpg.add_input_text(
                                        tag="narrative_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=185,
                                    )
                                with dpg.child_window(
                                    tag="horizon_panel",
                                    width=520,
                                    height=230,
                                    border=True,
                                ):
                                    dpg.add_text(
                                        "Langzeitziele (LongHorizon)", color=ACCENT
                                    )
                                    dpg.add_separator()
                                    dpg.add_input_text(
                                        tag="horizon_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=185,
                                    )
                            with dpg.group(horizontal=True):
                                with dpg.child_window(
                                    tag="tom_panel", width=520, height=180, border=True
                                ):
                                    dpg.add_text("Theory of Mind", color=ACCENT)
                                    dpg.add_separator()
                                    dpg.add_input_text(
                                        tag="tom_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=135,
                                    )
                                with dpg.child_window(
                                    tag="attn_bq_panel",
                                    width=520,
                                    height=180,
                                    border=True,
                                ):
                                    dpg.add_text(
                                        "Aufmerksamkeit · Quarantäne", color=ACCENT
                                    )
                                    dpg.add_separator()
                                    dpg.add_input_text(
                                        tag="attn_bq_text",
                                        multiline=True,
                                        readonly=True,
                                        width=500,
                                        height=135,
                                    )
                with dpg.child_window(
                    tag="right_pane", width=600, autosize_y=True, border=False
                ):
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(
                            tag="chat_input",
                            hint="Eingabe…",
                            width=330,
                            on_enter=True,
                            callback=self._send_chat,
                        )
                        dpg.add_button(label="Senden", callback=self._send_chat)
                        dpg.add_button(label="Speichern", callback=self._save_now)
                        dpg.add_button(
                            label="Beenden", callback=self._quit_now, small=True
                        )
                    dpg.add_text(
                        "", tag="thinking_indicator", color=(100, 210, 130, 255)
                    )
                    with dpg.tab_bar():
                        with dpg.tab(label="Conversation"):
                            dpg.add_input_text(
                                tag="conversation_text",
                                multiline=True,
                                readonly=True,
                                width=570,
                                height=450,
                            )
                        with dpg.tab(label="Thoughts"):
                            dpg.add_input_text(
                                tag="thoughts_text",
                                multiline=True,
                                readonly=True,
                                width=570,
                                height=450,
                            )
                    with dpg.collapsing_header(
                        label="Head Live Control", default_open=True
                    ):
                        with dpg.group(horizontal=True):
                            dpg.add_text("Preset")
                            dpg.add_combo(
                                items=self._head_preset_names(),
                                tag="head_preset_combo",
                                default_value=self._head_preset_names()[0],
                                width=170,
                            )
                            dpg.add_button(
                                label="Apply Preset", callback=self._apply_head_preset
                            )
                        for cfg in self._head_config_items():
                            self._add_head_slider(
                                str(cfg.get("joint_name", "")),
                                self._head_label(str(cfg.get("joint_name", ""))),
                            )
                    with dpg.collapsing_header(
                        label="Head PCA9685 Config", default_open=True
                    ):
                        with dpg.table(
                            tag="head_config_table",
                            header_row=True,
                            resizable=False,
                            policy=dpg.mvTable_SizingFixedFit,
                        ):
                            for label in (
                                "Servo",
                                "CH",
                                "MinD",
                                "MaxD",
                                "MinP",
                                "MaxP",
                            ):
                                dpg.add_table_column(label=label)
                            for cfg in self._head_config_items():
                                code = str(cfg.get("code", ""))
                                with dpg.table_row():
                                    dpg.add_text(
                                        f"{code} {self._head_label(str(cfg.get('joint_name', '')))}"
                                    )
                                    fields = {}
                                    for field in (
                                        "channel",
                                        "min_deg",
                                        "max_deg",
                                        "min_pulse",
                                        "max_pulse",
                                    ):
                                        tag = f"cfg_{code}_{field}"
                                        dpg.add_input_int(tag=tag, width=80, step=1)
                                        fields[field] = tag
                                    self._head_cfg_tags[code] = fields
                        with dpg.group(horizontal=True):
                            dpg.add_button(
                                label="Apply Config", callback=self._apply_head_config
                            )
                            dpg.add_button(
                                label="Send Config", callback=self._send_head_config
                            )
                    with dpg.collapsing_header(label="Serial", default_open=True):
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(
                                tag="serial_port", hint="COM5", width=140
                            )
                            dpg.add_input_int(
                                tag="serial_baud", default_value=115200, width=120
                            )
                            dpg.add_button(
                                label="Connect", callback=self._connect_serial
                            )
                            dpg.add_button(
                                label="Disconnect", callback=self._disconnect_serial
                            )
                        dpg.add_text("", tag="serial_status", wrap=560)
        self._sync_controls(force=True)

    def _sync_viewport_layout(self, force: bool = False) -> None:
        width = max(1280, int(dpg.get_viewport_client_width() or 0))
        height = max(720, int(dpg.get_viewport_client_height() or 0))
        if not force and (width, height) == self._viewport_size:
            return
        self._viewport_size = (width, height)
        dpg.configure_item("main_window", pos=(0, 0), width=width, height=height)
        right_width = max(520, min(680, int(width * 0.34)))
        left_width = max(720, width - right_width - 36)
        pane_height = max(680, height - 24)
        dpg.configure_item("left_pane", width=left_width, height=pane_height)
        dpg.configure_item("right_pane", width=right_width, height=pane_height)
        top_w = max(320, (left_width - 18) // 2)
        tri_w = max(230, (left_width - 24) // 3)
        plot_w = max(700, left_width - 24)
        convo_w = max(420, right_width - 30)
        dpg.configure_item("status_panel", width=top_w, height=250)
        dpg.configure_item("region_panel", width=top_w, height=250)
        dpg.configure_item("activity_plot", width=top_w - 20, height=210)
        dpg.configure_item("emotion_panel", width=tri_w, height=250)
        dpg.configure_item("weight_panel", width=tri_w, height=250)
        dpg.configure_item("drives_panel", width=tri_w, height=250)
        dpg.configure_item("emotion_plot", width=tri_w - 18, height=210)
        dpg.configure_item("weight_plot", width=tri_w - 18, height=210)
        dpg.configure_item("drives_text", width=tri_w - 18, height=205)
        dpg.configure_item(
            "camera_image", width=plot_w, height=max(280, int(plot_w * 0.56))
        )
        dpg.configure_item(
            "anatomy_drawlist", width=plot_w, height=max(320, int(plot_w * 0.42))
        )
        dpg.configure_item(
            "spatial_drawlist", width=plot_w, height=max(260, int(plot_w * 0.30))
        )
        dpg.configure_item(
            "spatial_text", width=plot_w, height=max(90, int(plot_w * 0.10))
        )
        dpg.configure_item(
            "synapses_plot", width=plot_w, height=max(320, int(plot_w * 0.42))
        )
        dpg.configure_item(
            "timeline_plot", width=plot_w, height=max(320, int(plot_w * 0.42))
        )
        dpg.configure_item("camera_status", wrap=plot_w)
        convo_h = max(420, pane_height - 370)
        dpg.configure_item("conversation_text", width=convo_w, height=convo_h)
        dpg.configure_item("thoughts_text", width=convo_w, height=convo_h)
        dpg.configure_item("chat_input", width=max(180, convo_w - 210))
        # ── New-panel sizing ────────────────────────────────────────────
        half_w = max(350, (plot_w - 12) // 2)
        soc_top_h = max(130, min(220, pane_height // 4))
        soc_bot_h = max(130, pane_height - soc_top_h - 160)
        dpg.configure_item("social_persons_panel", width=plot_w, height=soc_top_h)
        dpg.configure_item("social_events_panel", width=plot_w, height=soc_bot_h)
        dpg.configure_item(
            "social_persons_text", width=plot_w - 20, height=soc_top_h - 45
        )
        dpg.configure_item(
            "social_events_text", width=plot_w - 20, height=soc_bot_h - 45
        )
        dpg.configure_item(
            "episodes_text", width=plot_w, height=max(320, pane_height - 150)
        )
        dpg.configure_item("skills_stats_text", wrap=plot_w)
        skl_tbl_h = max(140, (pane_height - 270) // 3)
        dpg.configure_item("skills_cost_panel", width=half_w, height=skl_tbl_h + 35)
        dpg.configure_item("skills_success_panel", width=half_w, height=skl_tbl_h + 35)
        dpg.configure_item("skills_cost_text", width=half_w - 20, height=skl_tbl_h)
        dpg.configure_item("skills_success_text", width=half_w - 20, height=skl_tbl_h)
        dpg.configure_item(
            "skills_history_text",
            width=plot_w,
            height=max(120, pane_height - skl_tbl_h * 2 - 270),
        )
        dpg.configure_item("concepts_header_text", wrap=plot_w)
        cpt_h = max(300, pane_height - 200)
        dpg.configure_item("induced_panel", width=half_w, height=cpt_h)
        dpg.configure_item("concept_conf_panel", width=half_w, height=cpt_h)
        dpg.configure_item(
            "induced_concepts_text", width=half_w - 20, height=cpt_h - 55
        )
        dpg.configure_item(
            "concept_conf_plot", width=half_w - 20, height=max(180, cpt_h - 120)
        )
        dpg.configure_item("concept_conf_text", wrap=half_w - 20)
        # ── Kognition tab sizing ─────────────────────────────────────────
        cont_panel_w = max(320, (plot_w - 12) // 2)
        id_panel_w = max(320, plot_w - cont_panel_w - 12)
        kgn_top_h = max(200, pane_height - 260)
        dpg.configure_item("kontinuitat_panel", width=cont_panel_w, height=kgn_top_h)
        dpg.configure_item("identity_panel", width=id_panel_w, height=kgn_top_h)
        dpg.configure_item(
            "continuity_plot", width=cont_panel_w - 20, height=max(120, kgn_top_h - 110)
        )
        dpg.configure_item(
            "identity_text", width=id_panel_w - 20, height=kgn_top_h - 45
        )
        kgn_bot_h = max(120, pane_height - kgn_top_h - 120)
        dpg.configure_item("kognition_bottom_panel", width=plot_w, height=kgn_bot_h)
        dpg.configure_item(
            "kognition_bottom_text", width=plot_w - 20, height=max(80, kgn_bot_h - 45)
        )
        # ── Systemik tab sizing ───────────────────────────────────────────
        sys_top_h = max(250, (pane_height - 80) // 3 + 50)
        sys_mid_h = max(180, (pane_height - 80) // 3)
        sys_bot_h = max(150, pane_height - sys_top_h - sys_mid_h - 80)
        dpg.configure_item("arc_panel", width=half_w, height=sys_top_h)
        dpg.configure_item("valcausal_panel", width=half_w, height=sys_top_h)
        dpg.configure_item("arc_text", width=half_w - 20, height=sys_top_h - 45)
        dpg.configure_item("valcausal_text", width=half_w - 20, height=sys_top_h - 45)
        dpg.configure_item("narrative_panel", width=half_w, height=sys_mid_h)
        dpg.configure_item("horizon_panel", width=half_w, height=sys_mid_h)
        dpg.configure_item("narrative_text", width=half_w - 20, height=sys_mid_h - 45)
        dpg.configure_item("horizon_text", width=half_w - 20, height=sys_mid_h - 45)
        dpg.configure_item("tom_panel", width=half_w, height=sys_bot_h)
        dpg.configure_item("attn_bq_panel", width=half_w, height=sys_bot_h)
        dpg.configure_item("tom_text", width=half_w - 20, height=sys_bot_h - 45)
        dpg.configure_item("attn_bq_text", width=half_w - 20, height=sys_bot_h - 45)

    def _add_head_slider(self, joint_name: str, label: str) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text(label)
            slider_tag = f"slider_{joint_name}"
            value_tag = f"value_{joint_name}"
            dpg.add_slider_int(
                tag=slider_tag,
                min_value=0,
                max_value=180,
                width=360,
                callback=self._schedule_head_pose_send,
            )
            dpg.add_text("0", tag=value_tag)
            self._head_slider_tags[joint_name] = slider_tag
            self._head_value_tags[joint_name] = value_tag

    def _head_config_items(self) -> List[dict]:
        state = self.brain.robot_controller_state
        head_config = state.get("head_config", {}) if isinstance(state, dict) else {}
        items = [dict(cfg) for cfg in head_config.values() if isinstance(cfg, dict)]
        items.sort(
            key=lambda cfg: (int(cfg.get("channel", 999)), str(cfg.get("code", "")))
        )
        return items

    def _head_label(self, joint_name: str) -> str:
        return HEAD_LABELS.get(joint_name, joint_name.replace("_", " ").title())

    def _head_preset_names(self) -> List[str]:
        state = self.brain.robot_controller_state
        presets = list((state.get("head_presets", {}) or {}).keys())
        return presets or ["Center"]

    def _connect_serial(self, sender=None, app_data=None, user_data=None) -> None:
        del sender, app_data, user_data
        port = dpg.get_value("serial_port").strip()
        baud = int(dpg.get_value("serial_baud") or 115200)
        self.brain.connect_robot_serial(port, baud)
        self._sync_controls(force=True)

    def _disconnect_serial(self, sender=None, app_data=None, user_data=None) -> None:
        del sender, app_data, user_data
        self.brain.disconnect_robot_serial()
        self._sync_controls(force=True)

    def _apply_head_config(self, sender=None, app_data=None, user_data=None) -> None:
        del sender, app_data, user_data
        for code, fields in self._head_cfg_tags.items():
            self.brain.update_head_servo_config(
                code,
                channel=dpg.get_value(fields["channel"]),
                min_deg=dpg.get_value(fields["min_deg"]),
                max_deg=dpg.get_value(fields["max_deg"]),
                min_pulse=dpg.get_value(fields["min_pulse"]),
                max_pulse=dpg.get_value(fields["max_pulse"]),
            )
        self.brain.send_head_servo_config()
        self._sync_controls(force=True)

    def _send_head_config(self, sender=None, app_data=None, user_data=None) -> None:
        del sender, app_data, user_data
        self.brain.send_head_servo_config()
        self._sync_controls(force=True)

    def _apply_head_preset(self, sender=None, app_data=None, user_data=None) -> None:
        del sender, app_data, user_data
        self.brain.apply_head_preset(dpg.get_value("head_preset_combo"))
        self._sync_controls(force=True)

    def _schedule_head_pose_send(self, sender: str, app_data: int) -> None:
        del sender
        del app_data
        self._frame_times.append(time.perf_counter())

    def _send_current_head_pose(self) -> None:
        self.brain.set_head_targets(
            {
                joint_name: float(dpg.get_value(slider_tag))
                for joint_name, slider_tag in self._head_slider_tags.items()
            }
        )

    def _send_chat(self, sender=None, app_data=None, user_data=None) -> None:
        del sender, app_data, user_data
        text = dpg.get_value("chat_input").strip()
        if not text:
            return
        dpg.set_value("chat_input", "")
        self._handle_input(text)

    def _save_now(self, sender=None, app_data=None, user_data=None) -> None:
        del sender, app_data, user_data
        dpg.set_value("thinking_indicator", "Speichert…")

        def _do_save() -> None:
            try:
                from persistence import save_brain

                n = save_brain(self.brain)
                self._thoughts.appendleft(f"[gespeichert] {n:,} Synapsen")
                dpg.set_value("thinking_indicator", f"Gespeichert ({n:,} Syn)")
            except Exception as exc:
                self._thoughts.appendleft(f"[speicher-fehler] {exc}")
                dpg.set_value("thinking_indicator", f"Fehler: {exc}")

        threading.Thread(target=_do_save, daemon=True).start()

    def _quit_now(self, sender=None, app_data=None, user_data=None) -> None:
        del sender, app_data, user_data
        dpg.set_value("thinking_indicator", "Speichert und beendet…")

        def _do_quit() -> None:
            self.shutdown()

        threading.Thread(target=_do_quit, daemon=True).start()

    def _handle_input(self, line: str) -> None:
        low = line.lower().strip()
        if not low:
            return
        if low in ("!quit", "!beenden"):
            self.shutdown()
            return
        if low in ("!speichern", "!save"):
            try:
                from persistence import save_brain

                n = save_brain(self.brain)
                self._thoughts.appendleft(f"saved {n:,} synapses")
            except Exception as exc:
                self._thoughts.appendleft(f"save error: {exc}")
            return
        self.brain.request_reply(line)
        self._conversation.append(f"Du: {line}")
        self._reply_pending = True
        self._pending_user_text = line
        self._reply_time = time.perf_counter()
        dpg.set_value("thinking_indicator", "⟳ Denkt...")

    def _token_overlap(self, a: str, b: str) -> float:
        ta = {tok for tok in a.lower().split() if len(tok) > 2}
        tb = {tok for tok in b.lower().split() if len(tok) > 2}
        return len(ta & tb) / max(len(ta), 1)

    def _append_ai_message(self, text: str) -> None:
        clean = text.replace("[COMM] ", "").strip()
        if not clean:
            return
        recent_ai = [
            line[4:] for line in reversed(self._conversation) if line.startswith("AI: ")
        ]
        if any(self._token_overlap(clean, prev) > 0.70 for prev in recent_ai[:6]):
            return
        self._conversation.append(f"AI: {clean}")

    def _speak_fallback_reply(self, text: str) -> None:
        clean = text.replace("[COMM] ", "").strip()
        if not clean:
            return
        dm = getattr(self.brain, "_dialogue_manager", None)
        so = getattr(self.brain, "_speech_output", None)
        cs = getattr(self.brain, "_consciousness", None)
        sm = getattr(self.brain, "_social_manager", None)
        if dm is None or so is None or cs is None:
            return
        addressee = sm.primary_interlocutor() if sm is not None else ""
        tom_strategy = {}
        if addressee and hasattr(cs, "theory_of_mind"):
            tom_strategy = cs.theory_of_mind.recommend_strategy(str(addressee))
        try:
            plan = dm.build_utterance(
                clean,
                addressee or "user",
                cs,
                tick=self.brain.tick_count,
                tom_strategy=tom_strategy,
            )
            if so.speak_utterance(plan):
                dm.mark_output_delivered(self.brain.tick_count)
        except Exception:
            pass

    def _reply_relevance(self, text: str, reference: str) -> float:
        ref_tokens = {tok for tok in reference.lower().split() if len(tok) > 2}
        msg_tokens = {tok for tok in text.lower().split() if len(tok) > 2}
        if not ref_tokens:
            return 1.0
        return len(ref_tokens & msg_tokens) / max(len(ref_tokens), 1)

    def _sync_controls(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_controls_sync < 0.25:
            return
        self._last_controls_sync = now
        state = self.brain.robot_controller_state
        targets = state.get("target_joints", {}) if isinstance(state, dict) else {}
        for joint_name, slider_tag in self._head_slider_tags.items():
            value = int(
                round(float(targets.get(joint_name, dpg.get_value(slider_tag) or 0)))
            )
            if dpg.get_value(slider_tag) != value:
                dpg.set_value(slider_tag, value)
            dpg.set_value(self._head_value_tags[joint_name], str(value))
        head_config = state.get("head_config", {}) if isinstance(state, dict) else {}
        for code, fields in self._head_cfg_tags.items():
            cfg = head_config.get(code, {})
            for field, tag in fields.items():
                if cfg:
                    dpg.set_value(tag, int(cfg.get(field, dpg.get_value(tag) or 0)))
        serial = state.get("serial", {}) if isinstance(state, dict) else {}
        if serial.get("port") and not dpg.get_value("serial_port"):
            dpg.set_value("serial_port", str(serial.get("port", "")))
        dpg.set_value("serial_baud", int(serial.get("baudrate", 115200) or 115200))
        ports = ", ".join(list(map(str, serial.get("ports", [])))[:6]) or "none"
        dpg.set_value(
            "serial_status",
            f"connected={int(bool(serial.get('connected', False)))}  port={serial.get('port', '-') or '-'}  baud={int(serial.get('baudrate', 115200) or 115200)}\n"
            f"ports={ports}\n"
            f"last_frame={str(serial.get('last_frame', '') or '')[:120]}\n"
            f"error={str(serial.get('last_error', '') or 'none')[:120]}",
        )

    def _update_status(self) -> None:
        acts = self.brain.region_activity.copy()
        em = self.brain.emotion_state
        cs = self.brain._consciousness
        state = self.brain.robot_controller_state
        serial = state.get("serial", {}) if isinstance(state, dict) else {}
        telem = state.get("telemetry", {}) if isinstance(state, dict) else {}
        fps = 0.0
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                fps = (len(self._frame_times) - 1) / span
        status = (
            f"tick={self.brain.tick_count:,}  syn={len(self.brain._inter_synapses):,}  fps={fps:.1f}\n"
            f"emotion={em.describe()}  goal={self.brain.consciousness_state.goal}\n"
            f"robot={self.brain.last_robot_command[:140]}\n"
            f"focus={telem.get('selected_target_label', 'none')} @ ({float(telem.get('selected_target_center_x', 0.5)):.2f}, {float(telem.get('selected_target_center_y', 0.5)):.2f})\n"
            f"serial_connected={int(bool(serial.get('connected', False)))}  controller={self.brain.robot_controller_summary[:220]}"
        )
        if status != self._last_status:
            dpg.set_value("status_text", status)
            self._last_status = status
        vals = [acts.get(name, 0.0) for name in REGION_NAMES]
        dpg.set_value("activity_bars", [list(range(len(REGION_NAMES))), vals])
        em_vals = [max(0.0, min(1.0, float(getattr(em, dim, 0.0)))) for dim in EM_DIMS]
        dpg.set_value("emotion_bars", [list(range(len(EM_DIMS))), em_vals])
        dom_idx = int(np.argmax(em_vals)) if em_vals else 0
        dpg.set_value(
            "emotion_text",
            f"dominant={EM_DIMS[dom_idx]} ({em_vals[dom_idx]:.2f})  rpe={em.reward_prediction_error:+.4f}",
        )
        for region in KEY_REGIONS:
            self._act_hist[region].append(acts.get(region, 0.0))
            dpg.set_value(
                f"timeline_{region}",
                [list(range(HISTORY)), list(self._act_hist[region])],
            )
        wc = self._sample_weight_delta()
        self._wc_hist.append(wc)
        wc_vals = list(self._wc_hist)
        hi = max(0.05, max(abs(v) for v in wc_vals))
        dpg.set_axis_limits("weight_delta_y_axis", -hi * 1.3, hi * 1.3)
        dpg.set_value("weight_delta_line", [list(range(HISTORY)), wc_vals])
        pos = sum(1 for v in wc_vals if v > 1e-4)
        neg = sum(1 for v in wc_vals if v < -1e-4)
        dpg.set_value("weight_delta_text", f"ltp={pos}  ltd={neg}  latest={wc:+.5f}")
        self._update_drives_panel(cs, em, state)

    def _update_drives_panel(self, cs, em, state: dict) -> None:
        drives = cs.drives
        st = self.brain.consciousness_state
        lines = [
            f"info_hunger   {drives.information_hunger:.2f}",
            f"coherence     {drives.coherence_need:.2f}",
            f"expression    {drives.expression_pressure:.2f}",
            f"rest_need     {drives.rest_need:.2f}",
            f"rpe           {em.reward_prediction_error:+.4f}",
        ]
        # ── Semantic memory stats ──────────────────────────────────────────
        hipp = self.brain.hippocampus
        n_assoc = len(getattr(hipp, "_semantic_memory", {}))
        n_ep = len(getattr(hipp, "_semantic_episodes", []))
        n_beliefs = getattr(cs, "belief_store", None)
        n_beliefs = n_beliefs.size if n_beliefs is not None else 0
        lang_pref = getattr(getattr(cs, "lang", None), "_lang", "?")
        if n_assoc or n_ep or n_beliefs:
            lines.append("")
            lines.append(
                f"semantic mem  {n_assoc} assoc  {n_ep} ep  |  beliefs {n_beliefs}  lang={lang_pref}"
            )
        # ── Amygdala appraisal ─────────────────────────────────────────────
        apr = getattr(self.brain.amygdala, "_last_appraisal", {})
        if apr:
            reward = apr.get("reward", 0.0)
            threat = apr.get("threat", 0.0)
            novelty = apr.get("novelty", 0.0)
            social = apr.get("social", 0.0)
            lines.append(
                f"appraisal  rew={reward:.2f} thr={threat:.2f} nov={novelty:.2f} soc={social:.2f}"
            )
        # ── Workspace concepts ─────────────────────────────────────────────
        concepts = list(cs._concepts)[-5:]
        if concepts:
            lines.append("")
            lines.append("concepts")
            for concept in reversed(concepts):
                neighbors = cs.concept_graph.neighbors(concept, 3)
                nb = ", ".join(f"{name}({weight:.1f})" for name, weight in neighbors)
                lines.append(f"- {concept[:22]} -> {nb[:42]}")
        if st.task_summary:
            lines.append("")
            lines.append(f"task {st.task_summary[:88]}")
        if st.embodiment_summary:
            lines.append(f"body {st.embodiment_summary[:88]}")
        if st.robot_summary:
            lines.append(f"robot {st.robot_summary[:88]}")
        # ── Concept inductor / experience ──────────────────────────────────
        ci = getattr(cs, "concept_inductor", None)
        ea = getattr(self.brain._emotion_engine, "experience", None)
        if ci or ea:
            lines.append("")
            if ci:
                ci_st = ci.stats()
                lines.append(
                    f"inductor  patterns={ci_st['patterns_tracked']}  minted={ci_st['concepts_minted']}"
                )
            if ea:
                lines.append(f"experience  known={ea.known_concepts()} concepts")
        serial = state.get("serial", {}) if isinstance(state, dict) else {}
        if serial:
            lines.append(
                f"serial connected={int(bool(serial.get('connected', False)))} port={serial.get('port', '-') or '-'}"
            )
        # ── Continuity Monitor ─────────────────────────────────────────────
        cont = getattr(cs, "continuity", None)
        if cont:
            lines.append("")
            lines.append(f"continuity overall={cont.overall:.2f}  ({cont.describe()})")
        # ── Sensor Health ──────────────────────────────────────────────────
        sh = getattr(cs.self_model, "sensor_health", None)
        if sh:
            lines.append("")
            lines.append(f"sensors  {sh.describe()}")
            deg = sh.degraded_names
            if deg:
                lines.append(f"  [DEGRADED: {', '.join(deg)}]")
        dpg.set_value("drives_text", "\n".join(lines))

    # ─────────────────────────────────────────────────────────
    # Dedicated tab update methods (new panels)
    # ─────────────────────────────────────────────────────────

    def _update_social(self) -> None:
        """Refresh Social tab: person models and event log."""
        sm = self.brain._social_manager
        for evt in getattr(sm, "_social_events", []):
            self._social_event_log.appendleft(f"[t={self.brain.tick_count}] {evt}")
        pm_dict = getattr(sm, "_person_models", {})
        conv_dict = getattr(sm, "_conversations", {})
        lines = []
        for pid, pm in list(pm_dict.items())[:12]:
            conv = conv_dict.get(pid)
            eng = f"{conv.engagement:.2f}" if conv else "—"
            rap = f"{conv.rapport:.2f}" if conv else "—"
            greeted = "ja" if (conv and conv.greeted) else "nein"
            interests = ", ".join(pm.inferred_interests[:6]) or "—"
            prefs = (
                ", ".join(
                    f"{k}:{v:.2f}"
                    for k, v in sorted(pm.preferences.items(), key=lambda x: -x[1])[:5]
                )
                or "—"
            )
            lines.append(
                f"── Person {pm.person_id} {'─' * 50}\n"
                f"  trust={pm.trust:.2f}  famil={pm.familiarity:.2f}  "
                f"enc={pm.total_encounters}  val={pm.avg_valence:+.2f}  "
                f"eng={eng}  rap={rap}  begrüßt={greeted}\n"
                f"  Interessen : {interests}\n"
                f"  Präferenzen: {prefs}"
            )
        dpg.set_value(
            "social_persons_text",
            "\n\n".join(lines) if lines else "Keine Personen erkannt",
        )
        dpg.set_value(
            "social_events_text",
            "\n".join(list(self._social_event_log)[:40]) or "—",
        )

    def _update_episodes(self) -> None:
        """Refresh Episodik tab: recent episodic events with multi-person attribution."""
        cs = self.brain._consciousness
        events = cs.episodic.recent(40)
        lines = []
        for evt in reversed(events):
            pids_attr = getattr(evt, "social_person_ids", ())
            if pids_attr:
                pids = f" P{list(pids_attr)}"
            elif getattr(evt, "social_person_id", None) is not None:
                pids = f" P{evt.social_person_id}"
            else:
                pids = ""
            em = f" [{evt.emotion_snapshot[:14]}]" if evt.emotion_snapshot else ""
            lines.append(
                f"[t={evt.tick:<7}|{evt.kind:<10}{pids}{em}]  {evt.content[:80]}"
            )
        dpg.set_value(
            "episodes_text",
            "\n".join(lines) if lines else "Keine Episoden aufgezeichnet",
        )

    def _update_skills(self) -> None:
        """Refresh Skills tab: learned costs, contextual success, goal history."""
        te = self.brain._task_executive
        sl = self.brain._skill_library
        plans = getattr(te, "_plans_generated", 0)
        recipes = getattr(te, "_recipes_used", 0)
        history = getattr(te, "_history", [])
        active = getattr(te, "_active_goal", None)
        if active:
            active_str = (
                f"{active.intent} [{active.status.value}]"
                f" step={active.current_step}/{len(active.steps)}"
            )
        else:
            active_str = "—"
        dpg.set_value(
            "skills_stats_text",
            f"plans_generated={plans}  recipes_used={recipes}  "
            f"goal_history={len(history)}  aktiv: {active_str}",
        )
        costs = getattr(sl, "_learned_cost", {})
        dpg.set_value(
            "skills_cost_text",
            "\n".join(
                f"{name[:26]:<26}  {cost:.5f}"
                for name, cost in sorted(
                    costs.items(), key=lambda x: x[1], reverse=True
                )[:22]
            )
            or "—",
        )
        ctxs = getattr(sl, "_contextual_success", {})
        dpg.set_value(
            "skills_success_text",
            "\n".join(
                f"{str(k[0])[:18]}/{str(k[1])[:12]}  {v:.3f}"
                for k, v in sorted(ctxs.items(), key=lambda x: x[1], reverse=True)[:22]
            )
            or "—",
        )
        hist_lines = [
            f"[{g.status.value:<9}] {g.intent:<20} {g.result_msg[:42]}"
            for g in reversed(history[-25:])
        ]
        dpg.set_value(
            "skills_history_text",
            "\n".join(hist_lines) if hist_lines else "—",
        )

    def _update_concepts(self) -> None:
        """Refresh Konzepte tab: induced concepts + per-concept confidence bars."""
        cs = self.brain._consciousness
        ea = self.brain._emotion_engine.experience
        ci = getattr(cs, "concept_inductor", None)
        known = ea.known_concepts()
        minted = len(getattr(ci, "_minted", {})) if ci else 0
        dpg.set_value(
            "concepts_header_text",
            f"ExperienceAppraisal: {known} bekannte Konzepte  |  "
            f"ConceptInductor: {minted} induziert",
        )
        if ci:
            ci_stats = ci.stats()
            ind_lines = [
                f"Patterns getrackt: {ci_stats['patterns_tracked']}"
                f"  Konzepte geminted: {ci_stats['concepts_minted']}",
                "",
            ]
            for pattern, label in list(getattr(ci, "_minted", {}).items())[:35]:
                preds = ", ".join(sorted(pattern)[:5])
                ind_lines.append(f"  {label[:42]:<42}  [{preds[:38]}]")
            dpg.set_value("induced_concepts_text", "\n".join(ind_lines))
        else:
            dpg.set_value("induced_concepts_text", "ConceptInductor nicht verfügbar")
        all_concepts = sorted(
            getattr(ea, "_associations", {}).keys(),
            key=lambda c: ea.concept_confidence(c),
            reverse=True,
        )[:16]
        if all_concepts:
            confs = [ea.concept_confidence(c) for c in all_concepts]
            obs = [
                getattr(ea, "_observation_count", {}).get(c, 0) for c in all_concepts
            ]
            dpg.set_value("concept_conf_bars", [list(range(len(all_concepts))), confs])
            conf_text = "\n".join(
                f"{c[:14]:<14}  obs={obs[i]:<5}  conf={confs[i]:.3f}"
                for i, c in enumerate(all_concepts[:10])
            )
            dpg.set_value("concept_conf_text", conf_text)
        else:
            dpg.set_value("concept_conf_text", "—")

    def _update_kognition(self) -> None:
        """Refresh Kognition tab: continuity, identity, world model, simulation, beliefs."""
        cs = self.brain._consciousness
        cont = cs.continuity
        auto = cs.autobiography
        wm = cs.world_model
        sp = cs.sandbox_planner
        bs = cs.belief_store

        # ── Continuity bar chart ──────────────────────────────────────────
        dpg.set_value(
            "continuity_bars",
            [
                [0, 1, 2],
                [cont.memory_coherence, cont.agency_stability, cont.value_stability],
            ],
        )
        fragile = cont.fragile_segments
        fragile_str = "FRAGIL: " + ", ".join(fragile) if fragile else "stabil"
        recent_alarms = list(cont._alarms)[-3:]
        dpg.set_value(
            "continuity_text",
            f"memory={cont.memory_coherence:.3f}  agency={cont.agency_stability:.3f}  values={cont.value_stability:.3f}\n"
            f"overall={cont.overall:.3f}   {fragile_str}\n"
            f"Alarme: {'; '.join(recent_alarms) or '—'}",
        )

        # ── Identity & Guidelines ─────────────────────────────────────────
        id_cons = auto.identity_consistency
        guidelines = auto.guidelines
        summary = (auto.identity_summary or "—")[:140]
        gl_lines = [
            f"  [{gl.source}] {gl.text} (stärke={gl.strength:.2f})"
            for gl in guidelines[:12]
        ]
        id_str = (
            f"Konsistenz: {id_cons:.3f}\n"
            f"Summary: {summary}\n\n"
            f"Guidelines ({len(guidelines)}):\n"
            + ("\n".join(gl_lines) if gl_lines else "  —")
        )
        dpg.set_value("identity_text", id_str)

        # ── Weltmodell + Simulation + Beliefs (bottom) ────────────────────
        wm_lines = wm.summarise(3)
        wm_str = "\n".join(wm_lines) if wm_lines else "—"

        sim_recent = list(sp._recent)[-5:]
        sim_str = "\n".join(sim_recent) if sim_recent else "—"

        quarantined = bs.quarantined()
        if quarantined:
            q_parts = [f"{s}/{r}/{o}" for s, r, o, _ in quarantined[:6]]
            q_str = f"Quarantäne ({len(quarantined)}): " + ", ".join(q_parts)
        else:
            q_str = "keine Quarantäne"

        bottom_str = (
            f"── Weltmodell ───────────────────────────────────\n{wm_str}\n\n"
            f"── Simulation (letzte Pfade) ────────────────────\n{sim_str}\n\n"
            f"── Überzeugungen ────────────────────────────────\n"
            f"aktiv={bs.size}   {q_str}"
        )
        dpg.set_value("kognition_bottom_text", bottom_str)

    def _update_systemik(self) -> None:
        """Refresh Systemik tab: IdentityArc, ValueModel, CausalGraph,
        NarrativeThread, GoalStack, TheoryOfMind, AttentionCtrl, BeliefQuarantine."""
        cs = self.brain._consciousness

        # ── IdentityArc ───────────────────────────────────────────────────
        arc = cs.identity_arc
        arc_lines = [
            f"Konsistenz: {arc.consistency_score():.3f}   "
            f"Charakter: {arc.character_summary()[:55]}",
            f"Fehler-Stil: {arc.error_handling_style()}   "
            f"Meta-Ziele: {len(arc._meta_goals)}",
        ]
        recent_meta = list(arc._meta_goals)[-2:]
        for mg in recent_meta:
            arc_lines.append(f"  {mg[:70]}")
        arc_lines.append("")
        arc_lines.append(f"  {'Dimension':<20} {'Ist':>5}  {'Ziel':>5}  {'Gap':>6}  OK")
        arc_lines.append("  " + "─" * 48)
        for name, dim in arc.dimensions.items():
            ok = "✓" if dim.aligned() else "○"
            arc_lines.append(
                f"  {name:<20} {dim.current:>5.2f}  {dim.target:>5.2f}  "
                f"{dim.gap():>+5.2f}  {ok}"
            )
        comm_mods = arc.communication_style_modifiers()
        if comm_mods:
            arc_lines.append("")
            mods_str = "  ".join(f"{k}={v:.2f}" for k, v in comm_mods.items())
            arc_lines.append(f"Komm-Modifikatoren: {mods_str}")
        dpg.set_value("arc_text", "\n".join(arc_lines))

        # ── ValueModel + CausalGraph ──────────────────────────────────────
        vm = cs.value_model
        cg = cs.causal_graph
        vc_lines = [
            f"ValueModel:  td_err={vm.mean_td_error:.4f}  "
            f"trend={vm.value_trend:+.4f}  "
            f"improving={'ja' if vm.improving else 'nein'}",
            "",
            "── Top-States (Signatur → gelernter Wert) ──",
        ]
        for sig, val in vm.top_states(5):
            vc_lines.append(f"  {sig[:36]}  v={val:.3f}")
        vc_lines.append("")
        vc_lines.append("── CausalGraph: Zuverlässigkeit pro Ziel ──")
        for goal in ("explore", "consolidate", "respond", "rest"):
            rel = cg.action_reliability(goal)
            vc_lines.append(f"  {goal:<14}  rel={rel:.3f}")
        vc_lines.append("")
        for line in cg.summarise(4):
            vc_lines.append(f"  {line}")
        dpg.set_value("valcausal_text", "\n".join(vc_lines))

        # ── NarrativeThread ───────────────────────────────────────────────
        nt = cs.narrative_thread
        chapters = nt.recent_chapters(4)
        tp_count = nt.turning_point_count()
        nt_lines = [
            f"Kapitel={len(nt._chapters)}  Wendepunkte={tp_count}",
            "",
            "── Geschichte ──",
        ]
        story = nt.story_so_far(3)
        for line in story.splitlines()[:8]:
            nt_lines.append(f"  {line[:72]}")
        nt_lines.append("")
        nt_lines.append("── Letzte Kapitel ──")
        for ch in chapters:
            tp_marker = " [WENDEPUNKT]" if ch.is_turning_point else ""
            nt_lines.append(f"  [{ch.chapter_type}]{tp_marker} {ch.title[:44]}")
            if ch.lessons:
                nt_lines.append(f"    → {'; '.join(ch.lessons[:2])}")
        dpg.set_value("narrative_text", "\n".join(nt_lines))

        # ── GoalStack (LongHorizon) ───────────────────────────────────────
        lh = cs.long_horizon
        active = lh.active_goals()
        all_goals = list(lh._goals.values())
        pat_counts = getattr(lh, "_pattern_counts", {})
        lh_lines = [
            f"Aktiv={len(active)}  Gesamt={len(all_goals)}  " f"Max={lh.MAX_GOALS}",
        ]
        if pat_counts:
            pat_parts = "  ".join(
                f"{k}={v}"
                for k, v in sorted(pat_counts.items(), key=lambda x: -x[1])[:5]
            )
            lh_lines.append(f"Muster: {pat_parts}")
        lh_lines.append("")
        for gc in active[:8]:
            ms_done = sum(1 for m in gc.milestones if m.completed)
            lh_lines.append(
                f"  [{gc.goal_id}] p={gc.priority}  commit={gc.commitment:.2f}  "
                f"prog={gc.progress:.0%}  ms={ms_done}/{len(gc.milestones)}"
            )
            lh_lines.append(f"    {gc.description[:62]}")
        paused = sum(1 for g in all_goals if g.status == "paused")
        completed = sum(1 for g in all_goals if g.status == "completed")
        abandoned = sum(1 for g in all_goals if g.status == "abandoned")
        lh_lines.append(
            f"\npausiert={paused}  abgeschlossen={completed}  aufgegeben={abandoned}"
        )
        dpg.set_value("horizon_text", "\n".join(lh_lines))

        # ── Theory of Mind ────────────────────────────────────────────────
        tom = cs.theory_of_mind
        models = getattr(tom, "_models", {})
        tom_lines = [f"Geistige Modelle: {len(models)}"]
        if not models:
            tom_lines.append("  (keine Personen beobachtet)")
        else:
            tom_lines.append("")
            for pid, mm in list(models.items())[:6]:
                tom_lines.append(
                    f"  P{pid}  conf={mm.model_confidence:.2f}  "
                    f"emo={mm.inferred_emotion}  attn={mm.inferred_attention}  "
                    f"obs={mm.observation_count}"
                )
                tom_lines.append(
                    f"    style={mm.communication_style}  "
                    f"pattern={mm.response_pattern}  "
                    f"erwartet: {mm.expects_what()}"
                )
                if mm.knowledge_estimate:
                    top_k = sorted(mm.knowledge_estimate.items(), key=lambda x: -x[1])[
                        :3
                    ]
                    tom_lines.append(
                        "    Wissen: " + "  ".join(f"{k}:{v:.2f}" for k, v in top_k)
                    )
        dpg.set_value("tom_text", "\n".join(tom_lines))

        # ── AttentionController + BeliefQuarantine ────────────────────────
        ac = cs.attention_ctrl
        bq = cs.belief_quarantine
        ab_lines = ["── Aufmerksamkeit — Top-Prioritäten ──"]
        tick_now = self.brain.tick_count
        for p in ac.top_priorities(6):
            ab_lines.append(
                f"  [{p.source[:10]:<10}] {p.target[:24]:<24}  "
                f"w={p.weight:.3f}  age={tick_now - p.created_tick}"
            )
        ab_lines.append("")
        ab_lines.append(
            f"── Quarantäne — {bq.quarantined_count()} Überzeugungen ausstehend ──"
        )
        ab_lines.append(f"  {bq.summary()}")
        conflicts = bq.unresolved_conflicts()
        if conflicts:
            ab_lines.append(f"  Konflikte ({len(conflicts)}):")
            for cf in conflicts[:4]:
                ab_lines.append(f"    {cf.topic_a} ↔ {cf.topic_b}")
        dpg.set_value("attn_bq_text", "\n".join(ab_lines))

    def _sample_weight_delta(self) -> float:
        sample = self.brain._inter_synapses[::40]
        if not sample:
            return 0.0
        total = sum(s.weight for s in sample)
        delta = (total - self._prev_wsum) / max(len(sample), 1)
        self._prev_wsum = total
        return delta

    def _update_conversation(self) -> None:
        # Check for completed direct-reply results first (respond_to() in tick thread)
        while self.brain._reply_results:
            response = self.brain._reply_results.pop()
            if response:
                self._append_ai_message(response)
            self._reply_pending = False
            self._pending_user_text = ""
            dpg.set_value("thinking_indicator", "")
        if self.brain.wants_to_communicate:
            outbound = self.brain.get_outbound_messages()
            if self._reply_pending and self._pending_user_text:
                best_msg = max(
                    outbound,
                    key=lambda msg: self._reply_relevance(msg, self._pending_user_text),
                    default="",
                )
                if (
                    best_msg
                    and self._reply_relevance(best_msg, self._pending_user_text) >= 0.10
                ):
                    self._append_ai_message(best_msg)
                else:
                    fallback = self.brain._consciousness.compose_live_reply(
                        self._pending_user_text, self.brain
                    )
                    if fallback:
                        self._append_ai_message(fallback)
                        self._speak_fallback_reply(fallback)
                    else:
                        for msg in outbound:
                            self._append_ai_message(msg)
            else:
                for msg in outbound:
                    self._append_ai_message(msg)
            self._reply_pending = False
            self._pending_user_text = ""
            dpg.set_value("thinking_indicator", "")
        if self._reply_pending and (time.perf_counter() - self._reply_time) > 8.0:
            fallback = self.brain._consciousness.compose_live_reply(
                self._pending_user_text, self.brain
            )
            if fallback:
                self._append_ai_message(fallback)
                self._speak_fallback_reply(fallback)
            self._reply_pending = False
            self._pending_user_text = ""
            dpg.set_value("thinking_indicator", "")
        stream = list(self.brain._consciousness.stream)
        new_n = len(stream) - self._stream_count
        if new_n > 0:
            for entry in stream[-new_n:]:
                self._thoughts.append(entry)
            self._stream_count = len(stream)
        elif new_n < 0:
            self._stream_count = len(stream)
        dpg.set_value("conversation_text", "\n".join(self._conversation))
        dpg.set_value("thoughts_text", "\n".join(self._thoughts))

    def _update_camera(self) -> None:
        frame = self.brain.latest_frame
        if frame is None:
            return
        try:
            import cv2

            h, w = frame.shape[:2]
            scale = min(self._camera_width / max(1, w), self._camera_height / max(1, h))
            resized = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
            canvas = np.zeros(
                (self._camera_height, self._camera_width, 3), dtype=np.uint8
            )
            off_y = (self._camera_height - resized.shape[0]) // 2
            off_x = (self._camera_width - resized.shape[1]) // 2
            canvas[
                off_y : off_y + resized.shape[0], off_x : off_x + resized.shape[1]
            ] = resized
            rgba = np.empty(
                (self._camera_height, self._camera_width, 4), dtype=np.float32
            )
            rgba[:, :, 0] = canvas[:, :, 2] / 255.0
            rgba[:, :, 1] = canvas[:, :, 1] / 255.0
            rgba[:, :, 2] = canvas[:, :, 0] / 255.0
            rgba[:, :, 3] = 1.0
            dpg.set_value("camera_texture", rgba.flatten())
            dets = self.brain.latest_detections
            dpg.set_value("camera_status", "  ".join(dets[:8]))
        except Exception as exc:
            dpg.set_value("camera_status", f"camera render error: {exc}")

    def _update_anatomy(self) -> None:
        drawlist = "anatomy_drawlist"
        dpg.delete_item(drawlist, children_only=True)
        acts = self.brain.region_activity.copy()
        rect_size = dpg.get_item_rect_size(drawlist)
        width = int(rect_size[0]) if rect_size and rect_size[0] > 0 else 1080
        height = int(rect_size[1]) if rect_size and rect_size[1] > 0 else 420
        x_scale = width * 0.14
        y_scale = height * 0.22
        offset_x = width * 0.48
        offset_y = height * 0.50
        for s, t, w in ANAT_EDGES:
            x1, y1 = POS[s]
            x2, y2 = POS[t]
            a = max(acts.get(s, 0.0), acts.get(t, 0.0))
            color = (80, 120, 180, min(255, 70 + int(a * 180)))
            dpg.draw_line(
                (offset_x + x1 * x_scale, offset_y - y1 * y_scale),
                (offset_x + x2 * x_scale, offset_y - y2 * y_scale),
                color=color,
                thickness=1.0 + a * 4.0,
                parent=drawlist,
            )
        for name, (x, y) in POS.items():
            a = acts.get(name, 0.0)
            radius = 16 + a * 22
            color = (60 + int(a * 130), 90 + int(a * 80), 180 + int(a * 60), 255)
            center = (offset_x + x * x_scale, offset_y - y * y_scale)
            dpg.draw_circle(
                center,
                radius,
                color=(200, 220, 255, 200),
                fill=color,
                thickness=2,
                parent=drawlist,
            )
            dpg.draw_text(
                (center[0] - 24, center[1] - 8),
                LABELS.get(name, name),
                color=TEXT,
                size=16,
                parent=drawlist,
            )
            dpg.draw_text(
                (center[0] - 18, center[1] + 12),
                f"{a:.2f}",
                color=(160, 210, 255),
                size=14,
                parent=drawlist,
            )

    def _project_spatial_point(
        self,
        point: Tuple[float, float, float],
        bounds: Dict[str, float],
        width: int,
        height: int,
    ) -> Tuple[float, float]:
        x, y, z = point
        x_mid = (bounds["min_x"] + bounds["max_x"]) * 0.5
        y_mid = (bounds["min_y"] + bounds["max_y"]) * 0.5
        z_mid = (bounds["min_z"] + bounds["max_z"]) * 0.5
        x_span = max(bounds["max_x"] - bounds["min_x"], 1.0)
        y_span = max(bounds["max_y"] - bounds["min_y"], 1.0)
        z_span = max(bounds["max_z"] - bounds["min_z"], 1.0)
        nx = (x - x_mid) / x_span
        ny = (y - y_mid) / y_span
        nz = (z - z_mid) / z_span
        px = width * 0.50 + (nx - ny * 0.55) * width * 0.60
        py = height * 0.58 - (nz * height * 0.42) - (ny * height * 0.18)
        return px, py

    def _update_spatial(self) -> None:
        drawlist = "spatial_drawlist"
        dpg.delete_item(drawlist, children_only=True)
        snapshot = self.brain.spatial_layout_snapshot()
        regions = snapshot.get("regions", [])
        if not regions:
            dpg.set_value("spatial_text", "Keine räumlichen Layoutdaten verfügbar.")
            return
        rect_size = dpg.get_item_rect_size(drawlist)
        width = int(rect_size[0]) if rect_size and rect_size[0] > 0 else 1080
        height = int(rect_size[1]) if rect_size and rect_size[1] > 0 else 300

        all_points: List[Tuple[float, float, float]] = []
        for region in regions:
            centroid = tuple(region.get("centroid", (0.0, 0.0, 0.0)))
            all_points.append(centroid)
            for pt in region.get("sample_points", []):
                all_points.append((pt["x"], pt["y"], pt["z"]))
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        zs = [p[2] for p in all_points]
        bounds = {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "min_z": min(zs),
            "max_z": max(zs),
        }

        dpg.draw_rectangle(
            (8, 8),
            (width - 8, height - 8),
            color=(70, 90, 120, 120),
            thickness=1,
            parent=drawlist,
        )
        dpg.draw_text(
            (16, 12),
            "Isometrische Projektion der 3D-Region-Layouts",
            color=ACCENT,
            size=16,
            parent=drawlist,
        )

        acts = self.brain.region_activity.copy()
        for region in regions:
            name = str(region.get("name", ""))
            color = SPATIAL_COLORS.get(name, (180, 200, 255))
            centroid = tuple(region.get("centroid", (0.0, 0.0, 0.0)))
            activity = float(acts.get(name, 0.0))
            sample_points = region.get("sample_points", [])[:18]
            for pt in sample_points:
                proj = self._project_spatial_point(
                    (pt["x"], pt["y"], pt["z"]), bounds, width, height
                )
                alpha = min(220, 70 + int(activity * 120))
                dpg.draw_circle(
                    proj,
                    2.0 + activity * 1.2,
                    color=(*color, alpha),
                    fill=(*color, max(40, alpha // 2)),
                    parent=drawlist,
                )
            c_proj = self._project_spatial_point(centroid, bounds, width, height)
            radius = 8.0 + activity * 10.0
            dpg.draw_circle(
                c_proj,
                radius,
                color=(*color, 255),
                fill=(*color, 110),
                thickness=2,
                parent=drawlist,
            )
            dpg.draw_text(
                (c_proj[0] + 10, c_proj[1] - 10),
                LABELS.get(name, name),
                color=TEXT,
                size=14,
                parent=drawlist,
            )

        debug_lines = []
        for region in regions:
            name = str(region.get("name", ""))
            centroid = tuple(region.get("centroid", (0.0, 0.0, 0.0)))
            extent = tuple(region.get("extent", (0.0, 0.0, 0.0)))
            debug_lines.append(
                f"{LABELS.get(name, name)}  c=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})  "
                f"e=({extent[0]:.1f}, {extent[1]:.1f}, {extent[2]:.1f})  act={acts.get(name, 0.0):.2f}"
            )
        dpg.set_value("spatial_text", "\n".join(debug_lines[:12]))

    def _update_synapses(self) -> None:
        region_to_idx = {name: idx for idx, name in enumerate(REGION_NAMES)}
        syns = self.brain._inter_synapses
        stride = max(1, len(syns) // 500) if syns else 1
        sample = syns[::stride][:500]
        xs = []
        ys = []
        delays = []
        dists = []
        for syn in sample:
            pre_region = getattr(syn.pre, "region", "")
            post_region = getattr(syn.post, "region", "")
            if pre_region not in region_to_idx or post_region not in region_to_idx:
                continue
            xs.append(region_to_idx[pre_region] + ((syn.sid % 7) - 3) * 0.05)
            ys.append(region_to_idx[post_region] + (((syn.sid // 7) % 7) - 3) * 0.05)
            delays.append(float(getattr(syn, "delay", 0.0)))
            dists.append(float(getattr(syn, "distance", 0.0)))
        dpg.set_value("synapse_scatter", [xs, ys])
        avg_w = float(np.mean([s.weight for s in sample])) if sample else 0.0
        avg_d = float(np.mean(delays)) if delays else 0.0
        avg_dist = float(np.mean(dists)) if dists else 0.0
        dpg.set_value(
            "synapse_text",
            f"sampled={len(sample)} total={len(syns):,} avg_w={avg_w:.2f} avg_delay={avg_d:.2f} avg_dist={avg_dist:.2f}",
        )

    def update(self) -> None:
        self._sync_viewport_layout()
        while not _INPUT_QUEUE.empty():
            try:
                self._handle_input(_INPUT_QUEUE.get_nowait())
            except queue.Empty:
                break
        if self._frame_times and (time.perf_counter() - self._frame_times[-1]) > 0.04:
            self._send_current_head_pose()
            self._frame_times.clear()
        self._sync_controls()
        self._update_status()
        self._update_conversation()
        # DPG 2.x: get_value on a tab_bar returns an integer UUID, not the tag string.
        # Resolve the alias so all comparisons below work correctly.
        _tab_id = dpg.get_value("main_tabs")
        try:
            active_tab = dpg.get_item_alias(_tab_id) if _tab_id else None
        except Exception:
            active_tab = None
        if active_tab in (None, "camera_tab"):
            self._update_camera()
        if active_tab in (None, "anatomy_tab"):
            self._update_anatomy()
        if active_tab in (None, "spatial_tab"):
            self._update_spatial()
        if active_tab in (None, "synapses_tab"):
            self._update_synapses()
        if active_tab == "social_tab":
            self._update_social()
        if active_tab == "episodes_tab":
            self._update_episodes()
        if active_tab == "skills_tab":
            self._update_skills()
        if active_tab == "concepts_tab":
            self._update_concepts()
        if active_tab == "kognition_tab":
            self._update_kognition()
        if active_tab == "systemik_tab":
            self._update_systemik()

    def run(self) -> int:
        self.build()
        while dpg.is_dearpygui_running():
            self.update()
            dpg.render_dearpygui_frame()
        self.shutdown()
        return 0

    def shutdown(self) -> None:
        try:
            self.brain.stop()
        except Exception:
            pass
        try:
            dpg.stop_dearpygui()
        except Exception:
            pass


def main(args=None) -> int:
    threading.Thread(target=input_thread, daemon=True).start()
    app = FastMonitor(args)

    def _shutdown(*_) -> None:
        app.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
