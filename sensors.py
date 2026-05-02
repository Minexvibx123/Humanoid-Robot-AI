"""
sensors.py — Sensory Input Encoders

Converts real-world signals (camera frames, audio chunks) into
spike-compatible current vectors injected into sensory neurons.

Principle: Rate coding
  - High stimulus intensity  → high injected current  → faster spiking
  - Low  stimulus intensity  → low  injected current  → slow / no spiking

No learned features here — that emerges in the cortex via STDP.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────
# Visual Encoder
# ─────────────────────────────────────────────────────────────


class VisualEncoder:
    """
    Reads frames from a webcam and converts them to a 1-D current vector
    of length `n_neurons` suitable for injection into SensoryInputRegion.

    Features extracted per frame (no ML, pure signal processing):
      - Mean luminance
      - Local contrast (std dev in patches)
      - Horizontal & vertical gradient magnitude (Sobel-like)
      - Temporal difference (motion) vs previous frame
    """

    def __init__(self, n_neurons: int = 64, camera_index: int = 0) -> None:
        self.n_neurons = n_neurons
        self.camera_index = camera_index
        self._cap = None
        self._prev_gray: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._latest_currents: List[float] = [0.0] * n_neurons
        self._latest_frame = None  # latest BGR frame — written by capture thread
        self._running = False
        self._cap_thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> bool:
        try:
            import cv2

            self._cap = cv2.VideoCapture(self.camera_index)
            # Request 640×480 @ 30fps (camera will use closest supported mode)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only latest frame
            if not self._cap.isOpened():
                return False
            # Warm up: grab the first frame to let the driver initialize
            self._cap.grab()
            self._running = True
            # Background capture thread — reads frames as fast as camera delivers them
            # Brain._tick() just reads `_latest_frame` without blocking
            self._cap_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="CamCapture"
            )
            self._cap_thread.start()
            return True
        except Exception:
            return False

    def _capture_loop(self) -> None:
        """Continuously grabs frames and stores the latest one. Never blocks callers."""
        import cv2

        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.flip(frame, 0)
                # Resize to 320×240 for fast feature extraction + detection
                small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
                with self._lock:
                    self._latest_frame = small
                    gray = (
                        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        / 255.0
                    )
                    feats = self._extract_features(gray, small)
                    self._latest_currents = self._features_to_currents(feats)

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()

    # ── per-tick call (NON-BLOCKING) ──────────────────────────

    def encode(self) -> List[float]:
        """Return the latest pre-computed current vector — never blocks on camera."""
        with self._lock:
            return list(self._latest_currents)

    def _extract_features(self, gray: np.ndarray, bgr: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        features: List[float] = []

        # 1) Overall luminance
        features.append(float(gray.mean()))

        # 2) Spatial contrast (std of 8×8 patches)
        patch_h, patch_w = h // 4, w // 4
        for pi in range(4):
            for pj in range(4):
                patch = gray[
                    pi * patch_h : (pi + 1) * patch_h, pj * patch_w : (pj + 1) * patch_w
                ]
                features.append(float(patch.std()))

        # 3) Horizontal + vertical gradient magnitude (simple finite diff)
        grad_x = np.abs(np.diff(gray, axis=1)).mean()
        grad_y = np.abs(np.diff(gray, axis=0)).mean()
        features.append(float(grad_x))
        features.append(float(grad_y))

        # 4) Motion (temporal difference)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            motion = float(np.abs(gray - self._prev_gray).mean())
        else:
            motion = 0.0
        features.append(motion)
        self._prev_gray = gray

        # 5) Colour channels mean (B, G, R)
        for c in range(3):
            features.append(float(bgr[:, :, c].mean() / 255.0))

        return np.array(features, dtype=np.float32)

    def _features_to_currents(self, features: np.ndarray) -> List[float]:
        """
        Tile & scale features to fill n_neurons slots with currents in [0..15] nA.
        """
        n = self.n_neurons
        tiled = np.resize(features, n)
        # Normalise to [0, 1] then scale to biological range
        lo, hi = tiled.min(), tiled.max()
        if hi > lo:
            tiled = (tiled - lo) / (hi - lo)
        return (tiled * 22.0).tolist()


# ─────────────────────────────────────────────────────────────
# Vision Analyzer  (object / person / hand-gesture detection)
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectionTarget:
    label: str
    center_x: float
    center_y: float
    width: float
    height: float
    score: float = 0.0
    detail: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height,
            "score": self.score,
            "detail": self.detail,
        }


class VisionAnalyzer:
    """
    Runs on every camera frame alongside VisualEncoder.
    Uses MediaPipe (if available) for face, hand and pose detection.
    Falls back to OpenCV's built-in HOG person detector and
    Haar-cascade face detector when MediaPipe is not installed.

    Outputs
    -------
    detections : List[str]
        Human-readable labels for the current frame, e.g.
        ["person", "face", "hand:open", "hand:fist", "hand:point"]
        These are injected into the brain as concepts + speech tokens.

    detection_strength : float  [0, 1]
        Salience score — higher when more / larger objects are detected.
        Used to modulate sensory_visual injection strength.
    """

    # MediaPipe hand-gesture recognition: landmark → gesture name
    _FINGER_TIPS = [4, 8, 12, 16, 20]
    _FINGER_PIPS = [3, 6, 10, 14, 18]

    # Model files (downloaded once to ./models/)
    _MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    def __init__(self) -> None:
        self._face_det = None
        self._hand_lmk = None
        self._pose_lmk = None
        self._hog = None
        self._face_casc = None
        self._use_tasks = False  # True when new MediaPipe Tasks API is available
        self._ready = False

        self.detections: List[str] = []
        self.targets: List[Dict[str, object]] = []
        self.detection_strength: float = 0.0
        self._prev_strength: float = 0.0

        self._running_worker: bool = False
        self._pending_frame: Optional["np.ndarray"] = None
        self._worker_event: Optional[threading.Event] = None
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Initialise detectors and start the async worker thread."""
        ok = False

        # ── Try MediaPipe Tasks API (v0.10+) ──────────────────
        try:
            from mediapipe.tasks.python import vision as _mpv
            from mediapipe.tasks.python.core import base_options as _bo

            face_path = os.path.join(self._MODEL_DIR, "face_detector.task")
            hand_path = os.path.join(self._MODEL_DIR, "hand_landmarker.task")
            pose_path = os.path.join(self._MODEL_DIR, "pose_landmarker.task")

            if os.path.exists(face_path):
                self._face_det = _mpv.FaceDetector.create_from_options(
                    _mpv.FaceDetectorOptions(
                        base_options=_bo.BaseOptions(model_asset_path=face_path),
                        min_detection_confidence=0.40,
                    )
                )
            if os.path.exists(hand_path):
                self._hand_lmk = _mpv.HandLandmarker.create_from_options(
                    _mpv.HandLandmarkerOptions(
                        base_options=_bo.BaseOptions(model_asset_path=hand_path),
                        num_hands=2,
                        min_hand_detection_confidence=0.40,
                        min_hand_presence_confidence=0.35,
                        min_tracking_confidence=0.35,
                    )
                )
            if os.path.exists(pose_path):
                self._pose_lmk = _mpv.PoseLandmarker.create_from_options(
                    _mpv.PoseLandmarkerOptions(
                        base_options=_bo.BaseOptions(model_asset_path=pose_path),
                        min_pose_detection_confidence=0.35,
                        min_pose_presence_confidence=0.35,
                        min_tracking_confidence=0.35,
                    )
                )
            if self._face_det or self._hand_lmk or self._pose_lmk:
                self._use_tasks = True
                ok = True
        except Exception:
            pass

        # ── Fall back to OpenCV ───────────────────────────────
        if not ok:
            try:
                import cv2

                self._hog = cv2.HOGDescriptor()
                self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                casc_path = (
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self._face_casc = cv2.CascadeClassifier(casc_path)
                ok = True
            except Exception:
                pass

        self._ready = ok
        self._running_worker = True
        self._worker_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="VisionWorker"
        )
        self._worker_thread.start()
        return ok

    def stop(self) -> None:
        self._running_worker = False
        if self._worker_event:
            self._worker_event.set()
        try:
            if self._face_det:
                self._face_det.close()
            if self._hand_lmk:
                self._hand_lmk.close()
            if self._pose_lmk:
                self._pose_lmk.close()
        except Exception:
            pass

    def submit(self, frame_bgr: "np.ndarray") -> None:
        """Non-blocking: queue latest frame; worker picks it up."""
        self._pending_frame = frame_bgr
        if self._worker_event:
            self._worker_event.set()

    def _worker(self) -> None:
        while self._running_worker:
            self._worker_event.wait(timeout=0.05)
            self._worker_event.clear()
            frame = self._pending_frame
            self._pending_frame = None
            if frame is None or not self._ready:
                continue
            h, w = frame.shape[:2]
            area = h * w
            if self._use_tasks:
                found, targets = self._analyze_tasks(frame, h, w, area)
            else:
                found, targets = self._analyze_opencv(frame, h, w, area)
            self.detections = found
            self.targets = targets
            n = len(found)
            coverage = sum(
                float(t.get("width", 0.0)) * float(t.get("height", 0.0))
                for t in targets[:4]
            )
            raw = min(1.0, n * 0.18 + coverage * 1.25 + self._prev_strength * 0.3)
            self.detection_strength = raw
            self._prev_strength = raw

    def analyze(self, frame_bgr: "np.ndarray") -> List[str]:
        self.submit(frame_bgr)
        return self.detections

    # ── MediaPipe Tasks API path (v0.10+) ─────────────────────

    def _norm_target(
        self,
        label: str,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        score: float = 0.0,
        detail: str = "",
    ) -> Dict[str, object]:
        cx = max(0.0, min(1.0, center_x))
        cy = max(0.0, min(1.0, center_y))
        tw = max(0.0, min(1.0, width))
        th = max(0.0, min(1.0, height))
        return DetectionTarget(
            label, cx, cy, tw, th, max(0.0, min(1.0, score)), detail
        ).as_dict()

    def _bbox_target(
        self,
        label: str,
        x: float,
        y: float,
        box_w: float,
        box_h: float,
        frame_w: float,
        frame_h: float,
        score: float = 0.0,
        detail: str = "",
    ) -> Dict[str, object]:
        return self._norm_target(
            label,
            (x + box_w * 0.5) / max(1.0, frame_w),
            (y + box_h * 0.5) / max(1.0, frame_h),
            box_w / max(1.0, frame_w),
            box_h / max(1.0, frame_h),
            score,
            detail,
        )

    def _analyze_tasks(
        self, bgr, h, w, area
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        import cv2

        try:
            import mediapipe as mp
        except Exception:
            return [], []

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        found: List[str] = []
        targets: List[Dict[str, object]] = []

        # Face
        if self._face_det:
            try:
                res = self._face_det.detect(mp_img)
                for det in res.detections:
                    bbox = det.bounding_box
                    score = float(det.categories[0].score) if det.categories else 0.0
                    found.append("face")
                    targets.append(
                        self._bbox_target(
                            "face",
                            bbox.origin_x,
                            bbox.origin_y,
                            bbox.width,
                            bbox.height,
                            w,
                            h,
                            score,
                        )
                    )
                    if det.categories and det.categories[0].score > 0.45:
                        if "person" not in found:
                            found.append("person")
                        targets.append(
                            self._bbox_target(
                                "person",
                                bbox.origin_x,
                                bbox.origin_y,
                                bbox.width,
                                bbox.height,
                                w,
                                h,
                                score,
                                "face_proxy",
                            )
                        )
            except Exception:
                pass

        # Pose → person
        if self._pose_lmk:
            try:
                res = self._pose_lmk.detect(mp_img)
                if res.pose_landmarks:
                    if "person" not in found:
                        found.append("person")
                    for pose_landmarks in res.pose_landmarks[:1]:
                        xs = [lm.x for lm in pose_landmarks]
                        ys = [lm.y for lm in pose_landmarks]
                        min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
                        min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
                        targets.append(
                            self._norm_target(
                                "person",
                                (min_x + max_x) * 0.5,
                                (min_y + max_y) * 0.5,
                                max_x - min_x,
                                max_y - min_y,
                                0.6,
                                "pose_bbox",
                            )
                        )
            except Exception:
                pass

        # Hands → gesture
        if self._hand_lmk:
            try:
                res = self._hand_lmk.detect(mp_img)
                for hand_lms in res.hand_landmarks:
                    gesture = self._classify_gesture_tasks(hand_lms)
                    found.append(f"hand:{gesture}")
                    if "hand" not in found:
                        found.append("hand")
                    xs = [lm.x for lm in hand_lms]
                    ys = [lm.y for lm in hand_lms]
                    min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
                    min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
                    targets.append(
                        self._norm_target(
                            "hand",
                            (min_x + max_x) * 0.5,
                            (min_y + max_y) * 0.5,
                            max_x - min_x,
                            max_y - min_y,
                            0.65,
                            gesture,
                        )
                    )
            except Exception:
                pass

        return found, targets

    # ── OpenCV fallback ───────────────────────────────────────

    def _analyze_opencv(
        self, bgr, h, w, area
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        import cv2

        found: List[str] = []
        targets: List[Dict[str, object]] = []

        # Face (fast Haar, already at 320×240)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self._face_casc is not None:
            faces = self._face_casc.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
            )
            if len(faces) > 0:
                found.append("face")
                found.append("person")
                for x, y, fw, fh in faces[:2]:
                    targets.append(self._bbox_target("face", x, y, fw, fh, w, h, 0.55))
                    targets.append(
                        self._bbox_target(
                            "person", x, y, fw, fh, w, h, 0.5, "face_proxy"
                        )
                    )

        # Person: HOG (input already 320×240)
        if self._hog is not None and "person" not in found:
            rects, _ = self._hog.detectMultiScale(
                bgr, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            if len(rects) > 0:
                found.append("person")
                for x, y, pw, ph in rects[:2]:
                    targets.append(
                        self._bbox_target("person", x, y, pw, ph, w, h, 0.45, "hog")
                    )

        # Skin-colour hand detection (HSV)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        skin_ratio = mask.sum() / 255.0 / area
        if 0.04 < skin_ratio < 0.35:
            found.append("hand")
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, hw, hh = cv2.boundingRect(contour)
                targets.append(
                    self._bbox_target(
                        "hand", x, y, hw, hh, w, h, min(0.6, skin_ratio * 2.0), "skin"
                    )
                )

        return found, targets

    # ── Gesture classifier ────────────────────────────────────

    def _classify_gesture_tasks(self, landmarks) -> str:
        """Classify from Tasks API NormalizedLandmark list (21 points)."""
        tips = self._FINGER_TIPS
        pips = self._FINGER_PIPS
        extended = [landmarks[t].y < landmarks[p].y for t, p in zip(tips[1:], pips[1:])]
        thumb_ext = abs(landmarks[4].x - landmarks[0].x) > abs(
            landmarks[3].x - landmarks[0].x
        )
        n_ext = sum(extended)
        if n_ext == 4 and thumb_ext:
            return "open"
        if n_ext == 0 and not thumb_ext:
            return "fist"
        if n_ext == 1 and extended[0]:
            return "point"
        if n_ext == 2 and extended[0] and extended[1]:
            return "victory"
        if n_ext == 0 and thumb_ext:
            return "thumbs_up" if landmarks[4].y < landmarks[3].y else "thumbs_down"
        if n_ext >= 3:
            return "open"
        return "gesture"

    def _classify_gesture(self, landmarks) -> str:
        """Legacy solutions-API wrapper (same logic)."""
        return self._classify_gesture_tasks(landmarks)


# ─────────────────────────────────────────────────────────────
# Audio Encoder
# ─────────────────────────────────────────────────────────────


class AudioEncoder:
    """
    Captures microphone audio and converts frequency-domain features
    to a current vector for auditory cortex neurons.

    Features per chunk:
      - Power in 8 frequency bands (log-spaced, cochlea-inspired)
      - Overall RMS energy
      - Zero-crossing rate (roughness / consonant cues)
      - Delta energy (onset detection)

    Domain H additions:
      - speech_energy_ema: normalised EMA of RMS energy [0,1]
      - speech_tempo_var: onset-rate variability EMA (animated vs flat)
      - inferred_affect: "calm"|"excited"|"tense"|"sad"|"unknown"
    """

    def __init__(
        self,
        n_neurons: int = 32,
        sample_rate: int = 16000,
        chunk_size: int = 1600,  # 100 ms
    ) -> None:
        self.n_neurons = n_neurons
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._stream = None
        self._buffer = np.zeros(chunk_size, dtype=np.float32)
        self._prev_rms = 0.0
        self._lock = threading.Lock()
        self._running = False

        # ── Domain H: prosodic affect tracking ───────────────
        self.speech_energy_ema: float = 0.0   # normalised EMA [0,1]
        self.speech_tempo_var: float = 0.0    # onset variability EMA
        self.inferred_affect: str = "unknown" # "calm|excited|tense|sad|unknown"
        self._onset_history: List[float] = [] # recent onset magnitudes for var

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> bool:
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            return True
        except Exception:
            return False

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            self._buffer = indata[:, 0].copy()

    # ── per-tick call ─────────────────────────────────────────

    def encode(self) -> List[float]:
        """Return current vector from latest audio buffer."""
        with self._lock:
            buf = self._buffer.copy()

        features = self._extract_features(buf)
        currents = self._features_to_currents(features)
        return currents

    def _extract_features(self, buf: np.ndarray) -> np.ndarray:
        features: List[float] = []

        # RMS energy
        rms = float(np.sqrt(np.mean(buf**2)))
        features.append(rms)

        # 8 log-spaced frequency bands via FFT
        spectrum = np.abs(np.fft.rfft(buf * np.hanning(len(buf))))
        freqs = np.fft.rfftfreq(len(buf), 1.0 / self.sample_rate)
        band_edges = np.logspace(
            math.log10(20), math.log10(self.sample_rate / 2), num=9
        )
        for i in range(8):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            power = float(spectrum[mask].mean()) if mask.any() else 0.0
            features.append(power)

        # Zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(buf)))) / 2.0)
        features.append(zcr)

        # Onset / delta energy
        delta = rms - self._prev_rms
        onset = max(0.0, delta)
        features.append(onset)
        self._prev_rms = rms

        # ── Domain H: update prosodic affect estimates ────────
        self._update_affect(rms, onset, zcr)

        return np.array(features, dtype=np.float32)

    def _update_affect(self, rms: float, onset: float, zcr: float) -> None:
        """EMA-update prosodic affect fields from raw audio features."""
        # Normalise energy (heuristic: 0.05 = typical speech RMS)
        _norm_e = min(1.0, rms / 0.05)
        self.speech_energy_ema = self.speech_energy_ema * 0.92 + _norm_e * 0.08

        # Tempo variability: variance of recent onset magnitudes
        self._onset_history.append(onset)
        if len(self._onset_history) > 20:
            self._onset_history.pop(0)
        if len(self._onset_history) >= 4:
            _var = float(np.var(self._onset_history))
            self.speech_tempo_var = self.speech_tempo_var * 0.90 + min(1.0, _var * 80) * 0.10

        # Simple affect classifier:
        #   excited = high energy + high tempo var
        #   tense   = high zcr (rough/strained) + moderate energy
        #   sad     = low energy + low tempo var
        #   calm    = moderate energy + low tempo var
        _e = self.speech_energy_ema
        _tv = self.speech_tempo_var
        _zcr_n = min(1.0, zcr * 4.0)  # normalise ZCR
        if _e > 0.6 and _tv > 0.4:
            self.inferred_affect = "excited"
        elif _zcr_n > 0.6 and _e > 0.3:
            self.inferred_affect = "tense"
        elif _e < 0.15 and _tv < 0.1:
            self.inferred_affect = "sad"
        elif _e > 0.1:
            self.inferred_affect = "calm"
        # else: keep previous — avoid flicker on silence

    def _features_to_currents(self, features: np.ndarray) -> List[float]:
        n = self.n_neurons
        tiled = np.resize(features, n)
        lo, hi = tiled.min(), tiled.max()
        if hi > lo:
            tiled = (tiled - lo) / (hi - lo)
        return (tiled * 20.0).tolist()


# ─────────────────────────────────────────────────────────────
# Speech-to-Text (overlay — does not affect spike encoding)
# ─────────────────────────────────────────────────────────────


class SpeechListener:
    """
    Runs SpeechRecognition in a background thread.
    Latest recognised text is available via `.latest_text`.
    """

    def __init__(self, language: str = "de-DE") -> None:
        self.language = language
        self.latest_text = ""
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _listen_loop(self) -> None:
        """
        Record audio via sounddevice (no PyAudio needed), then send
        raw PCM to SpeechRecognition's AudioData for recognition.
        """
        try:
            import io
            import wave

            import sounddevice as sd
            import speech_recognition as sr

            r = sr.Recognizer()
            sample_rate = 16000
            chunk_sec = 4

            while self._running:
                try:
                    pcm = sd.rec(
                        int(chunk_sec * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype="int16",
                        blocking=True,
                    )
                    # Pack into an in-memory WAV for SpeechRecognition
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sample_rate)
                        wf.writeframes(pcm.tobytes())
                    buf.seek(0)
                    audio = sr.AudioData(buf.read(), sample_rate, 2)
                    text = r.recognize_google(audio, language=self.language)
                    if text:
                        self.latest_text = text.strip()
                except Exception:
                    pass
        except Exception:
            pass
