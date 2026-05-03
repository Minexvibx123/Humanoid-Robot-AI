"""
speech_output.py — Text-to-Speech Output Channel

Three-tier backend with automatic fallback:

  1. F5-TTS  (zero-shot offline voice cloning from a reference WAV)
       Set env:  ALBEDO_VOICE_WAV=<path to 10-30s clean voice clip>
                 ALBEDO_VOICE_TEXT=<transcript of that clip>
       OR place a sidecar <wav_path_without_ext>.txt next to the WAV.
       Model (~300 MB) auto-downloads from HuggingFace on first use.

  2. Kokoro-ONNX (offline, high quality — no cloning, uses female preset voice)
       Models auto-downloaded from HuggingFace on first use (~300 MB, cached).
       Requires:  kokoro-onnx  huggingface_hub  soundfile  sounddevice

  3. pyttsx3  (always available Windows SAPI5 fallback)

Force a backend via env:  TTS_BACKEND=f5tts | kokoro | pyttsx3
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from dialogue_manager import UtterancePlan

# ─── backend selection ────────────────────────────────────────────────────────
# Kokoro voice to use when running in offline mode.  Female preset voices:
#   af_heart / af_nova / af_sky  — American female (natural)
#   jf_gongitsune / jf_nezumi    — Japanese female (anime-style)
KOKORO_VOICE = os.environ.get("ALBEDO_KOKORO_VOICE", "af_heart")
KOKORO_SPEED = float(os.environ.get("ALBEDO_KOKORO_SPEED", "0.9"))
KOKORO_LANG = os.environ.get("ALBEDO_KOKORO_LANG", "en-us")

# Path where kokoro ONNX model files are cached after first download
_KOKORO_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models", "kokoro")


def _install_torchaudio_fallback(ref_wav: str) -> bool:
    """Patch torchaudio.load to use soundfile when TorchCodec is unavailable."""
    try:
        import torchaudio

        try:
            torchaudio.load(ref_wav)
            return True
        except RuntimeError as exc:
            if "Could not load libtorchcodec" not in str(exc):
                return False

        import numpy as np
        import soundfile as sf
        import torch

        def _soundfile_load(path, *args, **kwargs):
            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            tensor = torch.from_numpy(np.ascontiguousarray(data.T))
            return tensor, sample_rate

        torchaudio.load = _soundfile_load
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Speech request
# ─────────────────────────────────────────────────────────────


@dataclass
class _SpeechRequest:
    text: str
    rate: int = 170  # words per minute (pyttsx3 / kokoro speed hint)
    volume: float = 0.9
    voice_id: Optional[str] = None
    priority: int = 0
    timestamp: float = 0.0
    pitch_shift: float = 0.0  # [-1, 1] — prosodic pitch offset
    emphasis_words: List[str] = field(default_factory=list)
    pause_before_ms: int = 0  # silence to insert before this request


# ─────────────────────────────────────────────────────────────
# Prosody tag parser
# ─────────────────────────────────────────────────────────────

# Supported inline tags (case-insensitive):
#   [P0.3]  or [P300ms] — insert pause of N seconds / ms before next segment
#   [UP]                — rising intonation (pitch_shift +0.35)
#   [SLOW]              — deliberate/uncertain delivery (rate * 0.72)
#   [SOFT]              — quieter delivery (volume * 0.80)
#   [EMPH word]         — add word to emphasis_words
_PROSODY_TAG_RE = re.compile(
    r"\[(?:"
    r"P(\d+(?:\.\d+)?)(ms)?|"   # [P0.3] or [P300ms]
    r"(UP)|"
    r"(SLOW)|"
    r"(SOFT)|"
    r"EMPH\s+(\S+)"
    r")\]",
    re.IGNORECASE,
)


def _parse_prosody_tags(
    text: str, base_rate: int = 170, base_volume: float = 0.9
) -> List["_SpeechRequest"]:
    """
    Parse inline prosody tags from *text* and return a list of _SpeechRequest
    segments.  The tags are stripped from the rendered text.

    A [Px] tag splits the stream: the text before the tag becomes one request,
    and the following text starts a new request with a leading pause.
    Other tags ([UP] [SLOW] [SOFT] [EMPH]) accumulate into the *current* segment.

    Returns at least one _SpeechRequest (with pause_before_ms=0 if no [P] tag).
    """
    segments: List[_SpeechRequest] = []
    pending_pause_ms = 0
    current_rate = base_rate
    current_volume = base_volume
    current_pitch = 0.0
    current_emph: List[str] = []

    last_end = 0
    current_text_parts: List[str] = []

    def _flush(pause_ms: int) -> None:
        """Commit accumulated text as a new request."""
        raw = "".join(current_text_parts).strip()
        current_text_parts.clear()
        if raw:
            segments.append(
                _SpeechRequest(
                    text=raw,
                    rate=current_rate,
                    volume=current_volume,
                    pitch_shift=current_pitch,
                    emphasis_words=list(current_emph),
                    pause_before_ms=pause_ms,
                    timestamp=time.time(),
                )
            )
        current_emph.clear()

    for m in _PROSODY_TAG_RE.finditer(text):
        # Accumulate literal text before this tag
        current_text_parts.append(text[last_end : m.start()])
        last_end = m.end()

        p_val, p_ms, is_up, is_slow, is_soft, emph_word = m.groups()

        if p_val is not None:
            # Pause tag — flush current segment, start new one after pause
            pause_ms_val = float(p_val)
            if p_ms:  # [P300ms] → already in ms
                pause_ms_val = int(pause_ms_val)
            else:  # [P0.3] → seconds
                pause_ms_val = int(pause_ms_val * 1000)
            _flush(pending_pause_ms)
            pending_pause_ms = pause_ms_val
            # Reset per-segment modifiers for the new segment
            current_rate = base_rate
            current_volume = base_volume
            current_pitch = 0.0
        elif is_up:
            current_pitch = min(1.0, current_pitch + 0.35)
        elif is_slow:
            current_rate = max(80, int(current_rate * 0.72))
        elif is_soft:
            current_volume = max(0.4, current_volume * 0.80)
        elif emph_word:
            current_emph.append(emph_word)

    # Flush remaining text
    current_text_parts.append(text[last_end:])
    _flush(pending_pause_ms)

    if not segments:
        # No tags at all — return plain request
        segments.append(
            _SpeechRequest(
                text=text.strip(),
                rate=base_rate,
                volume=base_volume,
                timestamp=time.time(),
            )
        )
    return segments


# ─────────────────────────────────────────────────────────────
# F5-TTS helper
# ─────────────────────────────────────────────────────────────


def _f5tts_load(
    ref_wav: str, ref_text: str
) -> "Optional[Tuple[object, str, str]]":
    """Load F5-TTS model and return (F5TTS_instance, ref_wav, ref_text) or None."""
    try:
        _install_torchaudio_fallback(ref_wav)
        from f5_tts.api import F5TTS  # type: ignore

        instance = F5TTS()
        return (instance, ref_wav, ref_text)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Kokoro-ONNX helper
# ─────────────────────────────────────────────────────────────


def _kokoro_load() -> Optional[object]:
    """Download + load the Kokoro ONNX model. Returns Kokoro instance or None."""
    try:
        from huggingface_hub import hf_hub_download
        from kokoro_onnx import Kokoro

        os.makedirs(_KOKORO_CACHE_DIR, exist_ok=True)
        model_dst = os.path.join(_KOKORO_CACHE_DIR, "kokoro-v1.0.onnx")
        voices_dst = os.path.join(_KOKORO_CACHE_DIR, "voices-v1.0.bin")

        if not os.path.exists(model_dst):
            model_dst = hf_hub_download(
                repo_id="hexgrad/Kokoro-82M",
                filename="kokoro-v1.0.onnx",
                local_dir=_KOKORO_CACHE_DIR,
            )
        if not os.path.exists(voices_dst):
            voices_dst = hf_hub_download(
                repo_id="hexgrad/Kokoro-82M",
                filename="voices-v1.0.bin",
                local_dir=_KOKORO_CACHE_DIR,
            )

        return Kokoro(model_dst, voices_dst)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Speech Output Engine
# ─────────────────────────────────────────────────────────────


class SpeechOutput:
    """
    Async TTS engine — three-tier backend with jaw-sync callbacks.

    Backend priority (automatic, overridable via TTS_BACKEND env var):
      1. F5-TTS      — offline zero-shot voice cloning from ALBEDO_VOICE_WAV
      2. Kokoro-ONNX — offline, high quality female voice (auto-downloads model)
      3. pyttsx3     — always-available Windows SAPI5 fallback

    Usage:
        tts = SpeechOutput()
        tts.start()
        tts.speak("Ich bin Albedo.")
        tts.stop()
    """

    DEFAULT_RATE = 170
    DEFAULT_VOLUME = 0.9
    QUEUE_MAXSIZE = 20

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._speaking: bool = False
        self._current_text: str = ""
        self._lock = threading.Lock()

        # Backend instances (initialised in start())
        self._backend: str = ""  # "f5tts" | "kokoro" | "pyttsx3"
        self._pyttsx3_engine = None
        self._kokoro = None
        self._f5tts = None
        self._f5tts_ref_wav: str = ""
        self._f5tts_ref_text: str = ""

        # Callbacks wired by brain.py
        self._on_start: Optional[Callable] = None
        self._on_end: Optional[Callable] = None
        self._on_motor_cue: Optional[Callable] = (
            None  # (cue_type, params) → motor action
        )

        # Brain.py sets these as properties
        self.on_start: Optional[Callable] = None
        self.on_end: Optional[Callable] = None

        self._history: List[str] = []
        self._total_utterances: int = 0
        self.enabled: bool = True
        self._startup_errors: List[str] = []

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> bool:
        if self._running:
            return True
        force = os.environ.get("TTS_BACKEND", "").lower()
        ok = False
        self._startup_errors = []

        # ── 1. F5-TTS (zero-shot offline voice cloning) ───────
        if force in ("", "f5tts"):
            wav_path = os.environ.get("ALBEDO_VOICE_WAV", "")
            if wav_path and os.path.exists(wav_path):
                # Resolve reference text: env var → sidecar .txt → skip
                ref_text = os.environ.get("ALBEDO_VOICE_TEXT", "").strip()
                if not ref_text:
                    sidecar = os.path.splitext(wav_path)[0] + ".txt"
                    if os.path.exists(sidecar):
                        with open(sidecar, encoding="utf-8") as _fh:
                            ref_text = _fh.read().strip()
                if ref_text:
                    try:
                        result = _f5tts_load(wav_path, ref_text)
                        if result is not None:
                            self._f5tts, self._f5tts_ref_wav, self._f5tts_ref_text = result
                            self._backend = "f5tts"
                            ok = True
                        else:
                            self._startup_errors.append("f5tts backend unavailable")
                    except Exception:
                        self._startup_errors.append("f5tts backend unavailable")
                else:
                    self._startup_errors.append(
                        "f5tts: no reference text (set ALBEDO_VOICE_TEXT or place .txt sidecar)"
                    )

        # ── 2. Kokoro-ONNX ────────────────────────────────────
        if not ok and force in ("", "kokoro"):
            try:
                self._kokoro = _kokoro_load()
                if self._kokoro is not None:
                    self._backend = "kokoro"
                    ok = True
                else:
                    self._startup_errors.append("kokoro backend unavailable")
            except Exception:
                self._startup_errors.append("kokoro backend unavailable")

        # ── 3. pyttsx3 fallback ───────────────────────────────
        if not ok and force in ("", "pyttsx3"):
            if self._ensure_pyttsx3_engine():
                self._backend = "pyttsx3"
                ok = True
            else:
                self._startup_errors.append("pyttsx3 backend unavailable")

        if not ok:
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="SpeechOutput"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────

    def speak(
        self,
        text: str,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        priority: int = 0,
    ) -> bool:
        if not self.enabled or not self._running or not text.strip():
            return False
        req = _SpeechRequest(
            text=text.strip(),
            rate=rate or self.DEFAULT_RATE,
            volume=volume or self.DEFAULT_VOLUME,
            priority=priority,
            timestamp=time.time(),
        )
        try:
            self._queue.put_nowait(req)
            return True
        except queue.Full:
            return False

    def speak_utterance(self, plan: "UtterancePlan") -> bool:
        """Route the full UtterancePlan through TTS with prosody + motor cues."""
        if not self.enabled or not self._running or not plan.text.strip():
            return False
        rate = int(self.DEFAULT_RATE * plan.speed_factor)
        rate = max(100, min(300, rate))
        # Fire pre-speech motor cues via callbacks
        if plan.head_nod and self._on_motor_cue:
            try:
                self._on_motor_cue("head_nod", {})
            except Exception:
                pass
        if plan.gaze_at_person and self._on_motor_cue:
            try:
                self._on_motor_cue("gaze_at_person", {"addressee": plan.addressee})
            except Exception:
                pass
        # Build speech request with prosody
        req = _SpeechRequest(
            text=plan.text.strip(),
            rate=rate,
            volume=self.DEFAULT_VOLUME,
            priority=0,
            timestamp=time.time(),
            pitch_shift=plan.pitch_shift,
            emphasis_words=list(plan.emphasis_words),
            pause_before_ms=getattr(plan, "deliberation_delay_ms", 0),
        )
        try:
            self._queue.put_nowait(req)
            return True
        except queue.Full:
            return False

    def interrupt(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def current_text(self) -> str:
        return self._current_text if self._speaking else ""

    @property
    def active_backend(self) -> str:
        """Which TTS backend is active: 'f5tts', 'kokoro', or 'pyttsx3'."""
        return self._backend

    @property
    def startup_errors(self) -> List[str]:
        return list(self._startup_errors)

    def startup_status(self) -> str:
        if self._running and self._backend:
            return f"TTS aktiv: {self._backend}"
        if self._startup_errors:
            return "TTS deaktiviert: " + "; ".join(dict.fromkeys(self._startup_errors))
        return "TTS deaktiviert"

    def _ensure_pyttsx3_engine(self) -> bool:
        if self._pyttsx3_engine is not None:
            return True
        try:
            import pyttsx3

            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", self.DEFAULT_RATE)
            self._pyttsx3_engine.setProperty("volume", self.DEFAULT_VOLUME)
            voices = self._pyttsx3_engine.getProperty("voices")
            for v in voices:
                n = v.name.lower()
                if (
                    "german" in n
                    or "deutsch" in n
                    or "de-de" in str(v.languages).lower()
                ):
                    self._pyttsx3_engine.setProperty("voice", v.id)
                    break
            return True
        except Exception:
            self._pyttsx3_engine = None
            return False

    def set_callbacks(
        self, on_start: Optional[Callable] = None, on_end: Optional[Callable] = None
    ) -> None:
        self._on_start = on_start
        self._on_end = on_end

    # ── Worker loop ───────────────────────────────────────────

    def _worker_loop(self) -> None:
        while self._running:
            try:
                req = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if req is None:
                break
            if not isinstance(req, _SpeechRequest):
                continue

            try:
                # Parse prosody tags — may produce multiple sub-segments
                segments = _parse_prosody_tags(
                    req.text,
                    base_rate=req.rate,
                    base_volume=req.volume,
                )
                # Transfer priority/voice_id from parent to all segments
                for seg in segments:
                    seg.priority = req.priority
                    seg.voice_id = req.voice_id
                    if req.pitch_shift and seg.pitch_shift == 0.0:
                        seg.pitch_shift = req.pitch_shift
                    if req.emphasis_words and not seg.emphasis_words:
                        seg.emphasis_words = list(req.emphasis_words)
                # Apply deliberation gap to the first segment
                if segments and req.pause_before_ms > 0:
                    segments[0].pause_before_ms = max(
                        segments[0].pause_before_ms, req.pause_before_ms
                    )

                with self._lock:
                    self._speaking = True
                    self._current_text = req.text

                _cb = self.on_start or self._on_start
                if _cb:
                    try:
                        _cb()
                    except Exception:
                        pass

                for seg in segments:
                    if not self._running:
                        break
                    # Honour leading pause (from [P] tag in previous segment)
                    if seg.pause_before_ms > 0:
                        time.sleep(seg.pause_before_ms / 1000.0)
                    if not seg.text:
                        continue
                    if self._backend == "f5tts":
                        self._speak_f5tts(seg)
                    elif self._backend == "kokoro":
                        self._speak_kokoro(seg)
                    else:
                        self._speak_pyttsx3(seg)

                self._history.append(req.text)
                if len(self._history) > 100:
                    self._history = self._history[-100:]
                self._total_utterances += 1

            except Exception:
                pass
            finally:
                with self._lock:
                    self._speaking = False
                    self._current_text = ""

                _cb = self.on_end or self._on_end
                if _cb:
                    try:
                        _cb()
                    except Exception:
                        pass

    # ── Backend implementations ───────────────────────────────

    def _speak_f5tts(self, req: _SpeechRequest) -> None:
        """Synthesise with F5-TTS (zero-shot voice cloning) and play via sounddevice."""
        import sounddevice as sd

        # speed param: F5-TTS accepts 0.5–2.0; map from WPM ratio
        speed = max(0.5, min(2.0, req.rate / self.DEFAULT_RATE))
        try:
            wav, sr, _ = self._f5tts.infer(
                ref_file=self._f5tts_ref_wav,
                ref_text=self._f5tts_ref_text,
                gen_text=req.text,
                speed=speed,
                show_info=lambda *a, **kw: None,  # suppress console output
            )
            # Apply pitch shift via resampling when non-zero
            if abs(req.pitch_shift) > 0.01:
                _factor = 2.0 ** (req.pitch_shift * 0.5)
                sr = int(sr * _factor)
            sd.play(wav, sr)
            sd.wait()
        except Exception as exc:
            self._startup_errors.append(f"f5tts runtime failure: {type(exc).__name__}")
            # Degrade gracefully to pyttsx3 if inference fails
            if self._ensure_pyttsx3_engine():
                self._backend = "pyttsx3"
                self._speak_pyttsx3(req)

    def _speak_kokoro(self, req: _SpeechRequest) -> None:
        """Synthesise with Kokoro-ONNX and play via sounddevice."""
        import sounddevice as sd

        speed = max(0.5, min(2.0, req.rate / self.DEFAULT_RATE * KOKORO_SPEED))
        audio, sr = self._kokoro.create(
            req.text,
            voice=KOKORO_VOICE,
            speed=speed,
            lang=KOKORO_LANG,
        )
        # Apply pitch shift via resampling when non-zero
        if abs(req.pitch_shift) > 0.01:
            _factor = 2.0 ** (req.pitch_shift * 0.5)  # semitone-scale shift
            _new_sr = int(sr * _factor)
            sd.play(audio, _new_sr)
        else:
            sd.play(audio, sr)
        sd.wait()

    def _speak_pyttsx3(self, req: _SpeechRequest) -> None:
        # Apply pitch_shift via rate modulation (pyttsx3 has no pitch API)
        _pitch_rate_mod = 1.0 + req.pitch_shift * 0.15
        _adj_rate = int(req.rate * _pitch_rate_mod)
        self._pyttsx3_engine.setProperty("rate", max(80, min(350, _adj_rate)))
        self._pyttsx3_engine.setProperty("volume", req.volume)
        # Emphasis via SSML-like pauses around emphasis words (if engine supports it)
        _text = req.text
        for ew in req.emphasis_words[:5]:
            _text = _text.replace(ew, f", {ew},", 1)
        self._pyttsx3_engine.say(_text)
        self._pyttsx3_engine.runAndWait()
