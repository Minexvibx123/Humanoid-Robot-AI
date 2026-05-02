"""
Test F5-TTS zero-shot voice cloning with albedo_voice.wav.
First run downloads ~300MB model to HuggingFace cache.

Avoids auto-transcription by requiring a reference transcript via CLI,
environment variable, or sidecar text file.
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import sys


def _load_reference_text(ref_wav: str, cli_text: str) -> str:
    if cli_text.strip():
        return cli_text.strip()

    env_text = os.environ.get("ALBEDO_VOICE_TEXT", "").strip()
    if env_text:
        return env_text

    sidecar_path = os.path.splitext(ref_wav)[0] + ".txt"
    if os.path.exists(sidecar_path):
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            sidecar_text = handle.read().strip()
        if sidecar_text:
            return sidecar_text

    print("FEHLER: Kein Referenztext gefunden.")
    print("Lege eine Datei 'albedo_voice.txt' neben die WAV, setze ALBEDO_VOICE_TEXT,")
    print(
        "oder starte mit --ref-text, damit F5-TTS keine Auto-Transkription via torchcodec ausloest."
    )
    sys.exit(2)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ref-text", default="", help="Exakter Transkriptions-Text der Referenzaufnahme"
)
args = parser.parse_args()

ref_wav = os.path.join(os.path.dirname(__file__), "albedo_voice.wav")
if not os.path.exists(ref_wav):
    print(f"FEHLER: {ref_wav} nicht gefunden.")
    sys.exit(1)

ref_text = _load_reference_text(ref_wav, args.ref_text)

print("Lade F5-TTS Modell (erster Start = Download ~300MB)...")
try:
    from f5_tts.api import F5TTS
except ImportError as e:
    print(f"FEHLER: f5-tts nicht installiert: {e}")
    sys.exit(1)

tts = F5TTS()
print("Modell geladen.")

text = "Ich bin Albedo, die Wächterin des Grabes von Nazarick. Niemand betritt diesen heiligen Ort ohne meine Erlaubnis."
print(f"Synthetisiere: {text!r}")

wav, sr, _ = tts.infer(
    ref_file=ref_wav,
    ref_text=ref_text,
    gen_text=text,
    show_info=print,
)

print(f"Audio: {sr}Hz, {len(wav)} samples ({len(wav)/sr:.1f}s)")

import sounddevice as sd

sd.play(wav, sr)
sd.wait()
print("Fertig.")
