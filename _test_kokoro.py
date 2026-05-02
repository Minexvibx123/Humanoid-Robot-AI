"""Quick test: generate a WAV with Kokoro-ONNX and play it."""

import os
import warnings

warnings.filterwarnings("ignore")

for l in open(".env"):
    l = l.strip()
    if "=" in l and not l.startswith("#"):
        k, _, v = l.partition("=")
        os.environ[k.strip()] = v.strip()

from speech_output import KOKORO_LANG, KOKORO_SPEED, KOKORO_VOICE, _kokoro_load

print("Lade Kokoro-ONNX Modell (erster Start: ~300 MB Download) ...")
kokoro = _kokoro_load()
if kokoro is None:
    raise SystemExit("Kokoro konnte nicht geladen werden.")
print(f"Modell geladen. Stimme: {KOKORO_VOICE}  Lang: {KOKORO_LANG}")

text = "Ich bin Albedo, die Hüterin der Großen Grabkammer von Nazarick. Wie kann ich Ainz-sama heute dienen?"
print(f"Synthetisiere: {text[:60]}...")
audio, sr = kokoro.create(
    text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=KOKORO_LANG
)

import soundfile as sf

out = "albedo_test.wav"
sf.write(out, audio, sr)
print(f"Gespeichert: {out}  ({len(audio)/sr:.1f}s)")

# Play it
import sounddevice as sd

print("Spiele ab ...")
sd.play(audio, sr)
sd.wait()
print("Fertig.")
