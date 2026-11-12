"""Servicio de Text-to-Speech: convierte texto en un MP3.

El motor actual es Kokoro-82M sobre ONNX Runtime (CPU), porque los modelos de
voz de Ollama (Phonon) aún requieren MLX y no funcionan en Windows. El resto
del servicio solo depende de la interfaz `TtsEngine`, así que añadir otro
motor no afecta al pipeline.
"""
import asyncio
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import lameenc
import numpy as np
from sqlalchemy import select

from app.config import settings
from app.services.tts_text import split_paragraphs

logger = logging.getLogger(__name__)

_PAUSE_SENTENCE = 0.15  # segundos de silencio entre fragmentos de un mismo párrafo
_PAUSE_PARAGRAPH = 0.55  # segundos de silencio entre párrafos

# Prefijo de la voz de Kokoro → (código espeak, nombre del idioma)
_KOKORO_LANGS: dict[str, tuple[str, str]] = {
    "e": ("es", "Español"),
    "a": ("en-us", "Inglés (EE. UU.)"),
    "b": ("en-gb", "Inglés (Reino Unido)"),
    "f": ("fr-fr", "Francés"),
    "i": ("it", "Italiano"),
    "p": ("pt-br", "Portugués"),
}


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    language: str
    gender: str


@dataclass(frozen=True)
class SpeechUnit:
    """Fragmento que el motor sintetiza de una vez, y el silencio que le sigue."""

    payload: str
    pause_after: float


class TtsEngine(Protocol):
    sample_rate: int

    def voices(self) -> list[Voice]: ...

    def plan(self, paragraphs: list[list[str]], voice: str) -> list[SpeechUnit]: ...

    def synthesize(self, unit: SpeechUnit, voice: str, speed: float) -> np.ndarray: ...


# ─── Motor Kokoro (ONNX, CPU) ────────────────────────────────────────

class KokoroEngine:
    sample_rate = 24000
    _MAX_PHONEMES = 480  # el modelo admite 510 tokens; se deja margen

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session = None
        self._speed_dtype = np.float32
        self._tokenizer = None
        self._voices = None
        self._backends: dict[str, object] = {}

    # -- carga perezosa --

    def _download(self, filename: str) -> Path:
        dest = settings.tts_models_dir / filename
        if dest.exists():
            return dest
        settings.tts_models_dir.mkdir(parents=True, exist_ok=True)
        url = f"{settings.tts_kokoro_base_url}/{filename}"
        logger.info("Descargando modelo TTS %s", url)
        partial = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                for chunk in response.iter_bytes(1 << 20):
                    f.write(chunk)
        partial.replace(dest)
        return dest

    def _load_voices(self):
        with self._lock:
            if self._voices is None:
                self._voices = np.load(self._download(settings.tts_kokoro_voices_file))
            return self._voices

    def _load_model(self) -> None:
        self._load_voices()
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from kokoro_onnx.tokenizer import Tokenizer

            model_path = self._download(settings.tts_kokoro_model_file)
            # Tokenizer configura además la librería espeak-ng empaquetada
            self._tokenizer = Tokenizer()
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            inputs = {i.name: i.type for i in session.get_inputs()}
            # El tipo de `speed` cambia entre exportaciones del modelo
            self._speed_dtype = np.int32 if "int32" in inputs["speed"] else np.float32
            self._session = session

    # -- interfaz --

    def voices(self) -> list[Voice]:
        result = []
        for voice_id in sorted(self._load_voices().files):
            lang = _KOKORO_LANGS.get(voice_id[0])
            if not lang or len(voice_id) < 4 or voice_id[2] != "_":
                continue
            result.append(
                Voice(
                    id=voice_id,
                    name=voice_id[3:].replace("_", " ").title(),
                    language=lang[1],
                    gender="female" if voice_id[1] == "f" else "male",
                )
            )
        # Español primero: es el caso de uso principal
        return sorted(result, key=lambda v: (v.language != "Español", v.language, v.name))

    def _check_voice(self, voice: str) -> str:
        if voice not in self._load_voices().files or voice[0] not in _KOKORO_LANGS:
            raise ValueError(f"Voz no disponible: {voice}")
        return _KOKORO_LANGS[voice[0]][0]

    def _phonemize(self, sentences: list[str], lang: str) -> list[str]:
        backend = self._backends.get(lang)
        if backend is None:
            from phonemizer.backend import EspeakBackend

            backend = EspeakBackend(lang, preserve_punctuation=True, with_stress=True)
            self._backends[lang] = backend
        vocab = self._tokenizer.vocab
        return [
            "".join(ch for ch in phonemes if ch in vocab).strip()
            for phonemes in backend.phonemize(sentences)
        ]

    def _split_long(self, phonemes: str) -> list[str]:
        """Trocea una frase que no cabe en el modelo, preferiblemente por comas."""
        if len(phonemes) <= self._MAX_PHONEMES:
            return [phonemes]
        pieces: list[str] = []
        current = ""
        for part in re.split(r"(?<=[,;:])\s+", phonemes):
            while len(part) > self._MAX_PHONEMES:  # sin puntuación: cortar por palabras
                cut = part.rfind(" ", 0, self._MAX_PHONEMES)
                cut = cut if cut > 0 else self._MAX_PHONEMES
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(part[:cut])
                part = part[cut:].strip()
            if len(current) + len(part) + 1 > self._MAX_PHONEMES:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
        if current:
            pieces.append(current)
        return pieces

    def plan(self, paragraphs: list[list[str]], voice: str) -> list[SpeechUnit]:
        lang = self._check_voice(voice)
        self._load_model()
        units: list[SpeechUnit] = []
        for sentences in paragraphs:
            batches: list[str] = []
            current = ""
            for phonemes in self._phonemize(sentences, lang):
                for piece in self._split_long(phonemes):
                    if not piece:
                        continue
                    if current and len(current) + len(piece) + 1 > self._MAX_PHONEMES:
                        batches.append(current)
                        current = piece
                    else:
                        current = f"{current} {piece}".strip()
            if current:
                batches.append(current)
            for index, batch in enumerate(batches):
                last = index == len(batches) - 1
                units.append(SpeechUnit(batch, _PAUSE_PARAGRAPH if last else _PAUSE_SENTENCE))
        return units

    def synthesize(self, unit: SpeechUnit, voice: str, speed: float) -> np.ndarray:
        self._load_model()
        tokens = [i for i in map(self._tokenizer.vocab.get, unit.payload) if i is not None]
        if not tokens:
            return np.zeros(0, dtype=np.float32)
        tokens = tokens[:510]
        style = self._voices[voice][len(tokens)]
        audio = self._session.run(
            None,
            {
                "input_ids": np.array([[0, *tokens, 0]], dtype=np.int64),
                "style": np.asarray(style, dtype=np.float32),
                "speed": np.array([speed], dtype=self._speed_dtype),
            },
        )[0]
        return _trim_silence(np.asarray(audio, dtype=np.float32).reshape(-1), self.sample_rate)


def _trim_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.01) -> np.ndarray:
    """Recorta el silencio inicial y final para controlar las pausas explícitamente."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio[:0]
    margin = int(sample_rate * 0.03)
    return audio[max(loud[0] - margin, 0) : loud[-1] + margin]


_engine: TtsEngine | None = None


def get_engine() -> TtsEngine:
    global _engine
    if _engine is None:
        if settings.tts_provider != "kokoro":
            raise RuntimeError(f"Proveedor TTS no soportado: {settings.tts_provider}")
        _engine = KokoroEngine()
    return _engine


# ─── Generación del MP3 ──────────────────────────────────────────────

class _Mp3Writer:
    def __init__(self, path: Path, sample_rate: int) -> None:
        self._encoder = lameenc.Encoder()
        self._encoder.set_bit_rate(settings.tts_mp3_bitrate)
        self._encoder.set_in_sample_rate(sample_rate)
        self._encoder.set_channels(1)
        self._encoder.set_quality(2)
        self._file = path.open("wb")
        self._sample_rate = sample_rate
        self.samples = 0

    def write(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        self._file.write(bytes(self._encoder.encode(pcm.tobytes())))
        self.samples += pcm.size

    def silence(self, seconds: float) -> None:
        self.write(np.zeros(int(self._sample_rate * seconds), dtype=np.float32))

    def close(self, finalize: bool = True) -> None:
        if finalize:
            self._file.write(bytes(self._encoder.flush()))
        self._file.close()


# La síntesis satura la CPU: los trabajos se ejecutan de uno en uno
_job_lock = asyncio.Lock()


async def list_voices() -> list[Voice]:
    return await asyncio.to_thread(get_engine().voices)


async def run_job(audio_id: int) -> None:
    """Tarea en background: genera el MP3 de un TtsAudio y actualiza su registro."""
    from app.database import async_session
    from app.models.tts import TtsAudio

    async with _job_lock, async_session() as session:
        audio = await session.get(TtsAudio, audio_id)
        if not audio or audio.status != "pending":
            return

        settings.tts_audio_dir.mkdir(parents=True, exist_ok=True)
        final_path = settings.tts_audio_dir / f"{audio_id}.mp3"
        partial_path = final_path.with_suffix(".mp3.part")
        writer: _Mp3Writer | None = None
        try:
            audio.status = "processing"
            audio.progress = 0
            await session.commit()

            engine = get_engine()
            paragraphs = split_paragraphs(audio.text_content)
            units = await asyncio.to_thread(engine.plan, paragraphs, audio.voice)
            if not units:
                raise ValueError("El texto no contiene nada que leer.")

            writer = _Mp3Writer(partial_path, engine.sample_rate)
            for index, unit in enumerate(units):
                samples = await asyncio.to_thread(engine.synthesize, unit, audio.voice, audio.speed)
                writer.write(samples)
                writer.silence(unit.pause_after)

                progress = int((index + 1) * 100 / len(units))
                if progress != audio.progress:
                    # Si el usuario borró el audio a mitad, se abandona el trabajo
                    exists = await session.scalar(select(TtsAudio.id).where(TtsAudio.id == audio_id))
                    if exists is None:
                        writer.close(finalize=False)
                        partial_path.unlink(missing_ok=True)
                        return
                    audio.progress = min(progress, 99)
                    await session.commit()

            writer.close()
            partial_path.replace(final_path)
            audio.filepath = str(final_path)
            audio.size_bytes = final_path.stat().st_size
            audio.duration_seconds = round(writer.samples / engine.sample_rate, 1)
            audio.progress = 100
            audio.status = "done"
        except Exception as exc:
            logger.exception("TTS failed for audio %d", audio_id)
            if writer is not None:
                writer.close(finalize=False)
            partial_path.unlink(missing_ok=True)
            audio.status = "error"
            audio.error_message = str(exc) or exc.__class__.__name__
        await session.commit()
