"""Модуль 1: двойной захват аудио (микрофон + системный звук) — Audio Capture Engine.

Стратегия захвата системного звука (PRD, Модуль 1), в порядке приоритета:

1. WASAPI loopback: InputStream на output-устройстве (PortAudio >= 19.7).
2. Устройство loopback-типа «Стерео микшер» / «Stereo Mix» / «Loopback»
   (проверено на целевой машине: Realtek «Стерео микшер» через WDM-KS,
   callback-режим).
3. Только микрофон — при отсутствии loopback приложение продолжает работу,
   о чём сообщается статусом (GUI).

Оба потока пишутся во временные WAV-файлы (потокобезопасно, без роста RAM
для многочасовых записей), затем микшируются и ресемплируются в 16 kHz моно.
"""
from __future__ import annotations

import queue
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample_poly

from . import config

TARGET_SR = 16000  # стандарт дискретизации GigaAM (SAMPLE_RATE = 16000)

# Ключевые слова устройств-loopback в разных локалях Windows
_LOOPBACK_KEYWORDS = (
    "stereo mix", "stereomix", "loopback", "what u hear",
    "стерео микшер", "микшер", "сведение",
)


# ----------------------------------------------------------------- Устройства
def find_wasapi_index() -> Optional[int]:
    """Индекс host API WASAPI (Windows); None если недоступен."""
    for idx, api in enumerate(sd.query_hostapis()):
        if "wasapi" in api["name"].lower():
            return idx
    return None


def _find_wasapi_loopback_device() -> Optional[int]:
    """[Стратегия 1] Output-устройство WASAPI для out-as-in loopback."""
    wasapi = find_wasapi_index()
    if wasapi is None:
        return None
    out_dev = sd.query_hostapis()[wasapi]["default_output_device"]
    return out_dev if out_dev >= 0 else None


def _find_named_loopback_device() -> Optional[int]:
    """[Стратегия 2] Входное устройство с loopback-именем (Стерео микшер)."""
    for dev in sd.query_devices():
        if dev["max_input_channels"] <= 0:
            continue
        name = dev["name"].lower()
        if any(key in name for key in _LOOPBACK_KEYWORDS):
            return dev["index"]
    return None


def find_loopback_device() -> Optional[int]:
    """Найти доступное устройство захвата системного звука (или None)."""
    for finder in (_find_wasapi_loopback_device, _find_named_loopback_device):
        dev = finder()
        if dev is not None:
            try:  # проверяем, что устройство реально открывается на запись
                sd.check_input_settings(device=dev, channels=1, dtype="float32")
                return dev
            except Exception:
                continue
    return None


def _default_mic_device() -> Optional[int]:
    """[Стратегия 3] Микрофон: WASAPI default input, иначе системный default."""
    wasapi = find_wasapi_index()
    if wasapi is not None:
        in_dev = sd.query_hostapis()[wasapi]["default_input_device"]
        if in_dev >= 0:
            return in_dev
    return sd.default.device[0]


def _extra_settings_for(dev_index: int):
    """WasapiSettings(auto_convert) — только для устройств host API WASAPI.

    Передача WasapiSettings в поток другого host API (например WDM-KS
    «Стерео микшер») вызывает PaErrorCode -9984 «Incompatible host API
    specific stream info».
    """
    wasapi = find_wasapi_index()
    if wasapi is None or dev_index is None or dev_index < 0:
        return None
    if sd.query_devices(dev_index)["hostapi"] == wasapi:
        return sd.WasapiSettings(auto_convert=True)
    return None



# --------------------------------------------------------------- Writer
class _StreamWriter:
    """Потокобезопасная запись блоков из sounddevice-колбэка в WAV-файл."""

    def __init__(self, path: Path, samplerate: int, channels: int):
        self.path = path
        self.samplerate = samplerate
        self.channels = channels
        self._q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._wave: Optional[wave.Wave_write] = None

    def start(self) -> None:
        self._wave = wave.open(str(self.path), "wb")
        self._wave.setnchannels(self.channels)
        self._wave.setsampwidth(2)  # int16
        self._wave.setframerate(self.samplerate)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def callback(self, indata, frames, time_info, status) -> None:
        """Колбэк звукового потока: только копия буфера в очередь."""
        self._q.put(indata.copy())

    def _run(self) -> None:
        while True:
            block = self._q.get()
            if block is None:
                break
            if self._wave is not None:
                self._wave.writeframesraw(
                    (np.clip(block, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                )
        if self._wave is not None:
            self._wave.close()

    def stop(self, timeout: float = 10.0) -> None:
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)


class MeetingRecorder:
    """Запись встречи: микрофон + системный звук -> meeting_cache.wav (16 kHz моно).

    Использование::

        rec = MeetingRecorder()
        rec.start()
        level = rec.get_level()   # для визуализатора волны
        path, info = rec.stop()   # info: dict с предупреждениями
    """

    def __init__(self, cache_wav: Path = config.MEETING_WAV):
        self.cache_wav = Path(cache_wav)
        self.is_recording = False
        self.is_paused = False            # запись на паузе (звук не пишется)
        self.pause_events: list[dict] = []  # снятые паузы: resume_offset, duration
        self.loopback_ok = False
        self.mic_ok = False
        self.warnings: list[str] = []
        self._mic_stream: Optional[sd.InputStream] = None
        self._loop_stream: Optional[sd.InputStream] = None
        self._mic_writer: Optional[_StreamWriter] = None
        self._loop_writer: Optional[_StreamWriter] = None
        self._callbacks: list = []      # ссылки, чтобы GC не убил колбэки
        self._level = 0.0
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._pause_started: Optional[float] = None

    # ------------------------------------------------------------ Свойства
    def _paused_total(self) -> float:
        """Суммарная длительность всех пауз (включая текущую)."""
        total = sum(p["duration"] for p in self.pause_events)
        if self.is_paused and self._pause_started is not None:
            total += time.time() - self._pause_started
        return total

    @property
    def duration(self) -> float:
        """Длительность ЗАПИСАННОГО аудио в секундах (паузы вычтены)."""
        if not self.is_recording:
            return 0.0
        return max(0.0, time.time() - self._t0 - self._paused_total())

    def get_level(self) -> float:
        """Уровень сигнала 0..1 (усиленный RMS) — для визуализатора волны."""
        with self._lock:
            return self._level

    def _make_callback(self, writer: _StreamWriter):
        """Колбэк потока: пишет блок в файл + обновляет уровень волны.

        Во время паузы блоки отбрасываются — звук не пишется, файлы не
        растут, шкала времени финального WAV сжимается (перерыв отражается
        тайм-кодом в транскрипте, см. asr_engine.inject_pause_markers).
        """
        def _cb(indata, frames, time_info, status) -> None:
            if self.is_paused:
                with self._lock:
                    self._level = 0.0
                return
            writer.callback(indata, frames, time_info, status)
            with self._lock:
                # x3 — усиление, чтобы тихая речь была заметна на визуализаторе
                self._level = min(1.0, float(np.sqrt(np.mean(indata ** 2))) * 3.0)
        self._callbacks.append(_cb)
        return _cb

    # --------------------------------------------------------------- Пауза
    def pause(self) -> None:
        """Поставить запись на паузу (звук не пишется, таймер замирает)."""
        if not self.is_recording or self.is_paused:
            return
        self.is_paused = True
        self._pause_started = time.time()

    def resume(self) -> None:
        """Снять паузу; зафиксировать перерыв с точкой resume_offset.

        resume_offset — позиция во финальном WAV (в шкале записанного
        аудио), где продолжится запись; именно туда встанет тайм-код
        перерыва в транскрипте.
        """
        if not self.is_recording or not self.is_paused:
            return
        pause_dur = time.time() - (self._pause_started or time.time())
        resume_offset = self.duration   # записано до паузы (пауза ещё в счётчиках)
        self.pause_events.append({
            "resume_offset": round(resume_offset, 3),
            "duration": round(pause_dur, 3),
        })
        self.is_paused = False
        self._pause_started = None

    # ------------------------------------------------------------- Управление
    def start(self) -> None:
        """Открыть потоки микрофона и loopback, начать запись."""
        if self.is_recording:
            return
        config.ensure_dirs()
        self.warnings = []
        self.is_paused = False
        self.pause_events = []
        self._pause_started = None

        mic_dev = _default_mic_device()
        loop_dev = find_loopback_device()

        # ---- Поток микрофона
        try:
            mic_info = sd.query_devices(mic_dev)
            mic_sr = int(mic_info["default_samplerate"])
            mic_ch = min(mic_info["max_input_channels"], 2)
            self._mic_writer = _StreamWriter(config.MIC_RAW_WAV, mic_sr, mic_ch)
            self._mic_writer.start()
            self._mic_stream = sd.InputStream(
                device=mic_dev, samplerate=mic_sr, channels=mic_ch,
                dtype="float32", callback=self._make_callback(self._mic_writer),
                extra_settings=_extra_settings_for(mic_dev),
            )
            self._mic_stream.start()
            self.mic_ok = True
        except Exception as exc:
            self.warnings.append(f"Микрофон недоступен: {exc}")
            self.mic_ok = False

        # ---- Поток системного звука (loopback)
        if loop_dev is not None:
            try:
                loop_info = sd.query_devices(loop_dev)
                loop_sr = int(loop_info["default_samplerate"])
                loop_ch = min(max(loop_info["max_input_channels"], 1), 2)
                self._loop_writer = _StreamWriter(config.LOOP_RAW_WAV, loop_sr, loop_ch)
                self._loop_writer.start()
                self._loop_stream = sd.InputStream(
                    device=loop_dev, samplerate=loop_sr, channels=loop_ch,
                    dtype="float32", callback=self._make_callback(self._loop_writer),
                    extra_settings=_extra_settings_for(loop_dev),
                )
                self._loop_stream.start()
                self.loopback_ok = True
            except Exception as exc:
                self.warnings.append(f"Системный звук недоступен: {exc}")
                self.loopback_ok = False
        else:
            self.warnings.append(
                "Устройство loopback не найдено — записывается только микрофон."
            )
            self.loopback_ok = False

        if not self.mic_ok and not self.loopback_ok:
            raise RuntimeError(
                "Не удалось открыть ни микрофон, ни системный звук: "
                + "; ".join(self.warnings)
            )

        self.is_recording = True
        self._t0 = time.time()

    def stop(self) -> "tuple[Path, dict]":
        """Остановить запись, смикшировать потоки и сохранить 16 kHz моно WAV.

        Returns
        -------
        (путь к meeting_cache.wav, info-словарь с параметрами записи)
        """
        if not self.is_recording:
            raise RuntimeError("Запись не запущена")
        self.is_recording = False
        self.is_paused = False          # незавершённая пауза не фиксируется
        self._pause_started = None
        duration = time.time() - self._t0 - self._paused_total()

        for stream in (self._mic_stream, self._loop_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._mic_stream = self._loop_stream = None

        for writer in (self._mic_writer, self._loop_writer):
            if writer is not None:
                writer.stop()
        self._mic_writer = self._loop_writer = None

        mixed = self._mix_and_resample()
        wavfile.write(self.cache_wav, TARGET_SR, mixed)

        for temp in (config.MIC_RAW_WAV, config.LOOP_RAW_WAV):
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

        info = {
            "duration": duration,
            "mic": self.mic_ok,
            "loopback": self.loopback_ok,
            "warnings": list(self.warnings),
            "pauses": list(self.pause_events),   # для тайм-кодов перерывов
        }
        return self.cache_wav, info

    # ------------------------------------------------------------ Микшер
    @staticmethod
    def _load_mono(path: Path):
        """Прочитать WAV как float32 моно (усреднением каналов); None при ошибке."""
        try:
            sr, data = wavfile.read(path)
        except (OSError, ValueError):
            return None
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = data.astype(np.float32) / 32768.0
        return audio, int(sr)

    def _mix_and_resample(self) -> np.ndarray:
        """Микшировать потоки, ресемплировать в 16 kHz моно, вернуть int16."""
        streams: list = []
        for raw in (config.MIC_RAW_WAV, config.LOOP_RAW_WAV):
            loaded = self._load_mono(raw)
            if loaded is not None:
                streams.append(loaded)
        if not streams:
            return np.zeros(TARGET_SR, dtype="<i2")

        # Ресемплинг каждого потока к 16 kHz
        resampled = []
        for audio, sr in streams:
            if sr != TARGET_SR:
                divisor = np.gcd(int(sr), TARGET_SR)
                audio = resample_poly(audio, TARGET_SR // divisor, int(sr) // divisor)
            resampled.append(np.asarray(audio, dtype=np.float32))

        # Выравнивание длительностей нулями
        max_len = max(len(a) for a in resampled)
        resampled = [
            np.pad(a, (0, max_len - len(a))) if len(a) < max_len else a
            for a in resampled
        ]

        # Микш: среднее с клиппером
        mixed = np.mean(resampled, axis=0) if len(resampled) > 1 else resampled[0]
        mixed = np.clip(mixed, -1.0, 1.0)
        return (mixed * 32767).astype("<i2")



