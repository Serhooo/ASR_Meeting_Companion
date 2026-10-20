"""Реалтайм-транскрибация (near-real-time) на том же движке GigaAM.

Идея (проверено на целевой машине: инференс ~в 5–10 раз быстрее реального
времени на i5-11320H): пока идёт запись, отдельный поток читает сырые
потоки микрофона/loopback (которые дописываются MeetingRecorder), склеивает
их в скользящий 16 kHz буфер, режет на реплики по энергии (порог RMS +
пауза-тишина) и транскрибирует каждую реплику тензорным путём GigaAM.
Текст приходит в GUI с задержкой ~2–4 с — «живой» режим.

По остановке записи основной пайплайн всё равно запускается и строит
качественный транскрипт заново (полный контекст) + протокол.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.signal import resample_poly

from . import config
from .asr_engine import AsrEngine

POLL_S = 0.3            # период опроса сырых файлов, сек
SILENCE_END_S = 0.5     # столько тишины завершает реплику (текст сразу после паузы)
MIN_SEG_S = 0.4         # короче не транскрибируем (шум)
MAX_SEG_S = 6.0         # принудительный срез при непрерывной речи (текст каждые ~6 с)
BUF_CAP_S = 30.0        # ограничение RAM: в буфере не больше этого

# ---- Детектор речи: пик-RMS по кадрам 50 мс + адаптивный порог от фона
FRAME = 800             # 50 мс @ 16 kHz
NOISE_EMA_START = 0.002 # стартовая оценка шумового пола
T_ON_FLOOR = 0.0035     # минимальный порог включения (абсолютный)
T_ON_FACTOR = 1.5       # порог = max(floor, шумовой_фон * 1.5)
T_OFF_FACTOR = 0.6      # гистерезис: выключение в 0.6 от включения


def _peak_frame_rms(x: np.ndarray) -> float:
    """Максимум RMS по кадрам 50 мс — устойчиво находит пики речи.

    Средний RMS по длинному окну «размывается» паузами внутри фразы и
    не пересекает порог (причина, по которой текст не появлялся).
    """
    if x.size == 0:
        return 0.0
    nf = x.size // FRAME
    if nf == 0:
        return float(np.sqrt(np.mean(x ** 2)))
    frames = x[:nf * FRAME].reshape(nf, FRAME)
    return float(np.sqrt(np.mean(frames ** 2, axis=1)).max())


class _RawTailReader:
    """Читает новые сэмплы из WAV, который дописывается другим потоком.

    Файлы создаёт MeetingRecorder._StreamWriter (wave, int16, заголовок
    44 байта пишется сразу) — можно безопасно читать «хвост» от offset.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 44
        self.samplerate = 0
        self.channels = 0
        with open(self.path, "rb") as f:
            header = f.read(44)
        if len(header) < 44 or header[:4] != b"RIFF":
            raise ValueError(f"Не WAV: {self.path}")
        self.channels = int.from_bytes(header[22:24], "little")
        self.samplerate = int.from_bytes(header[24:28], "little")

    def read_new_int16(self) -> np.ndarray:
        """Новые сэмплы с последнего чтения (плоский int16)."""
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                raw = f.read()
        except OSError:
            return np.zeros(0, dtype="<i2")
        if not raw:
            return np.zeros(0, dtype="<i2")
        usable = len(raw) - (len(raw) % (2 * max(1, self.channels)))
        if usable <= 0:
            return np.zeros(0, dtype="<i2")
        self.offset += usable
        data = np.frombuffer(raw[:usable], dtype="<i2")
        if self.channels > 1:
            data = data.reshape(-1, self.channels).mean(axis=1)
        return data.astype(np.float32) / 32768.0


class LiveTranscriber:
    """Поток реалтайм-распознавания поверх идущей записи.

    Parameters
    ----------
    asr : AsrEngine            уже созданный движок (загрузится лениво)
    on_segment : callable      on_segment(start_sec, text) — готовая реплика
    on_status : callable       on_status(text) — статус/ошибки для GUI
    """

    def __init__(self, asr: AsrEngine,
                 on_segment: Callable[[float, str], None],
                 on_status: Optional[Callable[[str], None]] = None):
        self.asr = asr
        self.on_segment = on_segment
        self.on_status = on_status or (lambda _t: None)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- Управление
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------- Воркер
    def _run(self) -> None:
        try:
            self.asr.load(status_cb=lambda s: self.on_status(f"Реалтайм: {s}"))
        except Exception as exc:
            self.on_status(f"Реалтайм недоступен: {exc}")
            return

        # Файлы создаются StreamWriter в момент rec.start(); даём им время
        # дописать первые сэмплы (с прогруженной моделью раньше проверка
        # успевала пройти до первого аудиоблока -> «нет потоков»).
        readers = []
        deadline = time.time() + 5.0
        while time.time() < deadline and not self._stop.is_set():
            readers = []
            for path in (config.MIC_RAW_WAV, config.LOOP_RAW_WAV):
                try:
                    if path.is_file() and path.stat().st_size > 44:
                        readers.append(_RawTailReader(path))
                except (OSError, ValueError):
                    continue
            if readers:
                break
            time.sleep(0.2)
        if not readers:
            self.on_status("Реалтайм: нет открытых аудиопотоков")
            return

        buf = np.zeros(0, dtype=np.float32)   # скользящий 16 kHz буфер
        pos = 0                               # абсолютная позиция конца буфера (сэмплы)
        speaking = False
        seg_start = 0
        silence = 0
        noise_ema = NOISE_EMA_START           # адаптивная оценка шумового фона
        self.on_status("Реалтайм: текст сразу после паузы, "
                       "при непрерывной речи — каждые ~6 с")

        while not self._stop.is_set():
            time.sleep(POLL_S)
            deltas = []
            for rd in readers:
                chunk = rd.read_new_int16()
                if rd.samplerate != 16000 and len(chunk):
                    div = np.gcd(int(rd.samplerate), 16000)
                    chunk = resample_poly(chunk, 16000 // div,
                                          int(rd.samplerate) // div)
                deltas.append(np.asarray(chunk, dtype=np.float32))
            if not any(len(d) for d in deltas):
                continue
            n = max(len(d) for d in deltas)
            deltas = [np.pad(d, (0, n - len(d))) for d in deltas]
            # Для записи — среднее (антиклиппинг); для детектора — МАКСИМУМ
            # по потокам: речь обычно в одном потоке, mean гасит пики вдвое.
            peaks = [_peak_frame_rms(d) for d in deltas]
            delta = np.mean(deltas, axis=0) if len(deltas) > 1 else deltas[0]
            buf = np.concatenate([buf, delta])
            pos += n

            # ---- Детектор речи (пик-RMS по кадрам + адаптивный порог)
            peak = max(peaks) if peaks else 0.0
            t_on = max(T_ON_FLOOR, noise_ema * T_ON_FACTOR)
            if not speaking:
                noise_ema = 0.95 * noise_ema + 0.05 * peak
                if peak >= t_on:
                    speaking = True
                    seg_start = pos - n   # реплика начинается от текущего чтения
                    silence = 0
            else:
                t_off = t_on * T_OFF_FACTOR
                if peak >= t_off:
                    silence = 0
                else:
                    silence += n

            # ---- Срезы реплики: пауза ИЛИ макс. длительность
            if speaking:
                ended_by_silence = silence / 16000.0 >= SILENCE_END_S
                forced_cut = (pos - seg_start) / 16000.0 >= MAX_SEG_S
                if ended_by_silence or forced_cut:
                    seg_end = pos - silence
                    self._emit_segment(buf, pos, seg_start, seg_end)
                    if ended_by_silence:
                        speaking = False        # реплика закрыта по паузе
                    # при принудительном срезе реплика продолжается с seg_end
                    seg_start = seg_end
                    silence = 0

            if not speaking:
                # держим в буфере только хвост — RAM ограничен
                keep = int(BUF_CAP_S * 16000)
                if len(buf) > keep:
                    buf = buf[-keep:]

        # финальная незавершённая реплика
        if speaking and (pos - seg_start) / 16000.0 >= MIN_SEG_S:
            self._emit_segment(buf, pos, seg_start, pos - silence)
        self.on_status("Реалтайм-режим остановлен")

    def _emit_segment(self, buf: np.ndarray, pos: int,
                      seg_start: int, seg_end: int) -> None:
        """Транскрибировать реплику [seg_start, seg_end) из буфера."""
        if (seg_end - seg_start) / 16000.0 < MIN_SEG_S:
            return
        # позиции буфера: buf содержит последние len(buf) сэмплов до pos
        buf_start = pos - len(buf)
        i0 = max(seg_start - buf_start, 0)
        i1 = max(seg_end - buf_start, i0 + 1)
        if i1 - i0 < int(MIN_SEG_S * 16000):
            return
        seg = np.ascontiguousarray(buf[i0:i1])
        try:
            text = self.asr._transcribe_chunk(seg).strip()
        except Exception as exc:
            self.on_status(f"Реалтайм: ошибка распознавания — {exc}")
            return
        if text:
            self.on_segment(seg_start / 16000.0, text)
