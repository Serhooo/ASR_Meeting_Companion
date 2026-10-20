"""Модуль 2: локальное распознавание речи (ASR Engine) на GigaAM.

Реализация соответствует PRD (п. 2.1, п. 2.2, Модуль 2):

* Выбор устройства: Авто / CPU / CUDA (torch.cuda.is_available()).
* Длинные записи разбиваются на чанки (по умолчанию 60 с) — защита от
  переполнения VRAM/RAM + возможность показывать прогресс.
* Таймкоды [MM:SS] строятся из позиции чанка (start).

Замечание: официальный ``transcribe_longform`` требует pyannote.audio и
gated-модель pyannote/segmentation-3.0 с HF_TOKEN — не подходит для
автономного portable-приложения, поэтому чанкинг делаем сами, а на каждом
чанке вызываем официальный ``model.transcribe`` (через тензорный путь без
ffmpeg, т.к. наш WAV уже 16 kHz моно).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import config

# Веса моделей хранятся ВНУТРИ проекта (models_gigaam/) — портативность:
# папка переносится на другой компьютер вместе с весами, без повторных
# скачиваний (PRD: автономность). gigaam.load_model(download_root=...)
# качает сюда только если файлов ещё нет.
GIGAAM_DIR = config.BASE_DIR / "models_gigaam"

# Короткие имена моделей GigaAM -> полные (PRD п. 2.2)
GIGAAM_ALIASES = {
    "ctc": "v3_ctc", "rnnt": "v3_rnnt",
    "e2e_ctc": "v3_e2e_ctc", "e2e_rnnt": "v3_e2e_rnnt",
}


def resolve_device(mode: str) -> str:
    """Разрешить режим устройства («Авто»/«CPU»/«CUDA») в строку для torch.

    PRD п. 2.1: «Авто» -> cuda при наличии, иначе cpu; «CPU» -> всегда cpu;
    «CUDA» -> ошибка, если GPU NVIDIA недоступен.
    """
    import torch  # локальный импорт: torch тяжёлый

    if mode == "CPU":
        return "cpu"
    if mode == "CUDA":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Выбран режим CUDA, но GPU NVIDIA не обнаружен. "
                "Переключите «Устройство ASR» на «Авто» или «CPU» в настройках."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class AsrEngine:
    """Ленивая загрузка GigaAM + распознавание WAV с чанкингом и прогрессом."""

    def __init__(self, model_name: str = "v3_e2e_ctc", device_mode: str = "Авто",
                 chunk_seconds: int = 60):
        self.model_name = GIGAAM_ALIASES.get(model_name, model_name)
        self.device_mode = device_mode
        self.chunk_seconds = max(10, int(chunk_seconds))
        self._model = None
        self._device: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, status_cb: Optional[Callable[[str], None]] = None) -> None:
        """Загрузить модель на выбранное устройство (идемпотентно).

        Raises
        ------
        RuntimeError  недоступно выбранное устройство / не установлен torch/gigaam
        """
        if self._model is not None:
            return
        try:
            import torch
            import gigaam
        except ImportError as exc:
            raise RuntimeError(
                f"Не установлены зависимости ASR ({exc}). Выполните: "
                "pip install -r requirements.txt"
            ) from exc

        device = resolve_device(self.device_mode)
        if status_cb:
            note = "первый запуск: веса качаются в models_gigaam/ (нужен интернет)" \
                if not self._cached() else f"устройство {device.upper()}"
            status_cb(f"Загрузка модели {self.model_name} ({note})…")
        try:
            self._model = gigaam.load_model(
                self.model_name, fp16_encoder=True, use_flash=False, device=device,
                download_root=str(GIGAAM_DIR),   # веса живут внутри проекта
            )
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось загрузить модель {self.model_name}: {exc}"
            ) from exc
        self._device = device

    def _cached(self) -> bool:
        """Веса модели уже лежат локально (models_gigaam/ внутри проекта)?"""
        try:
            return GIGAAM_DIR.is_dir() and any(
                GIGAAM_DIR.glob(f"*{self.model_name}*.ckpt"))
        except OSError:
            return False

    def unload(self) -> None:
        """Выгрузить модель из памяти (смена настроек, освобождение RAM/VRAM)."""
        if self._model is None:
            return
        self._model = None
        self._device = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------- Распознавание
    def transcribe_file(
        self,
        wav_path: Path,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[dict]:
        """Распознать 16 kHz моно WAV, вернуть сегменты с таймкодами.

        Parameters
        ----------
        wav_path : Path        путь к WAV (16 kHz mono int16)
        progress_cb : callable progress_cb(fraction0..1, status_text)
        cancel_event : Event   событие отмены (set() -> досрочный выход)

        Returns
        -------
        list of {"start": float, "end": float, "text": str}
        """
        import torch
        from scipy.io import wavfile

        self.load(status_cb=(lambda s: progress_cb(0.0, s)) if progress_cb else None)

        sr, data = wavfile.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = data.astype(np.float32) / 32768.0
        if sr != 16000:
            from scipy.signal import resample_poly
            div = np.gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // div, int(sr) // div).astype(np.float32)

        chunk_len = self.chunk_seconds * 16000
        chunks = [audio[i:i + chunk_len] for i in range(0, len(audio), chunk_len)]
        total = len(chunks)
        segments: list[dict] = []

        for idx, chunk in enumerate(chunks):
            if cancel_event is not None and cancel_event.is_set():
                break
            if progress_cb:
                progress_cb(
                    idx / total,
                    f"Распознавание: сегмент {idx + 1} из {total} "
                    f"(~{idx * self.chunk_seconds // 60}:{idx * self.chunk_seconds % 60:02d})",
                )
            text = self._transcribe_chunk(chunk)
            start = idx * self.chunk_seconds
            end = min((idx + 1) * self.chunk_seconds, len(audio) / 16000)
            if text.strip():
                segments.append({"start": float(start), "end": float(end),
                                 "text": text.strip()})

        if progress_cb:
            progress_cb(1.0, "Распознавание завершено")
        return segments

    def _transcribe_chunk(self, chunk: np.ndarray) -> str:
        """Распознать один чанк (float32 моно 16 kHz) тензорным путём.

        Использует те же шаги, что и официальный ``model.transcribe``
        (prepare_wav -> forward -> _decode), но без ffmpeg и без проверки
        длины — чанк уже под контролем приложения.
        """
        import torch

        model = self._model
        device = torch.device(self._device)
        wav = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        wav = wav.to(next(model.parameters()).dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=device)
        with torch.inference_mode():
            encoded, encoded_len = model.forward(wav, length)
            text, _words = model._decode(encoded, encoded_len, length, False)[0]
        return text

    # ------------------------------------------------------------- утилиты
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Секунды -> [MM:SS] (PRD: формат «[00:12] Текст реплики»)."""
        total = int(max(0.0, seconds))
        return f"[{total // 60:02d}:{total % 60:02d}]"

    @classmethod
    def segments_to_text(cls, segments: list[dict]) -> str:
        """Сегменты -> текст транскрипта с таймкодами (по строке на сегмент)."""
        return "\n".join(
            f"{cls.format_timestamp(s['start'])} {s['text']}" for s in segments
        )


def inject_pause_markers(segments: list[dict], pauses: list) -> list[dict]:
    """Вставить в сегменты транскрипта отметки перерывов (кнопка «Пауза»).

    Во время паузы аудио не пишется (шкала времени финального WAV сжимается),
    поэтому отметка ставится в точке resume_offset — тайм-код показывает,
    с какого момента записи продолжилась после перерыва.
    """
    if not pauses:
        return segments
    out = list(segments)
    for p in pauses:
        off = float(p.get("resume_offset", 0.0))
        dur = int(round(float(p.get("duration", 0.0))))
        if dur < 1:
            continue
        out.append({"start": off, "end": off,
                    "text": f"⏸ Перерыв (пауза {dur} с)"})
    out.sort(key=lambda s: s["start"])
    return out

