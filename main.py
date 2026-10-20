"""Meeting Companion — точка входа.

Локальный ИИ-секретарь встреч: запись (микрофон + системный звук) ->
распознавание речи (GigaAM, CPU/CUDA на выбор) -> протокол (Ollama) -> .md.

Запуск:  python main.py
"""
from __future__ import annotations

import os
import sys


def _enforce_offline() -> None:
    """Закрепить автономность на уровне окружения (PRD, п. 4.1).

    Запрещаем ленивым импортам (huggingface_hub, transformers, pyannote)
    ходить в сеть из рантайма: веса GigaAM кэшируются локально, а любой
    сценарий, требующий скачивания моделей HF, честно упадёт с ошибкой
    вместо тихого сетевого запроса. Для транскрипции это безопасно:
    наш пайплайн использует только локальный кэш models_gigaam/ (в проекте).
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    # Телеметрия torch — отключена (не влияет на инференс)
    os.environ.setdefault("TORCH_INDUCER_FX_GRAPH_REMOTE_CACHE", "0")


def _enable_dpi_awareness() -> None:
    """Корректное отображение на HiDPI-экранах Windows."""
    try:
        import ctypes
        # Per-Monitor V2 (Windows 10 1703+)
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
    except Exception:
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def main() -> int:
    _enforce_offline()
    _enable_dpi_awareness()

    # Портативный (embeddable) Python задаёт sys.path файлом ._pth и НЕ
    # добавляет каталог скрипта автоматически — прописываем его явно,
    # чтобы пакет `app` находился при любом способе запуска.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    # Тяжёлые импорты (tkinter/torch) — после настройки DPI
    from app.gui.main_window import App

    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
