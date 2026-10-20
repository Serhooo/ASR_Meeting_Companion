"""Портативная Ollama внутри проекта (PRD: автономность + портативность).

Схема: официальный бандл `tools/ollama-windows-amd64.zip` распакован в
`ollama/` (проект), модели хранятся в `ollama_models/` (проект) через
переменную OLLAMA_MODELS. Сервер запускается как дочерний процесс приложения
(127.0.0.1:11434) и останавливается вместе с ним — никакой системной
установки, всё живёт в папке проекта и переносится вместе с ним.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import config

OLLAMA_DIR = config.BASE_DIR / "ollama"
OLLAMA_EXE = OLLAMA_DIR / "ollama.exe"
MODELS_DIR = config.BASE_DIR / "ollama_models"
LOG_PATH = config.CACHE_DIR / "ollama_serve.log"
HOST = "127.0.0.1:11434"

CREATE_NO_WINDOW = 0x08000000  # флаг Windows: не создавать консольное окно

_PROC: Optional[subprocess.Popen] = None   # сервер, запущенный НАМИ (дочерний)


def stop_server() -> None:
    """Остановить сервер, запущенный этим приложением (портативность).

    Системный Ollama (если запущен отдельно) не трогается: наш экземпляр
    отслеживается в _PROC.
    """
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.kill()
        except OSError:
            pass
    _PROC = None


def _serve_env() -> dict:
    """Окружение сервера: модели строго внутри папки проекта."""
    env = dict(os.environ)
    env["OLLAMA_MODELS"] = str(MODELS_DIR)
    env["OLLAMA_HOST"] = HOST
    return env


def is_installed() -> bool:
    """Портативный бандл Ollama распакован в проект?"""
    return OLLAMA_EXE.is_file()


def is_server_up(timeout: float = 1.5) -> bool:
    """Отвечает ли локальный сервер Ollama (127.0.0.1:11434)."""
    try:
        with urllib.request.urlopen(f"http://{HOST}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def start_server(wait: float = 20.0, status_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Запустить портативный ollama serve и дождаться готовности.

    Если сервер уже запущен (например, системный) — просто вернёт True.
    """
    if is_server_up():
        return True
    if not is_installed():
        return False
    config.ensure_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    global _PROC
    try:
        log = open(LOG_PATH, "ab", buffering=0)
        _PROC = subprocess.Popen(
            [str(OLLAMA_EXE), "serve"],
            env=_serve_env(), cwd=str(OLLAMA_DIR),
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError as exc:
        if status_cb:
            status_cb(f"Не удалось запустить Ollama: {exc}")
        return False

    deadline = time.time() + wait
    while time.time() < deadline:
        if is_server_up():
            return True
        time.sleep(0.5)
    return is_server_up()


def pull_model(model: str, status_cb: Optional[Callable[[str], None]] = None,
               timeout: float = 3600) -> bool:
    """Скачать модель в локальный ollama_models/ (однократно, ~2 ГБ).

    Прогресс выводится построчно в status_cb из stderr процесса `ollama pull`.
    """
    if not is_installed():
        if status_cb:
            status_cb("Портативная Ollama не распакована в проект (tools/ollama…zip)")
        return False
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(OLLAMA_EXE), "pull", model],
        env=_serve_env(), cwd=str(OLLAMA_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=CREATE_NO_WINDOW,
    )
    assert proc.stdout is not None
    start = time.time()
    for line in proc.stdout:
        if time.time() - start > timeout:
            proc.kill()
            if status_cb:
                status_cb("Таймаут скачивания модели")
            return False
        clean = line.strip().replace("\r", "")
        # Windows-консоль (cp1251) не переживает спиннеры ⠋/блоки — вычищаем
        clean = clean.encode("ascii", "replace").decode("ascii")
        if clean and status_cb:
            try:
                status_cb(f"Скачивание {model}: {clean[:70]}")
            except Exception:
                pass  # получатель статуса мог закрыться
    proc.wait()
    return proc.returncode == 0


def extract_bundle(zip_path: Optional[Path] = None,
                   status_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Распаковать скачанный бандл в ollama/ (однократно)."""
    import zipfile
    zip_path = zip_path or (config.BASE_DIR / "tools" / "ollama-windows-amd64.zip")
    if is_installed():
        return True
    if not zip_path.is_file():
        return False
    if status_cb:
        status_cb("Распаковка портативной Ollama в проект…")
    OLLAMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(OLLAMA_DIR)
    except (OSError, zipfile.BadZipFile) as exc:
        if status_cb:
            status_cb(f"Ошибка распаковки Ollama: {exc}")
        return False
    return is_installed()
