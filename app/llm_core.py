"""Модуль 3: интеллектуальный суммаризатор — интеграция с локальным Ollama.

Все запросы выполняются ТОЛЬКО к локальному серверу Ollama (http://localhost:11434).
Внешняя сеть не используется (PRD, п. 4.1). Реализация на stdlib (urllib) —
без дополнительных зависимостей.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

_CONNECT_TIMEOUT = 5      # сек: проверка доступности сервера
_READ_TIMEOUT = 1800      # сек: генерация ответа LLM (CPU может быть долгим)


def _build_user_prompt(transcript: str, topic: str = "") -> str:
    """Собрать сообщение пользователя: тема совещания + транскрипт.

    Тема задаёт LLM контекст (о чём совещание, участники, повестка) —
    протокол получается точнее даже на коротком/шумном транскрипте.
    """
    parts = []
    topic = (topic or "").strip()
    if topic:
        parts.append("Тема и повестка совещания:\n" + topic)
    parts.append("Транскрипт встречи:\n\n" + transcript)
    return "\n\n".join(parts)


def check_server(url: str = config.DEFAULTS["ollama_url"]) -> "tuple[bool, list[str]]":
    """Проверить доступность Ollama; вернуть (доступен, список_моделей)."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=_CONNECT_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "") for m in payload.get("models", [])]
        return True, models
    except Exception:
        return False, []


def summarize(
    transcript: str,
    model: str = config.DEFAULTS["ollama_model"],
    url: str = config.DEFAULTS["ollama_url"],
    max_chars: int = config.DEFAULTS["llm_max_chars"],
    status_cb=None,
    topic: str = "",
) -> str:
    """Отправить транскрипт в Ollama и получить протокол встречи (Markdown).

    Parameters
    ----------
    transcript : str        транскрипт с таймкодами
    model : str             имя модели Ollama (например "llama3.2:3b")
    url : str               адрес локального сервера Ollama
    max_chars : int         усечение транскрипта до лимита контекста
    status_cb : callable    callback(status_text) для отображения прогресса
    topic : str             тема/повестка совещания из вкладки «Тема» —
                            задаёт контекст секретарю при составлении протокола

    Raises
    ------
    RuntimeError  сервер недоступен или вернул ошибку
    """
    if not transcript.strip():
        raise RuntimeError("Транскрипт пуст — нечего суммировать.")

    if len(transcript) > max_chars:
        cut = transcript[:max_chars]
        # обрезаем по последнему предложению/строке, чтобы не рвать слово
        boundary = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        transcript = cut[: boundary + 1] if boundary > max_chars // 2 else cut
        if status_cb:
            status_cb("Транскрипт усечён до лимита контекста LLM…")

    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(transcript, topic)},
        ],
        "options": {"temperature": 0.2},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_READ_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama недоступна ({url}). Запустите сервер командой `ollama serve` "
            f"или установите Ollama (см. README). Детали: {exc.reason}"
        ) from exc

    message = payload.get("message", {})
    text = (message.get("content") or "").strip()
    if not text:
        raise RuntimeError(f"Ollama вернула пустой ответ (модель: {model}).")
    return text


def export_markdown(
    summary: str,
    transcript: str = "",
    exports_dir: Path = config.EXPORTS_DIR,
    topic: str = "",
) -> Path:
    """Сохранить протокол (и транскрипт) в .md; вернуть путь к файлу."""
    exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fname = datetime.now().strftime("protocol_%Y%m%d_%H%M%S.md")
    path = exports_dir / fname

    parts = [
        f"# Протокол встречи",
        f"_Meeting Companion · {stamp}_",
    ]
    if (topic or "").strip():
        parts += ["", "**Тема совещания:**", "", topic.strip()]
    parts += [
        "",
        summary.strip(),
    ]
    if transcript.strip():
        parts += [
            "",
            "---",
            "",
            "## Приложение: транскрипт встречи",
            "",
            transcript.strip(),
        ]
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def export_doc(
    summary: str,
    transcript: str = "",
    exports_dir: Path = config.EXPORTS_DIR,
    topic: str = "",
) -> Path:
    """Сохранить протокол в формат Word (.doc) без внешних библиотек.

    Используется HTML-совместимый MHTML: формат открывается Microsoft Word,
    LibreOffice Writer и WPS Office «двойным кликом» без установки python-docx.
    Три блока протокола (Решения/Договорённости/Следующие шаги) выделяются
    заголовками, транскрипт уходит в приложение.
    """
    from html import escape

    exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    path = exports_dir / datetime.now().strftime("protocol_%Y%m%d_%H%M%S.doc")

    body = [f"<h1>Протокол встречи</h1><p><i>Meeting Companion · {escape(stamp)}</i></p>"]
    if (topic or "").strip():
        body.append(f"<p><b>Тема совещания:</b><br>{escape(topic).strip().replace(chr(10), '<br>')}</p>")
    # Протокол: переносим как есть (Markdown-разметка Word проигнорирует,
    # но структура строк сохранится). Жирный ключевых слов не выделяем —
    # оставляем нативную читаемость.
    body.append("<hr>")
    for line in summary.strip().splitlines():
        if not line.strip():
            body.append("<br>")
        elif line.lstrip().startswith("#"):
            body.append(f"<h2>{escape(line.lstrip('#').strip())}</h2>")
        else:
            body.append(f"<p>{escape(line)}</p>")
    if transcript.strip():
        body.append("<hr><h2>Приложение: транскрипт встречи</h2>")
        body.append(
            f'<p style="font-family:Consolas,monospace;font-size:10pt">'
            f"{escape(transcript).strip().replace(chr(10), '<br>')}</p>"
        )

    html = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;font-size:11pt;"
        "line-height:1.45}h1{color:#1a3a2a}h2{color:#0b6b3a}</style></head><body>"
        + "".join(body)
        + "</body></html>"
    )
    path.write_text(html, encoding="utf-8")  # MHTML: Word открывает как .doc
    return path


# Ключевые слова блоков протокола (рус/en) для разбора ответа LLM по вкладкам
_BLOCK_KEYWORDS = {
    "decisions": ("решени", "decision"),
    "agreements": ("договор", "соглас", "достигнут", "agreement"),
    "next_steps": ("шаг", "задач", "дедлайн", "todo", "next", "action", "следующ"),
}


def parse_summary_blocks(summary: str) -> dict:
    """Разобрать Markdown-протокол на блоки: decisions / agreements / next_steps.

    Строки, не попавшие ни в один блок, игнорируются (полный текст всегда
    доступен в raw). Эвристика устойчива к разным форматированиям заголовков.
    """
    result = {"decisions": [], "agreements": [], "next_steps": [], "raw": summary}
    current = None
    for line in summary.splitlines():
        stripped = line.strip()
        is_heading = bool(
            stripped.startswith("#")
            or (stripped.startswith("**") and stripped.endswith("**"))
            or re.match(r"^\d+[.)]\s+\S", stripped)
        )
        if is_heading:
            low = stripped.lower()
            current = None
            for block, keys in _BLOCK_KEYWORDS.items():
                if any(k in low for k in keys):
                    current = block
                    break
            if current:
                result[current].append(stripped.lstrip("#").strip())
                continue
        if current is not None and stripped:
            result[current].append(line)
    return {k: ("\n".join(v).strip() if isinstance(v, list) else v)
            for k, v in result.items()}
