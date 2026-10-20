"""Конфигурация приложения: пути, настройки пользователя, системный промпт.

Настройки хранятся в config.json рядом с main.py и переживают перезапуски.
Поведение соответствует PRD (п. 2.1 «Режимы выполнения: CPU / GPU»).
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------- Пути
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
EXPORTS_DIR = BASE_DIR / "exports"
CONFIG_PATH = BASE_DIR / "config.json"

MEETING_WAV = CACHE_DIR / "meeting_cache.wav"          # итоговый 16 kHz моно WAV
MIC_RAW_WAV = CACHE_DIR / "_mic_raw.wav"               # временный сырой поток микрофона
LOOP_RAW_WAV = CACHE_DIR / "_loopback_raw.wav"         # временный сырой поток loopback
TRANSCRIPT_PATH = CACHE_DIR / "meeting_transcript.txt"  # транскрипт с таймкодами
SUMMARY_PATH = CACHE_DIR / "meeting_summary.md"        # сырой ответ LLM

# ------------------------------------------------- Системный промпт LLM
SYSTEM_PROMPT = (
    "Ты — профессиональный секретарь. Проанализируй этот транскрипт встречи "
    "и составь краткий протокол на русском языке строго по трем блокам: "
    "1. Принятые решения. "
    "2. Ключевые договорённости. "
    "3. Следующие шаги (Задачи с дедлайнами/исполнителями, если упомянуты). "
    "Выведи ответ в формате разметки Markdown."
)

# ------------------------------------------------------------- Настройки
DEVICE_CHOICES = ["Авто", "CPU", "CUDA"]                       # PRD п. 2.1
GIGAAM_MODEL_CHOICES = ["v3_e2e_ctc", "v3_e2e_rnnt", "v3_ctc", "v3_rnnt"]
DEFAULT_OLLAMA_MODELS = ["llama3.2:3b", "qwen2.5:3b", "mistral:7b", "llama3:8b"]

DEFAULTS: dict = {
    "asr_device": "Авто",            # Авто | CPU | CUDA
    "gigaam_model": "v3_e2e_ctc",    # версия GigaAM (PRD п. 2.2)
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.2:3b",
    "chunk_seconds": 60,             # длительность сегментов для ASR-чанкинга
    "llm_max_chars": 30000,          # усечение транскрипта для LLM-контекста
    "mode": "Встреча",               # «Встреча» | «Реалтайм» (режим записи)
}


def ensure_dirs() -> None:
    """Создать служебные каталоги, если их ещё нет."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Загрузить настройки; отсутствующие ключи берутся из DEFAULTS."""
    cfg = dict(DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                for key in DEFAULTS:
                    if key in stored:
                        cfg[key] = stored[key]
    except (OSError, json.JSONDecodeError):
        pass  # повреждённый конфиг -> дефолты, приложение не падает
    return cfg


def save_config(cfg: dict) -> None:
    """Сохранить настройки в config.json (атомарно через временный файл)."""
    data = {key: cfg.get(key, DEFAULTS[key]) for key in DEFAULTS}
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)
