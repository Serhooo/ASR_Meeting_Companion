"""Тема GUI: палитра «тёмная + неоновый зелёный», шрифты (PRD, Модуль 4).

Стиль: Dark Mode, неоновые/зелёные акценты, шрифт Segoe UI (Roboto — если
установлен, код пробует его первым).
"""
from __future__ import annotations

import tkinter.font as tkfont

# ---------------------------------------------------------------- Палитра
COLORS = {
    "bg":        "#0B0F14",   # фон приложения
    "panel":     "#111823",   # панели
    "card":      "#161F2E",   # карточки / блоки
    "border":    "#233044",   # тонкие границы
    "accent":    "#00E676",   # неоновый зелёный
    "accent_dim":"#00A854",   # приглушённый зелёный
    "text":      "#E6EDF3",   # основной текст
    "muted":     "#8B949E",   # вторичный текст
    "danger":    "#F85149",   # ошибки
    "warn":      "#D29922",   # предупреждения
    "rec":       "#FF4D4D",   # индикатор записи
    "wave":      "#00E676",   # визуализатор волны
}


def pick_font(candidates: tuple[str, ...] = ("Roboto", "Segoe UI")) -> str:
    """Выбрать первый доступный шрифт; фолбэк Segoe UI без Tk-root.

    theme импортируется до создания главного окна, поэтому Tk-root создаётся
    временный (и сразу уничтожается); при любой ошибке — Segoe UI.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        available = {f.lower(): f for f in tkfont.families(root)}
        root.destroy()
        for name in candidates:
            if name.lower() in available:
                return available[name.lower()]
    except Exception:
        pass
    return "Segoe UI"


FONT_FAMILY = pick_font()

FONTS = {
    "title":   (FONT_FAMILY, 26, "bold"),
    "h2":      (FONT_FAMILY, 16, "bold"),
    "normal":  (FONT_FAMILY, 13),
    "small":   (FONT_FAMILY, 11),
    "mono":    ("Consolas", 13),        # транскрипт
    "timer":   ("Consolas", 34, "bold"),# таймер записи
}
