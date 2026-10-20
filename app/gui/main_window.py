"""Главное окно Meeting Companion (CustomTkinter, тёмная тема — PRD, Модуль 4).

Компоновка:
    Левая панель   — запись: кнопка, таймер 00:00:00, визуализатор волны,
                      прогресс распознавания, статус.
    Центральная    — протокол: вкладки «Решения», «Договорённости»,
                      «Следующие шаги», «MD».
    Правая панель  — транскрипт с таймкодами [MM:SS].
    Нижний блок    — «Экспорт протокола в MD» + статус приватности.

Все тяжёлые операции (ASR, LLM) выполняются в фоновом потоке; взаимодействие
с UI — через очередь событий и after() (потокобезопасно для Tk).
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections import deque
from pathlib import Path

import customtkinter as ctk

from .. import config, llm_core, ollama_local
from ..asr_engine import AsrEngine, inject_pause_markers
from ..audio_capture import MeetingRecorder
from ..realtime import LiveTranscriber
from . import theme

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

WAVE_BARS = 44          # столбцов визуализатора
TICK_MS = 100           # период тика таймера/волны
POLL_MS = 150           # период опроса очереди событий воркера


class App(ctk.CTk):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        config.ensure_dirs()
        self.cfg = config.load_config()

        self.recorder = MeetingRecorder()
        self.asr: AsrEngine | None = None   # создаётся/пересоздаётся при смене настроек
        self.live: LiveTranscriber | None = None   # реалтайм-транскрибация
        self._ollama_started_by_us = False  # наш дочерний ollama serve
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._cancel = threading.Event()
        self._busy = False                  # идёт ASR/LLM-пайплайн
        self._wave_hist: deque[float] = deque([0.0] * WAVE_BARS, maxlen=WAVE_BARS)

        self.summary_text = ""
        self.transcript_text = ""
        self._pauses: list[dict] = []     # снятые паузы текущей записи
        self._topic = ""                  # тема совещания из вкладки «Тема»

        self.title("Meeting Companion — локальный ИИ-секретарь встреч")
        self.geometry("1360x820")
        self.minsize(1120, 680)
        self.configure(fg_color=theme.COLORS["bg"])

        self._build_header()
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()
        self._build_bottom_bar()
        self._layout_grid()

        self.after(POLL_MS, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(400, self._startup_check)   # диагностика локального стека

    # ------------------------------------------------------------ Сборка UI
    def _build_header(self) -> None:
        self.header = ctk.CTkFrame(self, fg_color=theme.COLORS["panel"], corner_radius=0)
        self.title_lbl = ctk.CTkLabel(
            self.header, text="🎙  Meeting Companion",
            font=theme.FONTS["title"], text_color=theme.COLORS["accent"],
        )
        self.title_lbl.pack(side="left", padx=18, pady=10)
        self.device_lbl = ctk.CTkLabel(
            self.header, text="", font=theme.FONTS["small"],
            text_color=theme.COLORS["muted"],
        )
        self.device_lbl.pack(side="left", padx=8)
        self.settings_btn = ctk.CTkButton(
            self.header, text="⚙ Настройки", width=130,
            fg_color=theme.COLORS["card"], hover_color=theme.COLORS["border"],
            command=self.open_settings,
        )
        self.settings_btn.pack(side="right", padx=16, pady=10)
        self.help_btn = ctk.CTkButton(
            self.header, text="? Справка", width=100,
            fg_color=theme.COLORS["card"], hover_color=theme.COLORS["border"],
            command=self.open_help,
        )
        self.help_btn.pack(side="right", padx=(0, 4), pady=10)
        self._refresh_device_label()

    def _build_left_panel(self) -> None:
        self.left = ctk.CTkFrame(self, fg_color=theme.COLORS["panel"], corner_radius=14)
        self.rec_btn = ctk.CTkButton(
            self.left, text="Записать встречу", height=64, corner_radius=14,
            font=(theme.FONT_FAMILY, 17, "bold"),
            fg_color=theme.COLORS["accent_dim"], hover_color=theme.COLORS["accent"],
            text_color="#04140B", command=self.toggle_record,
        )
        self.timer_lbl = ctk.CTkLabel(
            self.left, text="00:00:00", font=theme.FONTS["timer"],
            text_color=theme.COLORS["text"],
        )

        # Кнопка паузы (только во время записи в режиме «Встреча»)
        self.pause_btn = ctk.CTkButton(
            self.left, text="⏸  Пауза", height=34, corner_radius=10,
            font=(theme.FONT_FAMILY, 13, "bold"),
            fg_color=theme.COLORS["card"], hover_color=theme.COLORS["border"],
            border_width=1, border_color=theme.COLORS["border"],
            text_color=theme.COLORS["text"], state="disabled",
            command=self.toggle_pause,
        )

        self.rec_state_lbl = ctk.CTkLabel(
            self.left, text="Готов к записи", font=theme.FONTS["normal"],
            text_color=theme.COLORS["muted"],
        )

        # Выбор режима: «Встреча» (протокол) / «Реалтайм» (живая транскрибация)
        self.mode_btn = ctk.CTkSegmentedButton(
            self.left, values=["Встреча", "Реалтайм"],
            command=self._on_mode_change,
            font=theme.FONTS["normal"],
            fg_color=theme.COLORS["card"],
            selected_color=theme.COLORS["accent_dim"],
            selected_hover_color=theme.COLORS["accent"],
            unselected_color=theme.COLORS["panel"],
            unselected_hover_color=theme.COLORS["border"],
        )
        self.mode_btn.set(self.cfg.get("mode", "Встреча"))

        # Отдельная кнопка запуска обработки (ASR -> протокол), без автозапуска
        self.process_btn = ctk.CTkButton(
            self.left, text="⚙  Обработать запись", height=40, corner_radius=10,
            font=(theme.FONT_FAMILY, 14, "bold"),
            fg_color=theme.COLORS["card"], hover_color=theme.COLORS["border"],
            border_width=1, border_color=theme.COLORS["accent_dim"],
            text_color=theme.COLORS["accent"], state="disabled",
            command=self._start_pipeline,
        )
        if config.MEETING_WAV.is_file():   # запись осталась с прошлого запуска
            self.process_btn.configure(state="normal")

        # Визуализатор аудиоволны
        self.wave_canvas = tk.Canvas(
            self.left, height=110, highlightthickness=0,
            bg=theme.COLORS["card"],
        )
        self._wave_rects: list[int] = []
        for i in range(WAVE_BARS):
            x0 = 8 + i * (200 - 16) / WAVE_BARS
            rect = self.wave_canvas.create_rectangle(
                x0, 55, x0 + (200 - 16) / WAVE_BARS - 2, 55,
                fill=theme.COLORS["wave"], outline="",
            )
            self._wave_rects.append(rect)

        self.progress = ctk.CTkProgressBar(
            self.left, height=14, progress_color=theme.COLORS["accent"],
            fg_color=theme.COLORS["card"],
        )
        self.progress.set(0)
        self.progress_mode: str | None = None   # None | "indeterminate"

        self.status_lbl = ctk.CTkLabel(
            self.left, text="Ваши данные не покидают компьютер",
            font=theme.FONTS["small"], text_color=theme.COLORS["muted"],
            wraplength=280, justify="left",
        )

    def _build_center_panel(self) -> None:
        self.center = ctk.CTkFrame(self, fg_color=theme.COLORS["panel"], corner_radius=14)
        self.center_lbl = ctk.CTkLabel(
            self.center, text="Протокол встречи", font=theme.FONTS["h2"],
            text_color=theme.COLORS["text"], anchor="w",
        )
        self.tabs = ctk.CTkTabview(
            self.center, fg_color=theme.COLORS["card"],
            segmented_button_selected_color=theme.COLORS["accent_dim"],
            segmented_button_selected_hover_color=theme.COLORS["accent"],
        )
        self.tab_texts: dict[str, ctk.CTkTextbox] = {}
        topic_path = config.CACHE_DIR / "meeting_topic.txt"
        for name in ("Тема", "Решения", "Договорённости", "Следующие шаги", "MD"):
            self.tabs.add(name)
            box = ctk.CTkTextbox(
                self.tabs.tab(name), fg_color=theme.COLORS["panel"],
                text_color=theme.COLORS["text"], font=theme.FONTS["normal"],
                wrap="word",
            )
            box.pack(fill="both", expand=True, padx=6, pady=6)
            if name == "Тема":
                # Редактируемая вкладка: тема/повестка для LLM-протокола
                box.configure(state="normal")
                if topic_path.is_file():
                    try:
                        box.insert(
                            "1.0", topic_path.read_text(encoding="utf-8"))
                    except OSError:
                        pass
            else:
                box.configure(state="disabled")
            self.tab_texts[name] = box
        self.tabs.set("Тема")

    def _build_right_panel(self) -> None:
        self.right = ctk.CTkFrame(self, fg_color=theme.COLORS["panel"], corner_radius=14)
        self.right_lbl = ctk.CTkLabel(
            self.right, text="Транскрипт (с таймкодами)", font=theme.FONTS["h2"],
            text_color=theme.COLORS["text"], anchor="w",
        )
        self.transcript_box = ctk.CTkTextbox(
            self.right, fg_color=theme.COLORS["card"], wrap="word",
            text_color=theme.COLORS["text"], font=theme.FONTS["mono"],
        )

    def _build_bottom_bar(self) -> None:
        self.bottom = ctk.CTkFrame(self, fg_color=theme.COLORS["panel"], corner_radius=14)
        self.export_btn = ctk.CTkButton(
            self.bottom, text="⬇  Экспорт протокола в MD", height=42,
            font=(theme.FONT_FAMILY, 14, "bold"),
            fg_color=theme.COLORS["accent_dim"], hover_color=theme.COLORS["accent"],
            text_color="#04140B", command=self.export_md,
        )
        self.export_btn.pack(side="left", padx=(16, 6), pady=10)
        self.export_doc_btn = ctk.CTkButton(
            self.bottom, text="⬇  Экспорт в Word (.doc)", height=42,
            font=(theme.FONT_FAMILY, 14, "bold"),
            fg_color=theme.COLORS["card"], hover_color=theme.COLORS["border"],
            border_width=1, border_color=theme.COLORS["accent_dim"],
            text_color=theme.COLORS["accent"], command=self.export_doc,
        )
        self.export_doc_btn.pack(side="left", padx=(0, 6), pady=10)
        self.privacy_lbl = ctk.CTkLabel(
            self.bottom, text="🔒  Ваши данные никогда не покидают компьютер",
            font=theme.FONTS["normal"], text_color=theme.COLORS["accent"],
        )
        self.privacy_lbl.pack(side="right", padx=20)

    def _layout_grid(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=2)
        self.header.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.left.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=12)
        self.center.grid(row=1, column=1, sticky="nsew", padx=6, pady=12)
        self.right.grid(row=1, column=2, sticky="nsew", padx=(6, 12), pady=12)
        self.bottom.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 12))

        # Левая панель — вертикальный набор виджетов
        self.rec_btn.pack(fill="x", padx=16, pady=(18, 6))
        self.pause_btn.pack(fill="x", padx=16, pady=(0, 4))
        self.timer_lbl.pack(fill="x", pady=(4, 0))
        self.rec_state_lbl.pack(fill="x", padx=16, pady=(0, 6))
        self.mode_btn.pack(fill="x", padx=16, pady=(0, 6))
        self.process_btn.pack(fill="x", padx=16, pady=(0, 8))
        self.wave_canvas.pack(fill="x", padx=16, pady=6)
        self.progress.pack(fill="x", padx=16, pady=(10, 4))
        self.status_lbl.pack(fill="x", padx=16, pady=(6, 16))

        self.center_lbl.pack(fill="x", padx=12, pady=(12, 4))
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 12))

        self.right_lbl.pack(fill="x", padx=12, pady=(12, 4))
        self.transcript_box.pack(fill="both", expand=True, padx=10, pady=(0, 12))



    # ----------------------------------------------------------- Запись
    def toggle_record(self) -> None:
        """Старт/стоп записи; по остановке — автоматический пайплайн ASR+LLM."""
        if self._busy:
            return
        if not self.recorder.is_recording:
            self._start_record()
        else:
            self._stop_record_and_pipeline()

    def _start_record(self) -> None:
        try:
            self.recorder.start()
        except Exception as exc:
            self._set_status(f"Ошибка записи: {exc}", error=True)
            return
        self.process_btn.configure(state="disabled")  # кэш перезаписывается
        self.pause_btn.configure(state="normal", text="⏸  Пауза",
                                 fg_color=theme.COLORS["card"],
                                 border_color=theme.COLORS["border"],
                                 text_color=theme.COLORS["text"])
        self.rec_btn.configure(
            text="⏹  Остановить запись",
            fg_color=theme.COLORS["rec"], hover_color="#FF6B6B",
            text_color="#FFFFFF",
        )
        self.rec_state_lbl.configure(
            text="Идёт запись — говорите свободно",
            text_color=theme.COLORS["rec"],
        )
        self._clear_protocol()
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.configure(state="disabled")
        self._set_progress(0.0)
        note = ""
        if self.recorder.loopback_ok and self.recorder.mic_ok:
            note = " | 🎤 + 🔊 системный звук"
        elif self.recorder.mic_ok:
            note = " | 🎤 только микрофон"
        self._set_status("Запись началась" + note)
        if self.cfg.get("mode") == "Реалтайм":
            self._ensure_asr()
            self.transcript_box.configure(state="normal")
            self.live = LiveTranscriber(
                self.asr,
                on_segment=lambda st, tx: self._events.put(("rt_segment", (st, tx))),
                on_status=lambda t: self._events.put(("rt_status", t)),
            )
            self.live.start()
            self.rec_state_lbl.configure(
                text="Реалтайм-запись: текст появляется по мере речи",
                text_color=theme.COLORS["accent"],
            )
        self.after(TICK_MS, self._tick)

    def _on_mode_change(self, value: str) -> None:
        """Смена режима записи (Встреча/Реалтайм) с сохранением."""
        if self.recorder.is_recording or self._busy:
            self.mode_btn.set(self.cfg.get("mode", "Встреча"))
            self._set_status("Режим нельзя менять во время записи", warn=True)
            return
        self.cfg["mode"] = value
        config.save_config(self.cfg)
        if value == "Реалтайм":
            self._set_status(
                "Реалтайм: текст сразу после паузы (~0.5 с) и каждые ~6 с "
                "при непрерывной речи; прогружаю модель…")
            self._preload_asr()
        else:
            self._set_status("Режим «Встреча»: запись → распознавание → протокол")

    def _ensure_asr(self) -> AsrEngine:
        """Создать движок ASR один раз (вызывать только из Tk-потока)."""
        if self.asr is None:
            self.asr = AsrEngine(
                model_name=self.cfg["gigaam_model"],
                device_mode=self.cfg["asr_device"],
                chunk_seconds=self.cfg["chunk_seconds"],
            )
        return self.asr

    def _preload_asr(self) -> None:
        """Прогрузить GigaAM заранее (выбор «Реалтайм») — первый текст без паузы.

        Иначе модель (6–10 с) грузилась бы уже во время записи.
        """
        engine = self._ensure_asr()   # конструктор мгновенный, в Tk-потоке

        def _worker() -> None:
            try:
                engine.load(status_cb=lambda s: self._events.put(("status", s)))
                self._events.put(
                    ("status", "Модель GigaAM готова — можно записывать"))
            except Exception as exc:
                self._events.put(("error", f"Не удалось подготовить модель: {exc}"))
        threading.Thread(target=_worker, daemon=True).start()

    def toggle_pause(self) -> None:
        """Пауза/снятие паузы записи; при снятии — тайм-код перерыва."""
        rec = self.recorder
        if not rec.is_recording:
            return
        if not rec.is_paused:
            rec.pause()
            self.pause_btn.configure(
                text="▶  Продолжить",
                fg_color=theme.COLORS["warn"], border_color=theme.COLORS["warn"],
                text_color="#0B0F14",
            )
            self.rec_state_lbl.configure(
                text="⏸ На паузе — звук не пишется",
                text_color=theme.COLORS["warn"],
            )
            self._set_status(
                "Пауза: запись заморожена. При снятии с паузы в транскрипт "
                "будет вставлен тайм-код перерыва.")
        else:
            rec.resume()
            self.pause_btn.configure(
                text="⏸  Пауза",
                fg_color=theme.COLORS["card"], border_color=theme.COLORS["border"],
                text_color=theme.COLORS["text"],
            )
            self.rec_state_lbl.configure(
                text="Идёт запись — говорите свободно",
                text_color=theme.COLORS["rec"],
            )
            secs = int(rec.duration)
            self._set_status(
                f"Запись продолжена с тайм-кодом "
                f"{secs // 60:02d}:{secs % 60:02d} — перерыв отмечен "
                f"(всего перерывов: {len(rec.pause_events)})")

    def _stop_record_and_pipeline(self) -> None:
        self.rec_btn.configure(state="disabled")
        self.pause_btn.configure(state="disabled", text="⏸  Пауза")
        if self.live is not None:          # сначала гасим реалтайм-поток
            self.live.stop()
            self.live = None
        self.rec_state_lbl.configure(
            text="Сохранение аудио…", text_color=theme.COLORS["warn"],
        )
        self._set_status("Останавливаю запись и сохраняю meeting_cache.wav…")
        threading.Thread(target=self._stop_record_worker, daemon=True).start()

    def _stop_record_worker(self) -> None:
        try:
            _path, info = self.recorder.stop()
            warnings = info.get("warnings") or []
            self._pauses = info.get("pauses", [])   # тайм-коды перерывов
            if warnings:
                self._events.put(("warn", " | ".join(warnings)))
            # Автозапуска больше НЕТ: обработка — отдельной кнопкой
            self._events.put(("recording_saved", info.get("duration", 0.0)))
        except Exception as exc:
            self._events.put(("error", f"Не удалось сохранить запись: {exc}"))

    def _tick(self) -> None:
        """Таймер 00:00:00 + визуализатор волны (вызывается из Tk-потока)."""
        if not self.recorder.is_recording:
            return
        secs = int(self.recorder.duration)
        self.timer_lbl.configure(
            text=f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"
        )
        self._wave_hist.append(self.recorder.get_level())
        width = self.wave_canvas.winfo_width() or 200
        bar_w = max(2.0, (width - 16) / WAVE_BARS)
        mid = 55
        for i, (rect, level) in enumerate(zip(self._wave_rects, self._wave_hist)):
            x0 = 8 + i * bar_w
            half = max(1.0, level * 50)
            self.wave_canvas.coords(rect, x0, mid - half, x0 + bar_w - 2, mid + half)
        self.after(TICK_MS, self._tick)

    # -------------------------------------------------------- Пайплайн ASR+LLM
    def _start_pipeline(self) -> None:
        """Запустить распознавание + суммаризацию (кнопка «Обработать запись»)."""
        if self._busy or self.recorder.is_recording:
            return
        if not config.MEETING_WAV.is_file():
            self._set_status("Нет записи для обработки — сначала запишите встречу",
                             warn=True)
            return
        # Тема совещания — читаем в Tk-потоке и сохраняем в кэш (для LLM)
        topic_box = self.tab_texts.get("Тема")
        self._topic = topic_box.get("1.0", "end").strip() if topic_box else ""
        if self._topic:
            try:
                (config.CACHE_DIR / "meeting_topic.txt").write_text(
                    self._topic, encoding="utf-8")
            except OSError:
                pass
        self._busy = True
        self._cancel.clear()
        self.settings_btn.configure(state="disabled")
        self.process_btn.configure(state="disabled")
        self.mode_btn.configure(state="disabled")
        self.rec_btn.configure(state="disabled", text="Идёт обработка…")
        self.rec_state_lbl.configure(
            text="Распознавание речи…", text_color=theme.COLORS["warn"],
        )
        threading.Thread(target=self._pipeline_worker, daemon=True).start()

    def _pipeline_worker(self) -> None:
        """Фоновый поток: GigaAM -> транскрипт -> Ollama -> протокол."""
        emit = self._events.put
        try:
            # ---- ASR (Модуль 2)
            engine = self._ensure_asr()   # без гонок: настройки заблокированы при busy
            emit(("progress_mode", "determinate"))
            segments = engine.transcribe_file(
                config.MEETING_WAV,
                progress_cb=lambda frac, text: emit(("progress", (frac, text))),
                cancel_event=self._cancel,
            )
            segments = inject_pause_markers(segments, self._pauses)
            self.transcript_text = AsrEngine.segments_to_text(segments)
            config.TRANSCRIPT_PATH.write_text(self.transcript_text, encoding="utf-8")
            emit(("transcript", self.transcript_text))
            if not segments:
                emit(("error", "Речь не обнаружена — пустая запись?"))
                return

            # ---- LLM (Модуль 3)
            emit(("progress_mode", "indeterminate"))
            emit(("status", f"Суммаризация через Ollama ({self.cfg['ollama_model']})…"))
            ok, _models = llm_core.check_server(self.cfg["ollama_url"])
            if not ok and ollama_local.is_installed():
                # автономный вариант: поднимаем портативный сервер из проекта
                emit(("status", "Запускаю портативную Ollama из проекта…"))
                if ollama_local.start_server(
                        status_cb=lambda t: emit(("status", t))):
                    self._ollama_started_by_us = True
                    ok, _models = llm_core.check_server(self.cfg["ollama_url"])
            if not ok:
                emit(("warn",
                      "Ollama недоступна — протокол не построен. Транскрипт готов, "
                      "запустите Ollama (Настройки → портативная Ollama) и повторите."))
                return
            self.summary_text = llm_core.summarize(
                self.transcript_text,
                model=self.cfg["ollama_model"],
                url=self.cfg["ollama_url"],
                max_chars=self.cfg["llm_max_chars"],
                status_cb=lambda text: emit(("status", text)),
                topic=self._topic,      # тема/повестка из вкладки «Тема»
            )
            config.SUMMARY_PATH.write_text(self.summary_text, encoding="utf-8")
            emit(("summary", self.summary_text))
            emit(("status", "Протокол готов — можно экспортировать в MD"))
        except Exception as exc:
            emit(("error", str(exc)))
        finally:
            emit(("pipeline_done", None))

    # ------------------------------------------------------- События UI
    def _poll_events(self) -> None:
        """Потокобезопасный перенос событий фонового потока в UI."""
        try:
            while True:
                kind, payload = self._events.get_nowait()
                getattr(self, f"_ev_{kind}")(payload)
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll_events)

    def _ev_status(self, text: str) -> None:
        self._set_status(text)

    def _ev_warn(self, text: str) -> None:
        self._set_status(text, warn=True)

    def _ev_error(self, text: str) -> None:
        self._set_status(text, error=True)

    def _ev_progress(self, payload) -> None:
        frac, text = payload
        self.progress.set(max(0.0, min(1.0, frac)))
        self.rec_state_lbl.configure(text=text, text_color=theme.COLORS["muted"])

    def _ev_progress_mode(self, mode: str | None) -> None:
        if mode == "indeterminate":
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress.set(0)

    def _ev_transcript(self, text: str) -> None:
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", text)
        self.transcript_box.configure(state="disabled")

    def _ev_summary(self, text: str) -> None:
        blocks = llm_core.parse_summary_blocks(text)
        mapping = {
            "Решения": blocks["decisions"],
            "Договорённости": blocks["agreements"],
            "Следующие шаги": blocks["next_steps"],
            "MD": blocks["raw"],
        }
        for name, content in mapping.items():
            box = self.tab_texts[name]
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", content or "—")
            box.configure(state="disabled")
        self.tabs.set("Решения")

    def _ev_recording_saved(self, duration: float) -> None:
        """Запись сохранена; обработка запускается только кнопкой."""
        secs = int(duration or 0)
        self.process_btn.configure(state="normal")
        self.rec_btn.configure(
            state="normal", text="Записать встречу",
            fg_color=theme.COLORS["accent_dim"], hover_color=theme.COLORS["accent"],
            text_color="#04140B",
        )
        self.rec_state_lbl.configure(
            text=f"Запись готова ({secs // 60:02d}:{secs % 60:02d}) — "
                 "нажмите «Обработать запись»",
            text_color=theme.COLORS["accent"],
        )
        self._set_status(
            "Аудио сохранено (16 kHz моно, cache/meeting_cache.wav). "
            "Запуск распознавания и протокола — кнопкой «Обработать запись».")

    def _ev_rt_status(self, text: str) -> None:
        """Статус реалтайм-потока (загрузка модели, ошибки)."""
        error = ("недоступен" in text or "ошибка" in text.lower())
        color = theme.COLORS["danger"] if error else theme.COLORS["accent"]
        self.rec_state_lbl.configure(text=text, text_color=color)

    def _ev_rt_segment(self, payload) -> None:
        """Готовая реплика реалтайма -> строка в панели транскрипта."""
        start, text = payload
        line = f"{AsrEngine.format_timestamp(start)} {text}\n"
        self.transcript_box.configure(state="normal")
        self.transcript_box.insert("end", line)
        self.transcript_box.see("end")

    def _ev_pipeline_done(self, _payload) -> None:
        self._busy = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.rec_btn.configure(
            state="normal", text="Записать встречу",
            fg_color=theme.COLORS["accent_dim"], hover_color=theme.COLORS["accent"],
            text_color="#04140B",
        )
        self.settings_btn.configure(state="normal")
        self.mode_btn.configure(state="normal")
        # Обработать можно заново (например, после смены модели Ollama)
        if config.MEETING_WAV.is_file():
            self.process_btn.configure(state="normal")
        self.rec_state_lbl.configure(
            text="Готово — можно записать заново или переобработать",
            text_color=theme.COLORS["muted"],
        )

    # ------------------------------------------------------------ Экспорт
    def export_md(self) -> None:
        """Сохранить протокол в .md (кнопка нижней панели)."""
        text = self.summary_text or ""
        if not text.strip():
            self._set_status("Протокол пока пуст — сначала запишите встречу.",
                             warn=True)
            return
        try:
            path = llm_core.export_markdown(
                text, transcript=self.transcript_text, topic=self._topic)
            self._set_status(f"Протокол сохранён: {path.name} (папка exports)")
        except OSError as exc:
            self._set_status(f"Ошибка экспорта: {exc}", error=True)

    def export_doc(self) -> None:
        """Сохранить протокол в Word .doc (кнопка нижней панели)."""
        text = self.summary_text or ""
        if not text.strip():
            self._set_status("Протокол пока пуст — сначала запишите встречу.",
                             warn=True)
            return
        try:
            path = llm_core.export_doc(
                text, transcript=self.transcript_text, topic=self._topic)
            self._set_status(f"Протокол сохранён: {path.name} (папка exports)")
        except OSError as exc:
            self._set_status(f"Ошибка экспорта: {exc}", error=True)

    def open_help(self) -> None:
        """Открыть справку (HELP.md) внешним просмотрщиком."""
        help_path = config.BASE_DIR / "HELP.md"
        if not help_path.is_file():
            self._set_status("Файл справки HELP.md не найден рядом с main.py",
                             warn=True)
            return
        try:
            import os
            os.startfile(str(help_path))   # Windows: открыть по умолчанию
        except OSError as exc:
            self._set_status(f"Не удалось открыть справку: {exc}", error=True)
    def open_settings(self) -> None:
        """Модальное окно настройки: устройство ASR, модели (PRD, п. 2.1)."""
        win = ctk.CTkToplevel(self)
        win.title("Настройки")
        win.geometry("520x650")
        win.configure(fg_color=theme.COLORS["bg"])
        win.grab_set()
        win.transient(self)

        body = ctk.CTkFrame(win, fg_color=theme.COLORS["panel"], corner_radius=12)
        body.pack(fill="both", expand=True, padx=16, pady=16)

        def section(parent, text):
            lbl = ctk.CTkLabel(parent, text=text, font=theme.FONTS["h2"],
                               text_color=theme.COLORS["accent"], anchor="w")
            lbl.pack(fill="x", padx=16, pady=(14, 2))

        def field_label(parent, text):
            lbl = ctk.CTkLabel(parent, text=text, font=theme.FONTS["small"],
                               text_color=theme.COLORS["muted"], anchor="w")
            lbl.pack(fill="x", padx=16)

        section(body, "Распознавание речи (GigaAM)")
        field_label(body, "Устройство ASR — выбор CPU сохраняется между запусками")
        device_menu = ctk.CTkOptionMenu(
            body, values=config.DEVICE_CHOICES, width=260,
            fg_color=theme.COLORS["card"],
            button_color=theme.COLORS["accent_dim"],
        )
        device_menu.set(self.cfg["asr_device"])
        device_menu.pack(fill="x", padx=16, pady=(2, 8))

        field_label(body, "Версия модели GigaAM")
        model_menu = ctk.CTkOptionMenu(
            body, values=config.GIGAAM_MODEL_CHOICES, width=260,
            fg_color=theme.COLORS["card"],
            button_color=theme.COLORS["accent_dim"],
        )
        model_menu.set(self.cfg["gigaam_model"])
        model_menu.pack(fill="x", padx=16, pady=(2, 8))

        section(body, "Суммаризация (Ollama, локально)")
        field_label(body, "Адрес сервера")
        url_entry = ctk.CTkEntry(body, width=320, fg_color=theme.COLORS["card"])
        url_entry.insert(0, self.cfg["ollama_url"])
        url_entry.pack(fill="x", padx=16, pady=(2, 8))

        field_label(body, "Модель Ollama (например llama3.2:3b)")
        model_box = ctk.CTkComboBox(
            body, values=config.DEFAULT_OLLAMA_MODELS, width=320,
            fg_color=theme.COLORS["card"],
            button_color=theme.COLORS["accent_dim"],
        )
        model_box.set(self.cfg["ollama_model"])
        model_box.pack(fill="x", padx=16, pady=(2, 4))

        srv_lbl = ctk.CTkLabel(body, text="", font=theme.FONTS["small"],
                               text_color=theme.COLORS["muted"], anchor="w",
                               wraplength=460, justify="left")
        srv_lbl.pack(fill="x", padx=16, pady=(2, 4))

        def check_server():
            srv_lbl.configure(text_color=theme.COLORS["muted"],
                              text="Проверяю сервер…")

            def _worker():
                ok, models = llm_core.check_server(url_entry.get().strip())

                def _apply():
                    if ok:
                        srv_lbl.configure(
                            text_color=theme.COLORS["accent"],
                            text=f"✅ Ollama доступна. Модели: "
                                 f"{', '.join(models) if models else 'нет загруженных'}",
                        )
                        if models:
                            model_box.configure(values=models)
                    else:
                        srv_lbl.configure(
                            text_color=theme.COLORS["danger"],
                            text="❌ Ollama недоступна — запустите ollama serve",
                        )
                self.after(0, _apply)
            threading.Thread(target=_worker, daemon=True).start()

        ctk.CTkButton(
            body, text="Проверить сервер", width=160,
            fg_color=theme.COLORS["card"], hover_color=theme.COLORS["border"],
            command=check_server,
        ).pack(fill="x", padx=16, pady=(2, 4))

        # ---- Портативная Ollama внутри проекта (автономность, PRD)
        section(body, "Портативная Ollama (в папке проекта)")
        port_lbl = ctk.CTkLabel(body, text="", font=theme.FONTS["small"],
                                text_color=theme.COLORS["muted"], anchor="w",
                                wraplength=460, justify="left")
        port_lbl.pack(fill="x", padx=16, pady=(2, 2))

        def set_port(text: str, ok=None) -> None:
            """Потокобезопасный статус (обновление через Tk after)."""
            color = theme.COLORS["muted"] if ok is None else (
                theme.COLORS["accent"] if ok else theme.COLORS["danger"])

            def apply() -> None:
                try:
                    port_lbl.configure(text=text, text_color=color)
                except Exception:
                    pass  # окно могло быть закрыто
            try:
                self.after(0, apply)
            except Exception:
                pass

        def port_status() -> None:
            zip_path = config.BASE_DIR / "tools" / "ollama-windows-amd64.zip"
            if ollama_local.is_installed():
                has_models = ollama_local.MODELS_DIR.is_dir() and any(
                    ollama_local.MODELS_DIR.iterdir())
                set_port("Бандл распакован ✓  ·  модели: "
                         + ("скачаны ✓" if has_models else "нет — нажмите «Скачать модель»"),
                         True)
            elif zip_path.is_file():
                set_port("Бандл скачан — нажмите «Распаковать»", None)
            else:
                set_port("Бандл не скачан: положите ollama-windows-amd64.zip "
                         "в папку tools/", None)
        port_status()

        def _bg(fn):
            """Выполнить fn в фоне (статус — через set_port)."""
            def wrap() -> None:
                def worker() -> None:
                    try:
                        fn()
                    except Exception as exc:  # noqa: BLE001
                        set_port(f"Ошибка: {exc}", False)
                threading.Thread(target=worker, daemon=True).start()
            return wrap

        def do_unpack() -> None:
            if ollama_local.extract_bundle(status_cb=set_port):
                set_port("Распаковано ✓ запускаю сервер…", True)
                do_start()
            else:
                set_port("Не удалось распаковать бандл", False)

        def do_start() -> None:
            if ollama_local.start_server(status_cb=set_port):
                self._ollama_started_by_us = True
                set_port("Портативная Ollama запущена ✓ (127.0.0.1:11434)", True)
                port_status()
            else:
                set_port("Сервер не запустился — см. cache/ollama_serve.log", False)

        def do_pull() -> None:
            model = model_box.get().strip() or "llama3.2:3b"
            set_port(f"Скачиваю {model} (~2 ГБ, один раз)…", None)
            if ollama_local.pull_model(model, status_cb=set_port):
                set_port(f"Модель {model} скачана ✓", True)
                port_status()
            else:
                set_port(f"Не удалось скачать {model}", False)

        btn_row2 = ctk.CTkFrame(body, fg_color="transparent")
        btn_row2.pack(fill="x", padx=16, pady=(2, 2))
        for _text, _cmd, _w in (
            ("Распаковать", _bg(do_unpack), 120),
            ("Запустить", _bg(do_start), 100),
            ("Скачать модель", _bg(do_pull), 140),
        ):
            ctk.CTkButton(
                btn_row2, text=_text, width=_w,
                fg_color=theme.COLORS["card"], hover_color=theme.COLORS["border"],
                command=_cmd,
            ).pack(side="left", padx=(0, 8))

        def save():
            old_key = (self.cfg["asr_device"], self.cfg["gigaam_model"])
            self.cfg["asr_device"] = device_menu.get()
            self.cfg["gigaam_model"] = model_menu.get()
            self.cfg["ollama_url"] = url_entry.get().strip() \
                or config.DEFAULTS["ollama_url"]
            self.cfg["ollama_model"] = model_box.get().strip() \
                or config.DEFAULTS["ollama_model"]
            config.save_config(self.cfg)
            if (self.cfg["asr_device"], self.cfg["gigaam_model"]) != old_key \
                    and self.asr is not None:
                self.asr.unload()          # сбросить модель под новые настройки
                self.asr = None
            self._refresh_device_label()
            self._set_status("Настройки сохранены")
            win.destroy()

        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=10)
        ctk.CTkButton(
            btn_row, text="Сохранить", width=150,
            fg_color=theme.COLORS["accent_dim"], hover_color=theme.COLORS["accent"],
            text_color="#04140B", command=save,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_row, text="Отмена", width=110,
            fg_color=theme.COLORS["card"], hover_color=theme.COLORS["border"],
            command=win.destroy,
        ).pack(side="right")

    # ------------------------------------------------------------ Утилиты
    def _startup_check(self) -> None:
        """Диагностика локального стека при запуске (без внешней сети).

        Показывает в статусной строке: веса GigaAM уже локально или будут
        скачаны при первом распознавании, а также доступность Ollama
        (запрос ТОЛЬКО к localhost — PRD, п. 4.1 «Автономность»).
        """
        def _worker() -> None:
            from ..asr_engine import GIGAAM_DIR
            weights = GIGAAM_DIR.is_dir() and any(
                GIGAAM_DIR.glob(f"*{self.cfg['gigaam_model']}*.ckpt")
            )
            ok, models = llm_core.check_server(self.cfg["ollama_url"])
            parts = ["🔒 Автономный режим"]
            parts.append(
                f"GigaAM {self.cfg['gigaam_model']}: "
                + ("веса локально ✓" if weights
                   else "скачаются при первом распознавании (1 раз)")
            )
            parts.append(
                "Ollama: " + (f"запущена, моделей: {len(models)} ✓" if ok
                              else "не запущена (нужна для протокола)")
            )
            self._events.put(("status", "  ·  ".join(parts)))
        threading.Thread(target=_worker, daemon=True).start()

    def _refresh_device_label(self) -> None:
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            gpu = None
        mode = self.cfg["asr_device"]
        hw = f"GPU: {gpu}" if (mode != "CPU" and gpu) else "CPU-режим"
        self.device_lbl.configure(
            text=f"Устройство: {mode} · {hw} · Модель: {self.cfg['gigaam_model']}"
        )

    def _set_status(self, text: str, error: bool = False, warn: bool = False) -> None:
        color = theme.COLORS["muted"]
        if error:
            color = theme.COLORS["danger"]
        elif warn:
            color = theme.COLORS["warn"]
        self.status_lbl.configure(text=text, text_color=color)

    def _set_progress(self, value: float) -> None:
        if self.progress_mode == "indeterminate":
            self._ev_progress_mode(None)
        self.progress.set(max(0.0, min(1.0, value)))

    def _clear_protocol(self) -> None:
        """Очистить результаты (вкладка «Тема» сохраняется — пользовательские данные)."""
        self.summary_text = ""
        for name, box in self.tab_texts.items():
            if name in ("Тема",):
                continue
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.configure(state="disabled")

    def _on_close(self) -> None:
        """Корректная остановка потоков записи при закрытии окна."""
        self._cancel.set()
        try:
            if self.live is not None:
                self.live.stop(timeout=2.0)
        except Exception:
            pass
        try:
            if self.recorder.is_recording:
                self.recorder.stop()
        except Exception:
            pass
        if self._ollama_started_by_us:
            ollama_local.stop_server()   # портативный сервер — гасим с приложением
        self.destroy()



