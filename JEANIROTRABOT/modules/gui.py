"""
JEANIROTRABOT - Main GUI Module
4-panel layout with dark theme using ttkbootstrap.
Plug & Play: semua settings dikelola via GUI, auto-save ke JSON.
"""

import logging
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from datetime import datetime
from typing import Optional

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.config import AppConfig
from modules.mt5_connector import MT5Connector
from modules.trading_engine import TradingEngine
from modules.risk_manager import RiskManager
from modules.ai_agent import AIAgent, PROVIDERS, DEFAULT_SYSTEM_PROMPT
from modules.chart import ChartManager
from modules.specialist_agents import AgentSignal
from modules.composer_agent import FinalDecision

logger = logging.getLogger("JEANIROTRABOT.gui")

ACCENT_GREEN = "#00ff88"
ACCENT_RED = "#ff4444"
BG_DARK = "#1a1a2e"


class JeaniroTrabotApp:
    """Main application window - Plug & Play."""

    def __init__(self):
        # Core components
        self.config = AppConfig()
        self.connector = MT5Connector()
        self.risk = RiskManager(self.config)
        self.ai = AIAgent(
            provider=self.config.get("AI_PROVIDER", "OpenAI"),
            api_key=self.config.get("AI_API_KEY"),
            model=self.config.get("AI_MODEL"),
            base_url=self.config.get("AI_BASE_URL"),
            system_prompt=self.config.get("AI_SYSTEM_PROMPT"),
        )
        self.engine = TradingEngine(self.connector, self.risk, self.ai, self.config)
        self.chart_mgr = ChartManager()

        # GUI root
        self.root = ttkb.Window(
            title="JEANIROTRABOT - Robot Trading Otomatis",
            themename="darkly",
            size=(1280, 800),
            minsize=(1024, 600),
        )
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set callbacks from engine
        self.engine.set_log_callback(self._append_log_threadsafe)
        self.engine.set_signal_callback(self._update_agent_panel_threadsafe)

        self._build_ui()
        self._start_refresh_timer()

    def run(self):
        self.root.mainloop()

    # ──────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────

    def _build_ui(self):
        # Top-level horizontal: LEFT | CENTER+BOTTOM | RIGHT
        self.main_pw = ttkb.Panedwindow(self.root, orient=tk.HORIZONTAL)
        self.main_pw.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ── Left panel ──
        left_frame = ttkb.Frame(self.main_pw, width=300)
        self.main_pw.add(left_frame, weight=1)
        self._build_left_panel(left_frame)

        # ── Center + Bottom vertical split ──
        center_outer = ttkb.Panedwindow(self.main_pw, orient=tk.VERTICAL)
        self.main_pw.add(center_outer, weight=4)

        center_frame = ttkb.Frame(center_outer)
        center_outer.add(center_frame, weight=3)
        self._build_center_panel(center_frame)

        bottom_frame = ttkb.Frame(center_outer)
        center_outer.add(bottom_frame, weight=1)
        self._build_bottom_panel(bottom_frame)

        # ── Right panel ──
        right_frame = ttkb.Frame(self.main_pw, width=280)
        self.main_pw.add(right_frame, weight=1)
        self._build_right_panel(right_frame)

    # ── Left Panel ──

    def _build_left_panel(self, parent):
        # MT5 Connection
        lf_login = ttkb.Labelframe(parent, text="MT5 Connection", bootstyle="info")
        lf_login.pack(fill=tk.X, padx=5, pady=5)

        ttkb.Label(lf_login, text="Server:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ent_server = ttkb.Entry(lf_login, width=20)
        self.ent_server.grid(row=0, column=1, padx=5, pady=2)
        self.ent_server.insert(0, self.config.get("MT5_SERVER"))

        ttkb.Label(lf_login, text="Login:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ent_login = ttkb.Entry(lf_login, width=20)
        self.ent_login.grid(row=1, column=1, padx=5, pady=2)
        self.ent_login.insert(0, self.config.get("MT5_LOGIN"))

        ttkb.Label(lf_login, text="Password:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.ent_password = ttkb.Entry(lf_login, width=20, show="*")
        self.ent_password.grid(row=2, column=1, padx=5, pady=2)
        self.ent_password.insert(0, self.config.get("MT5_PASSWORD"))

        self.btn_connect = ttkb.Button(
            lf_login, text="Connect", bootstyle="success",
            command=self._on_connect
        )
        self.btn_connect.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        self.lbl_conn_status = ttkb.Label(lf_login, text="Disconnected", foreground=ACCENT_RED)
        self.lbl_conn_status.grid(row=4, column=0, columnspan=2, padx=5, pady=2)

        # Account Info
        lf_account = ttkb.Labelframe(parent, text="Account Info", bootstyle="info")
        lf_account.pack(fill=tk.X, padx=5, pady=5)

        self.acc_labels = {}
        for i, field in enumerate(["Balance", "Equity", "Margin", "Free Margin", "Profit"]):
            ttkb.Label(lf_account, text=f"{field}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=1)
            lbl = ttkb.Label(lf_account, text="--")
            lbl.grid(row=i, column=1, sticky=tk.E, padx=5, pady=1)
            self.acc_labels[field] = lbl

        # Trading Controls
        lf_trade = ttkb.Labelframe(parent, text="Trading Controls", bootstyle="warning")
        lf_trade.pack(fill=tk.X, padx=5, pady=5)

        ttkb.Label(lf_trade, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.cmb_symbol = ttkb.Combobox(lf_trade, width=16, state="readonly")
        self.cmb_symbol.grid(row=0, column=1, padx=5, pady=2)
        self.cmb_symbol.set("EURUSD")

        # Multi-symbol entry
        ttkb.Label(lf_trade, text="Symbols:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ent_symbols = ttkb.Entry(lf_trade, width=18)
        self.ent_symbols.grid(row=1, column=1, padx=5, pady=2)
        saved_symbols = self.config.get("TRADE_SYMBOLS", "EURUSD")
        self.ent_symbols.insert(0, saved_symbols)

        ttkb.Label(lf_trade, text="Timeframe:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.cmb_tf = ttkb.Combobox(
            lf_trade, width=16, state="readonly",
            values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
        )
        self.cmb_tf.grid(row=2, column=1, padx=5, pady=2)
        self.cmb_tf.set("M5")
        self.cmb_tf.bind("<<ComboboxSelected>>", lambda e: self._refresh_chart())

        ttkb.Label(lf_trade, text="Lot Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.spn_lot = ttkb.Spinbox(lf_trade, from_=0.01, to=10.0, increment=0.01, width=14)
        self.spn_lot.grid(row=3, column=1, padx=5, pady=2)
        self.spn_lot.set("0.01")

        # Autonomy level
        ttkb.Label(lf_trade, text="AI Mode:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.cmb_autonomy = ttkb.Combobox(
            lf_trade, width=16, state="readonly",
            values=["Full Autonomous", "Open + Close", "Signals Only"]
        )
        self.cmb_autonomy.grid(row=4, column=1, padx=5, pady=2)
        saved_level = self.config.get("AI_AUTONOMY_LEVEL", "full")
        level_map = {"full": "Full Autonomous", "open_close": "Open + Close", "signals_only": "Signals Only"}
        self.cmb_autonomy.set(level_map.get(saved_level, "Full Autonomous"))

        self.btn_start = ttkb.Button(
            lf_trade, text="Start Robot", bootstyle="success",
            command=self._on_start_robot
        )
        self.btn_start.grid(row=5, column=0, columnspan=2, padx=5, pady=3, sticky=tk.EW)

        self.btn_stop = ttkb.Button(
            lf_trade, text="Stop Robot", bootstyle="secondary",
            command=self._on_stop_robot, state=tk.DISABLED
        )
        self.btn_stop.grid(row=6, column=0, columnspan=2, padx=5, pady=3, sticky=tk.EW)

        self.btn_emergency = ttkb.Button(
            lf_trade, text="EMERGENCY STOP", bootstyle="danger",
            command=self._on_emergency_stop
        )
        self.btn_emergency.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        # Risk Management
        lf_risk = ttkb.Labelframe(parent, text="Risk Management", bootstyle="danger")
        lf_risk.pack(fill=tk.X, padx=5, pady=5)

        risk_fields = [
            ("Max DD %:", "MAX_DRAWDOWN_PERCENT", "5.0"),
            ("Max Orders:", "MAX_ORDERS_PER_DAY", "20"),
            ("Max Lot:", "MAX_LOT_SIZE", "1.0"),
            ("SL (pts):", "DEFAULT_SL_POINTS", "100"),
            ("TP (pts):", "DEFAULT_TP_POINTS", "200"),
        ]
        self.risk_entries = {}
        for i, (label, key, default) in enumerate(risk_fields):
            ttkb.Label(lf_risk, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=1)
            ent = ttkb.Entry(lf_risk, width=10)
            ent.grid(row=i, column=1, padx=5, pady=1)
            ent.insert(0, self.config.get(key, default))
            self.risk_entries[key] = ent

        btn_save_risk = ttkb.Button(
            lf_risk, text="Save Risk Settings", bootstyle="warning-outline",
            command=self._on_save_risk
        )
        btn_save_risk.grid(row=len(risk_fields), column=0, columnspan=2, padx=5, pady=3, sticky=tk.EW)

    # ── Center Panel (Chart) ──

    def _build_center_panel(self, parent):
        lf = ttkb.Labelframe(parent, text="Chart", bootstyle="info")
        lf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ctrl = ttkb.Frame(lf)
        ctrl.pack(fill=tk.X, padx=5, pady=2)

        self.var_sma20 = tk.BooleanVar(value=True)
        self.var_sma50 = tk.BooleanVar(value=True)
        self.var_rsi = tk.BooleanVar(value=True)
        self.var_macd = tk.BooleanVar(value=True)
        self.var_bb = tk.BooleanVar(value=True)

        ttkb.Checkbutton(ctrl, text="SMA 20", variable=self.var_sma20,
                         bootstyle="info-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)
        ttkb.Checkbutton(ctrl, text="SMA 50", variable=self.var_sma50,
                         bootstyle="warning-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)
        ttkb.Checkbutton(ctrl, text="RSI (14)", variable=self.var_rsi,
                         bootstyle="secondary-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)
        ttkb.Checkbutton(ctrl, text="MACD", variable=self.var_macd,
                         bootstyle="primary-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)
        ttkb.Checkbutton(ctrl, text="BB", variable=self.var_bb,
                         bootstyle="danger-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)

        ttkb.Button(ctrl, text="Refresh", bootstyle="info-outline",
                    command=self._refresh_chart).pack(side=tk.RIGHT, padx=5)

        self.chart_frame = ttkb.Frame(lf)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        self._chart_canvas: Optional[FigureCanvasTkAgg] = None

    # ── Right Panel (AI Agent - Multi Provider) ──

    def _build_right_panel(self, parent):
        # ── Provider & Key ──
        lf_provider = ttkb.Labelframe(parent, text="AI Provider & API Key", bootstyle="info")
        lf_provider.pack(fill=tk.X, padx=5, pady=5)

        ttkb.Label(lf_provider, text="Provider:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.cmb_provider = ttkb.Combobox(
            lf_provider, width=26, state="readonly",
            values=AIAgent.get_providers()
        )
        self.cmb_provider.pack(fill=tk.X, padx=5, pady=2)
        saved_provider = self.config.get("AI_PROVIDER", "OpenAI")
        self.cmb_provider.set(saved_provider if saved_provider in AIAgent.get_providers() else "OpenAI")
        self.cmb_provider.bind("<<ComboboxSelected>>", self._on_provider_changed)

        ttkb.Label(lf_provider, text="API Key:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.ent_api_key = ttkb.Entry(lf_provider, width=28, show="*")
        self.ent_api_key.pack(fill=tk.X, padx=5, pady=2)
        self.ent_api_key.insert(0, self.config.get("AI_API_KEY"))

        self.lbl_key_hint = ttkb.Label(lf_provider, text="", foreground="#666666",
                                       font=("Consolas", 8))
        self.lbl_key_hint.pack(anchor=tk.W, padx=5)
        self._update_key_hint()

        ttkb.Label(lf_provider, text="Model:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.cmb_model = ttkb.Combobox(lf_provider, width=26)
        self.cmb_model.pack(fill=tk.X, padx=5, pady=2)
        self._update_model_list()
        saved_model = self.config.get("AI_MODEL")
        if saved_model:
            self.cmb_model.set(saved_model)

        # Custom base URL (for "Custom" provider or overrides)
        self.lf_custom_url = ttkb.Labelframe(lf_provider, text="Base URL (optional)", bootstyle="secondary")
        self.lf_custom_url.pack(fill=tk.X, padx=5, pady=2)
        self.ent_base_url = ttkb.Entry(self.lf_custom_url, width=26)
        self.ent_base_url.pack(fill=tk.X, padx=5, pady=2)
        saved_url = self.config.get("AI_BASE_URL")
        if saved_url:
            self.ent_base_url.insert(0, saved_url)

        # Buttons row
        btn_row = ttkb.Frame(lf_provider)
        btn_row.pack(fill=tk.X, padx=5, pady=3)
        ttkb.Button(btn_row, text="Test", bootstyle="info-outline",
                    command=self._on_test_ai).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttkb.Button(btn_row, text="Save", bootstyle="success-outline",
                    command=self._on_save_ai).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        self.lbl_ai_status = ttkb.Label(lf_provider, text="Not tested", foreground="#888888")
        self.lbl_ai_status.pack(padx=5, pady=2)

        # Enable toggle
        self.var_ai_enabled = tk.BooleanVar(value=False)
        ttkb.Checkbutton(
            lf_provider, text="Enable AI Agent", variable=self.var_ai_enabled,
            bootstyle="success-round-toggle", command=self._on_toggle_ai
        ).pack(padx=5, pady=5)

        # ── Editable System Prompt ──
        lf_prompt = ttkb.Labelframe(parent, text="AI System Prompt (editable)", bootstyle="warning")
        lf_prompt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.txt_prompt = tk.Text(
            lf_prompt, height=8, bg=BG_DARK, fg="#cccccc",
            insertbackground="#ffffff", font=("Consolas", 9),
            wrap=tk.WORD, relief=tk.FLAT, padx=5, pady=5
        )
        self.txt_prompt.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        saved_prompt = self.config.get("AI_SYSTEM_PROMPT")
        self.txt_prompt.insert("1.0", saved_prompt if saved_prompt else DEFAULT_SYSTEM_PROMPT)

        prompt_btns = ttkb.Frame(lf_prompt)
        prompt_btns.pack(fill=tk.X, padx=2, pady=2)
        ttkb.Button(prompt_btns, text="Apply Prompt", bootstyle="warning-outline",
                    command=self._on_apply_prompt).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttkb.Button(prompt_btns, text="Reset Default", bootstyle="secondary-outline",
                    command=self._on_reset_prompt).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # ── AI Decision Log ──
        lf_ai_log = ttkb.Labelframe(parent, text="AI Decision Log", bootstyle="secondary")
        lf_ai_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.txt_ai_log = scrolledtext.ScrolledText(
            lf_ai_log, height=8, bg=BG_DARK, fg="#cccccc",
            insertbackground="#ffffff", font=("Consolas", 9),
            state=tk.DISABLED, wrap=tk.WORD
        )
        self.txt_ai_log.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ── Agent Signals Panel ──
        self._build_agent_panel(parent)

    def _build_agent_panel(self, parent):
        """Build the multi-agent signals display panel."""
        lf = ttkb.Labelframe(parent, text="🤖 Agent Signals", bootstyle="primary")
        lf.pack(fill=tk.X, padx=5, pady=5)

        # Header row
        hdr = ttkb.Frame(lf)
        hdr.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttkb.Label(hdr, text="Agent", width=14, anchor=tk.W,
                   font=("Consolas", 8, "bold"), foreground="#aaaaaa").pack(side=tk.LEFT)
        ttkb.Label(hdr, text="Signal", width=6, anchor=tk.CENTER,
                   font=("Consolas", 8, "bold"), foreground="#aaaaaa").pack(side=tk.LEFT)
        ttkb.Label(hdr, text="Score", width=10, anchor=tk.CENTER,
                   font=("Consolas", 8, "bold"), foreground="#aaaaaa").pack(side=tk.LEFT)
        ttkb.Label(hdr, text="Conf", width=5, anchor=tk.CENTER,
                   font=("Consolas", 8, "bold"), foreground="#aaaaaa").pack(side=tk.LEFT)

        self._agent_rows: dict[str, dict] = {}
        agent_names = ["TrendAgent", "MomentumAgent", "VolatilityAgent", "VolumeAgent"]
        for name in agent_names:
            row = ttkb.Frame(lf)
            row.pack(fill=tk.X, padx=4, pady=1)
            lbl_name = ttkb.Label(row, text=name, width=14, anchor=tk.W,
                                  font=("Consolas", 8), foreground="#cccccc")
            lbl_name.pack(side=tk.LEFT)
            lbl_action = ttkb.Label(row, text="---", width=6, anchor=tk.CENTER,
                                    font=("Consolas", 8, "bold"), foreground="#aaaaaa")
            lbl_action.pack(side=tk.LEFT)
            lbl_bar = ttkb.Label(row, text="░░░░░░░░", width=10, anchor=tk.CENTER,
                                 font=("Consolas", 8), foreground="#555555")
            lbl_bar.pack(side=tk.LEFT)
            lbl_conf = ttkb.Label(row, text="0%", width=5, anchor=tk.CENTER,
                                  font=("Consolas", 8), foreground="#aaaaaa")
            lbl_conf.pack(side=tk.LEFT)
            self._agent_rows[name] = {
                "action": lbl_action, "bar": lbl_bar, "conf": lbl_conf
            }

        # Separator
        ttkb.Separator(lf, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=3)

        # Composer result row
        comp_row = ttkb.Frame(lf)
        comp_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttkb.Label(comp_row, text="DECISION", width=14, anchor=tk.W,
                   font=("Consolas", 8, "bold"), foreground="#ffffff").pack(side=tk.LEFT)
        self.lbl_composer_action = ttkb.Label(
            comp_row, text="HOLD", width=6, anchor=tk.CENTER,
            font=("Consolas", 9, "bold"), foreground="#aaaaaa"
        )
        self.lbl_composer_action.pack(side=tk.LEFT)
        self.lbl_composer_mode = ttkb.Label(
            comp_row, text="", width=15, anchor=tk.W,
            font=("Consolas", 7), foreground="#666666"
        )
        self.lbl_composer_mode.pack(side=tk.LEFT, padx=(4, 0))

    def _update_agent_panel_threadsafe(self, signals: list, decision):
        """Thread-safe callback — schedules update on the Tk main thread."""
        self.root.after(0, lambda: self._update_agent_panel(signals, decision))

    def _update_agent_panel(self, signals: list, decision):
        """Update agent signal rows and composer decision display."""
        ACTION_COLORS = {
            "BUY": ACCENT_GREEN,
            "SELL": ACCENT_RED,
            "HOLD": "#aaaaaa",
        }
        for sig in signals:
            row = self._agent_rows.get(sig.agent_name)
            if row is None:
                continue
            color = ACTION_COLORS.get(sig.action, "#aaaaaa")
            if sig.error:
                row["action"].configure(text="ERR", foreground="#ff8800")
                row["bar"].configure(text="░░░░░░░░", foreground="#555555")
                row["conf"].configure(text="0%")
            else:
                bar = self._make_score_bar(sig.score)
                row["action"].configure(text=sig.action, foreground=color)
                row["bar"].configure(text=bar, foreground=color)
                row["conf"].configure(text=f"{int(sig.confidence * 100)}%")

        if decision is not None:
            color = ACTION_COLORS.get(decision.action, "#aaaaaa")
            self.lbl_composer_action.configure(
                text=decision.action, foreground=color
            )
            self.lbl_composer_mode.configure(text=decision.mode)

    @staticmethod
    def _make_score_bar(score: float, width: int = 8) -> str:
        """Return a simple text bar representing |score| (0–1 range)."""
        filled = int(min(abs(score), 1.0) * width)
        return "█" * filled + "░" * (width - filled)

    def _update_key_hint(self):
        provider = self.cmb_provider.get()
        hint = AIAgent.get_key_hint(provider)
        self.lbl_key_hint.configure(text=f"Format: {hint}")

    def _update_model_list(self):
        provider = self.cmb_provider.get()
        models = AIAgent.get_models(provider)
        self.cmb_model.configure(values=models)
        if models:
            self.cmb_model.set(models[0])

    def _on_provider_changed(self, event=None):
        self._update_key_hint()
        self._update_model_list()
        self.lbl_ai_status.configure(text="Not tested", foreground="#888888")
        self._append_log(f"AI Provider changed: {self.cmb_provider.get()}")

    # ── Bottom Panel ──

    def _build_bottom_panel(self, parent):
        notebook = ttkb.Notebook(parent, bootstyle="dark")
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Open Positions
        frame_pos = ttkb.Frame(notebook)
        notebook.add(frame_pos, text="Open Positions")
        cols_pos = ("Ticket", "Symbol", "Type", "Volume", "Open Price", "Current", "Profit", "SL", "TP")
        self.tree_positions = ttkb.Treeview(frame_pos, columns=cols_pos, show="headings", height=5, bootstyle="dark")
        for c in cols_pos:
            self.tree_positions.heading(c, text=c)
            self.tree_positions.column(c, width=90, anchor=tk.CENTER)
        self.tree_positions.pack(fill=tk.BOTH, expand=True)

        # Order History
        frame_hist = ttkb.Frame(notebook)
        notebook.add(frame_hist, text="Order History")
        cols_hist = ("Ticket", "Time", "Symbol", "Type", "Volume", "Price", "Profit")
        self.tree_history = ttkb.Treeview(frame_hist, columns=cols_hist, show="headings", height=5, bootstyle="dark")
        for c in cols_hist:
            self.tree_history.heading(c, text=c)
            self.tree_history.column(c, width=100, anchor=tk.CENTER)
        self.tree_history.pack(fill=tk.BOTH, expand=True)

        # Activity Log
        frame_log = ttkb.Frame(notebook)
        notebook.add(frame_log, text="Activity Log")
        self.txt_log = scrolledtext.ScrolledText(
            frame_log, height=8, bg=BG_DARK, fg="#cccccc",
            insertbackground="#ffffff", font=("Consolas", 9),
            state=tk.DISABLED, wrap=tk.WORD
        )
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    # ──────────────────────────────────────────────
    # Event Handlers
    # ──────────────────────────────────────────────

    def _on_connect(self):
        server = self.ent_server.get().strip()
        login_str = self.ent_login.get().strip()
        password = self.ent_password.get().strip()

        if not server or not login_str or not password:
            messagebox.showwarning("Input Error", "Please fill in all MT5 credentials.")
            return
        try:
            login = int(login_str)
        except ValueError:
            messagebox.showerror("Input Error", "Login must be a number.")
            return

        self.btn_connect.configure(state=tk.DISABLED)
        self._append_log("Connecting to MT5...")

        def do_connect():
            ok, msg = self.connector.connect(server, login, password)
            self.root.after(0, lambda: self._handle_connect_result(ok, msg, server, login_str, password))

        threading.Thread(target=do_connect, daemon=True).start()

    def _handle_connect_result(self, ok, msg, server, login_str, password):
        self.btn_connect.configure(state=tk.NORMAL)
        if ok:
            self.lbl_conn_status.configure(text="Connected", foreground=ACCENT_GREEN)
            self._append_log(f"MT5: {msg}")

            # Auto-save credentials on success
            self.config.set("MT5_SERVER", server)
            self.config.set("MT5_LOGIN", login_str)
            self.config.set("MT5_PASSWORD", password)
            self.config.save()

            # Load symbols
            symbols = self.connector.get_symbols()
            if symbols:
                self.cmb_symbol.configure(values=symbols)
                if "EURUSD" in symbols:
                    self.cmb_symbol.set("EURUSD")
                elif symbols:
                    self.cmb_symbol.set(symbols[0])
            self._refresh_chart()
        else:
            self.lbl_conn_status.configure(text="Failed", foreground=ACCENT_RED)
            self._append_log(f"MT5 connection failed: {msg}")
            messagebox.showerror("Connection Error", msg)

    def _on_start_robot(self):
        # Parse multi-symbol list
        symbols_text = self.ent_symbols.get().strip()
        if symbols_text:
            symbols = [s.strip() for s in symbols_text.replace(";", ",").split(",") if s.strip()]
        else:
            symbols = [self.cmb_symbol.get()]

        tf = self.cmb_tf.get()
        try:
            lot = float(self.spn_lot.get())
        except ValueError:
            lot = 0.01

        # Save autonomy level
        autonomy_text = self.cmb_autonomy.get()
        level_rmap = {"Full Autonomous": "full", "Open + Close": "open_close", "Signals Only": "signals_only"}
        autonomy = level_rmap.get(autonomy_text, "full")
        self.config.set("AI_AUTONOMY_LEVEL", autonomy)
        self.config.set("TRADE_SYMBOLS", ",".join(symbols))
        self.config.save()

        self.engine.set_symbols(symbols)
        self.engine.set_timeframe(tf)
        self.engine.set_lot_size(lot)

        ok, msg = self.engine.start()
        if ok:
            self.btn_start.configure(state=tk.DISABLED)
            self.btn_stop.configure(state=tk.NORMAL)
            symbols_str = ", ".join(symbols)
            self._append_log(f"Robot started: [{symbols_str}] {tf} lot={lot} mode={autonomy}")
        else:
            messagebox.showwarning("Start Failed", msg)

    def _on_stop_robot(self):
        self.engine.stop()
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self._append_log("Robot stopped.")

    def _on_emergency_stop(self):
        self._append_log("EMERGENCY STOP triggered!")
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_emergency.configure(state=tk.DISABLED)

        def do_emergency():
            self.engine.emergency_stop()
            self.root.after(0, self._handle_emergency_done)

        threading.Thread(target=do_emergency, daemon=True).start()

    def _handle_emergency_done(self):
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_emergency.configure(state=tk.NORMAL)
        self._append_log("Emergency stop complete.")

    def _on_save_risk(self):
        validations = {
            "MAX_DRAWDOWN_PERCENT": (0.1, 100.0, "Max Drawdown % harus 0.1 - 100"),
            "MAX_ORDERS_PER_DAY": (1, 1000, "Max Orders/Day harus 1 - 1000"),
            "MAX_LOT_SIZE": (0.01, 100.0, "Max Lot Size harus 0.01 - 100"),
            "DEFAULT_SL_POINTS": (1, 10000, "SL points harus 1 - 10000"),
            "DEFAULT_TP_POINTS": (1, 10000, "TP points harus 1 - 10000"),
        }
        for key, ent in self.risk_entries.items():
            val_str = ent.get().strip()
            try:
                val = float(val_str)
            except ValueError:
                messagebox.showerror("Error", f"'{val_str}' bukan angka valid.")
                return
            if key in validations:
                lo, hi, msg = validations[key]
                if val < lo or val > hi:
                    messagebox.showerror("Error", msg)
                    return

        for key, ent in self.risk_entries.items():
            self.config.set(key, ent.get().strip())
        self.config.save()
        self.risk.reload_params()
        self._append_log("Risk settings saved.")

    def _on_test_ai(self):
        key = self.ent_api_key.get().strip()
        if not key:
            messagebox.showwarning("API Key", "Masukkan API key terlebih dahulu.")
            return

        provider = self.cmb_provider.get()
        model = self.cmb_model.get()
        base_url = self.ent_base_url.get().strip()

        # Configure AI agent with current selections
        self.ai.configure(provider=provider, api_key=key, model=model, base_url=base_url)
        self.lbl_ai_status.configure(text="Testing...", foreground="#ffaa00")

        def do_test():
            ok, msg = self.ai.test_connection()
            self.root.after(0, lambda: self._handle_ai_test(ok, msg))

        threading.Thread(target=do_test, daemon=True).start()

    def _handle_ai_test(self, ok, msg):
        if ok:
            self.lbl_ai_status.configure(text="Connected", foreground=ACCENT_GREEN)
            self._append_log(f"AI test OK: {msg}")
        else:
            self.lbl_ai_status.configure(text="Failed", foreground=ACCENT_RED)
            self._append_log(f"AI test failed: {msg}")

    def _on_save_ai(self):
        provider = self.cmb_provider.get()
        key = self.ent_api_key.get().strip()
        model = self.cmb_model.get()
        base_url = self.ent_base_url.get().strip()

        self.config.set("AI_PROVIDER", provider)
        self.config.set("AI_API_KEY", key)
        self.config.set("AI_MODEL", model)
        self.config.set("AI_BASE_URL", base_url)
        self.config.save()

        # Apply to AI agent
        self.ai.configure(provider=provider, api_key=key, model=model, base_url=base_url)
        self._append_log(f"AI settings saved: {provider} / {model}")

    def _on_apply_prompt(self):
        prompt_text = self.txt_prompt.get("1.0", tk.END).strip()
        if not prompt_text:
            messagebox.showwarning("Prompt", "System prompt cannot be empty.")
            return
        self.ai.system_prompt = prompt_text
        self.config.set("AI_SYSTEM_PROMPT", prompt_text)
        self.config.save()
        self._append_log("AI system prompt updated and saved.")

    def _on_reset_prompt(self):
        self.txt_prompt.delete("1.0", tk.END)
        self.txt_prompt.insert("1.0", DEFAULT_SYSTEM_PROMPT)
        self.ai.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.config.set("AI_SYSTEM_PROMPT", "")
        self.config.save()
        self._append_log("AI system prompt reset to default.")

    def _on_toggle_ai(self):
        enabled = self.var_ai_enabled.get()
        self.ai.enabled = enabled
        state = "ENABLED" if enabled else "DISABLED"
        self._append_log(f"AI Agent {state}")

    # ──────────────────────────────────────────────
    # Refresh / Timers
    # ──────────────────────────────────────────────

    def _start_refresh_timer(self):
        self._do_periodic_refresh()

    def _do_periodic_refresh(self):
        if self.connector.connected:
            def do_refresh():
                try:
                    info = self.connector.get_account_info()
                    positions = self.connector.get_positions()
                    history = self.connector.get_history_orders(days=1)
                    self.root.after(0, lambda: self._update_display(info, positions, history))
                except Exception as e:
                    logger.warning(f"Refresh error: {e}")
            threading.Thread(target=do_refresh, daemon=True).start()
        self.root.after(2000, self._do_periodic_refresh)

    def _update_display(self, info, positions, history):
        if info:
            self.acc_labels["Balance"].configure(text=f"{info.balance:,.2f} {info.currency}")
            self.acc_labels["Equity"].configure(text=f"{info.equity:,.2f}")
            self.acc_labels["Margin"].configure(text=f"{info.margin:,.2f}")
            self.acc_labels["Free Margin"].configure(text=f"{info.free_margin:,.2f}")
            color = ACCENT_GREEN if info.profit >= 0 else ACCENT_RED
            self.acc_labels["Profit"].configure(text=f"{info.profit:,.2f}", foreground=color)

        self.tree_positions.delete(*self.tree_positions.get_children())
        for p in positions:
            self.tree_positions.insert("", tk.END, values=(
                p["ticket"], p["symbol"], p["type"], p["volume"],
                f"{p['price_open']:.5f}", f"{p['price_current']:.5f}",
                f"{p['profit']:.2f}", f"{p['sl']:.5f}", f"{p['tp']:.5f}",
            ))

        self.tree_history.delete(*self.tree_history.get_children())
        for h in history[-50:]:
            t = datetime.fromtimestamp(h["time"]).strftime("%H:%M:%S") if h["time"] else "--"
            self.tree_history.insert("", tk.END, values=(
                h["ticket"], t, h["symbol"], h["type"],
                h["volume"], f"{h['price']:.5f}", f"{h['profit']:.2f}",
            ))

    def _refresh_chart(self):
        if not self.connector.connected:
            return
        symbol = self.cmb_symbol.get()
        tf = self.cmb_tf.get()
        self.chart_mgr.toggle_sma20(self.var_sma20.get())
        self.chart_mgr.toggle_sma50(self.var_sma50.get())
        self.chart_mgr.toggle_rsi(self.var_rsi.get())
        self.chart_mgr.toggle_macd(self.var_macd.get())
        self.chart_mgr.toggle_bb(self.var_bb.get())

        def do_chart():
            df = self.connector.get_ohlcv(symbol, tf, count=100)
            if df is not None and len(df) > 5:
                self.root.after(0, lambda: self._render_chart(df, symbol, tf))

        threading.Thread(target=do_chart, daemon=True).start()

    def _render_chart(self, df, symbol, tf):
        if self._chart_canvas:
            self._chart_canvas.get_tk_widget().destroy()
        fig = self.chart_mgr.render(df, symbol, tf)
        if fig:
            self._chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            self._chart_canvas.draw()
            self._chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ──────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────

    def _append_log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, line)
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)
        # Mirror AI-related logs
        if "AI" in msg or "ai" in msg.lower():
            self.txt_ai_log.config(state=tk.NORMAL)
            self.txt_ai_log.insert(tk.END, line)
            self.txt_ai_log.see(tk.END)
            self.txt_ai_log.config(state=tk.DISABLED)

    def _append_log_threadsafe(self, msg: str):
        self.root.after(0, lambda: self._append_log(msg))

    def _on_close(self):
        if self.engine.running:
            self.engine.stop()
        self.connector.disconnect()
        self.chart_mgr.clear()
        self.root.destroy()
