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
from modules.news_agent import NewsAgent
from modules.research_agent import ResearchAgent
from modules.composer_agent import ComposerAgent, MARKET_MODES
from modules.exchange_connector import ExchangeConnector, SUPPORTED_EXCHANGES

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

        # New agents
        self.news_agent = NewsAgent()
        self.research_agent = ResearchAgent()
        self.composer = ComposerAgent(self.ai, self.config)
        self.exchange_connector: Optional[ExchangeConnector] = None

        # Inject agents ke engine
        self.engine.set_composer(self.composer)
        self.engine.set_news_agent(self.news_agent)
        self.engine.set_research_agent(self.research_agent)

        # Auto-enable AI jika API key tersedia (Fix F/L)
        if self.config.get("AI_API_KEY"):
            self.ai.enabled = True

        # Composer mode change callback
        self.composer.set_mode_change_callback(self._on_composer_mode_change)

        # GUI root
        self.root = ttkb.Window(
            title="JEANIROTRABOT - Robot Trading Otomatis",
            themename="darkly",
            size=(1280, 800),
            minsize=(1024, 600),
        )
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set log callback from engine
        self.engine.set_log_callback(self._append_log_threadsafe)

        self._build_ui()
        self._start_refresh_timer()

        # Auto-connect MT5 jika credentials tersimpan (Fix K)
        if (self.config.get("MT5_SERVER") and
                self.config.get("MT5_LOGIN") and
                self.config.get("MT5_PASSWORD")):
            self.root.after(500, self._auto_connect_on_startup)

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
            ent.bind("<FocusOut>", lambda e: self._on_save_risk())  # Fix N: auto-save
            self.risk_entries[key] = ent

        btn_save_risk = ttkb.Button(
            lf_risk, text="Save Risk Settings", bootstyle="warning-outline",
            command=self._on_save_risk
        )
        btn_save_risk.grid(row=len(risk_fields), column=0, columnspan=2, padx=5, pady=3, sticky=tk.EW)

        # Exchange Connector (Crypto/Indonesia)
        lf_exchange = ttkb.Labelframe(parent, text="Exchange Connector (Crypto)", bootstyle="secondary")
        lf_exchange.pack(fill=tk.X, padx=5, pady=5)

        ccxt_ok = ExchangeConnector.is_available()
        ccxt_status = "ccxt OK" if ccxt_ok else "pip install ccxt"
        ttkb.Label(lf_exchange, text=f"CCXT: {ccxt_status}",
                   foreground=ACCENT_GREEN if ccxt_ok else "#888888",
                   font=("Consolas", 8)).pack(anchor=tk.W, padx=5, pady=1)

        ttkb.Label(lf_exchange, text="Exchange:").pack(anchor=tk.W, padx=5, pady=(3, 0))
        exchange_names = list(SUPPORTED_EXCHANGES.keys())
        self.cmb_exchange = ttkb.Combobox(
            lf_exchange, values=exchange_names, state="readonly", width=24
        )
        self.cmb_exchange.pack(fill=tk.X, padx=5, pady=2)
        saved_ex = self.config.get("CONNECTOR_TYPE", "MT5")
        self.cmb_exchange.set(saved_ex if saved_ex in exchange_names else exchange_names[0])

        ttkb.Label(lf_exchange, text="API Key:").pack(anchor=tk.W, padx=5, pady=(3, 0))
        self.ent_ex_key = ttkb.Entry(lf_exchange, width=26, show="*")
        self.ent_ex_key.pack(fill=tk.X, padx=5, pady=2)
        self.ent_ex_key.insert(0, self.config.get("EXCHANGE_API_KEY", ""))

        ttkb.Label(lf_exchange, text="API Secret:").pack(anchor=tk.W, padx=5, pady=(3, 0))
        self.ent_ex_secret = ttkb.Entry(lf_exchange, width=26, show="*")
        self.ent_ex_secret.pack(fill=tk.X, padx=5, pady=2)
        self.ent_ex_secret.insert(0, self.config.get("EXCHANGE_API_SECRET", ""))

        self.var_testnet = tk.BooleanVar(value=self.config.get("EXCHANGE_TESTNET", "false") == "true")
        ttkb.Checkbutton(
            lf_exchange, text="Testnet / Sandbox", variable=self.var_testnet,
            bootstyle="warning-round-toggle"
        ).pack(anchor=tk.W, padx=5, pady=2)

        self.btn_connect_exchange = ttkb.Button(
            lf_exchange, text="Connect Exchange",
            bootstyle="info" if ccxt_ok else "secondary",
            command=self._on_connect_exchange,
            state=tk.NORMAL if ccxt_ok else tk.DISABLED,
        )
        self.btn_connect_exchange.pack(fill=tk.X, padx=5, pady=3)

        self.lbl_exchange_status = ttkb.Label(lf_exchange, text="Disconnected", foreground=ACCENT_RED)
        self.lbl_exchange_status.pack(padx=5, pady=2)

    # ── Center Panel (Chart) ──

    def _build_center_panel(self, parent):
        lf = ttkb.Labelframe(parent, text="Chart", bootstyle="info")
        lf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ctrl = ttkb.Frame(lf)
        ctrl.pack(fill=tk.X, padx=5, pady=2)

        self.var_sma20 = tk.BooleanVar(value=True)
        self.var_sma50 = tk.BooleanVar(value=True)
        self.var_rsi = tk.BooleanVar(value=True)

        ttkb.Checkbutton(ctrl, text="SMA 20", variable=self.var_sma20,
                         bootstyle="info-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)
        ttkb.Checkbutton(ctrl, text="SMA 50", variable=self.var_sma50,
                         bootstyle="warning-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)
        ttkb.Checkbutton(ctrl, text="RSI (14)", variable=self.var_rsi,
                         bootstyle="secondary-round-toggle",
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
        saved_key = self.config.get("AI_API_KEY", "")
        self.var_ai_enabled = tk.BooleanVar(value=bool(saved_key))  # Fix L: auto-enable jika key ada
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

        # ── Composer Agent Panel ──
        lf_composer = ttkb.Labelframe(parent, text="Composer Agent (Meta-Orchestrator)", bootstyle="warning")
        lf_composer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Enable toggle
        self.var_composer_enabled = tk.BooleanVar(
            value=self.config.get("COMPOSER_ENABLED", "true") == "true"
        )
        ttkb.Checkbutton(
            lf_composer, text="Enable Composer Agent", variable=self.var_composer_enabled,
            bootstyle="warning-round-toggle", command=self._on_toggle_composer
        ).pack(anchor=tk.W, padx=5, pady=3)

        # Target profit + progress
        target_frame = ttkb.Frame(lf_composer)
        target_frame.pack(fill=tk.X, padx=5, pady=2)
        ttkb.Label(target_frame, text="Target Profit:").pack(side=tk.LEFT)
        self.ent_profit_target = ttkb.Entry(target_frame, width=6)
        self.ent_profit_target.pack(side=tk.LEFT, padx=3)
        self.ent_profit_target.insert(0, self.config.get("PROFIT_TARGET_PERCENT", "70.0"))
        ttkb.Label(target_frame, text="%").pack(side=tk.LEFT)

        # Interval
        interval_frame = ttkb.Frame(lf_composer)
        interval_frame.pack(fill=tk.X, padx=5, pady=2)
        ttkb.Label(interval_frame, text="Interval (menit):").pack(side=tk.LEFT)
        self.spn_composer_interval = ttkb.Spinbox(
            interval_frame, from_=5, to=60, increment=5, width=5
        )
        self.spn_composer_interval.pack(side=tk.LEFT, padx=3)
        self.spn_composer_interval.set(self.config.get("COMPOSER_INTERVAL", "10"))

        ttkb.Button(
            lf_composer, text="Save Composer Settings", bootstyle="warning-outline",
            command=self._on_save_composer
        ).pack(fill=tk.X, padx=5, pady=2)

        ttkb.Button(
            lf_composer, text="Force Composer Update", bootstyle="info-outline",
            command=self._on_force_composer
        ).pack(fill=tk.X, padx=5, pady=2)

        # Mode display
        self.lbl_composer_mode = ttkb.Label(
            lf_composer, text="Mode: CONSERVATIVE", foreground="#00aaff",
            font=("Consolas", 9, "bold")
        )
        self.lbl_composer_mode.pack(anchor=tk.W, padx=5, pady=2)

        # Composer log
        self.txt_composer_log = scrolledtext.ScrolledText(
            lf_composer, height=5, bg=BG_DARK, fg="#ffcc88",
            insertbackground="#ffffff", font=("Consolas", 8),
            state=tk.DISABLED, wrap=tk.WORD
        )
        self.txt_composer_log.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

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

        # News & Research tab
        frame_news = ttkb.Frame(notebook)
        notebook.add(frame_news, text="News & Research")
        self._build_news_research_tab(frame_news)

    def _build_news_research_tab(self, parent):
        """Tab News Agent + Research Agent di bottom panel."""
        pw = ttkb.Panedwindow(parent, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # News Agent side
        lf_news = ttkb.Labelframe(pw, text="News Agent", bootstyle="info")
        pw.add(lf_news, weight=1)

        ctrl_news = ttkb.Frame(lf_news)
        ctrl_news.pack(fill=tk.X, padx=3, pady=2)
        self.var_news_enabled = tk.BooleanVar(
            value=self.config.get("NEWS_AGENT_ENABLED", "false") == "true"
        )
        ttkb.Checkbutton(ctrl_news, text="Aktif", variable=self.var_news_enabled,
                         bootstyle="info-round-toggle",
                         command=self._on_toggle_news).pack(side=tk.LEFT, padx=3)
        ttkb.Label(ctrl_news, text="Interval (menit):").pack(side=tk.LEFT, padx=3)
        self.spn_news_interval = ttkb.Spinbox(ctrl_news, from_=5, to=120, increment=5, width=5)
        self.spn_news_interval.pack(side=tk.LEFT)
        self.spn_news_interval.set(self.config.get("NEWS_AGENT_INTERVAL", "30"))
        ttkb.Button(ctrl_news, text="Fetch Now", bootstyle="info-outline",
                    command=self._on_fetch_news).pack(side=tk.RIGHT, padx=3)

        self.txt_news = scrolledtext.ScrolledText(
            lf_news, height=5, bg=BG_DARK, fg="#aaddff",
            insertbackground="#ffffff", font=("Consolas", 8),
            state=tk.DISABLED, wrap=tk.WORD
        )
        self.txt_news.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Research Agent side
        lf_research = ttkb.Labelframe(pw, text="DeepResearch Agent", bootstyle="warning")
        pw.add(lf_research, weight=1)

        ctrl_res = ttkb.Frame(lf_research)
        ctrl_res.pack(fill=tk.X, padx=3, pady=2)
        self.var_research_enabled = tk.BooleanVar(
            value=self.config.get("RESEARCH_AGENT_ENABLED", "false") == "true"
        )
        ttkb.Checkbutton(ctrl_res, text="Aktif", variable=self.var_research_enabled,
                         bootstyle="warning-round-toggle",
                         command=self._on_toggle_research).pack(side=tk.LEFT, padx=3)
        ttkb.Label(ctrl_res, text="Interval (menit):").pack(side=tk.LEFT, padx=3)
        self.spn_research_interval = ttkb.Spinbox(ctrl_res, from_=15, to=240, increment=15, width=5)
        self.spn_research_interval.pack(side=tk.LEFT)
        self.spn_research_interval.set(self.config.get("RESEARCH_AGENT_INTERVAL", "60"))
        ttkb.Button(ctrl_res, text="Analyze Now", bootstyle="warning-outline",
                    command=self._on_run_research).pack(side=tk.RIGHT, padx=3)

        self.txt_research = scrolledtext.ScrolledText(
            lf_research, height=5, bg=BG_DARK, fg="#ffddaa",
            insertbackground="#ffffff", font=("Consolas", 8),
            state=tk.DISABLED, wrap=tk.WORD
        )
        self.txt_research.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    # ──────────────────────────────────────────────
    # Event Handlers
    # ──────────────────────────────────────────────

    def _auto_connect_on_startup(self):
        """Auto-connect ke MT5 menggunakan credentials tersimpan (Fix K)."""
        self._append_log("Auto-connecting ke MT5...")
        self._on_connect()

    def _auto_start_if_ready(self):
        """Auto-start robot setelah connect jika semua setting tersedia (Fix M)."""
        if not self.engine.running and self.connector.connected:
            self._append_log("Auto-starting robot (settings tersimpan)...")
            self._on_start_robot()

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

            # Auto-start robot jika API key tersedia dan robot belum jalan (Fix M)
            if self.config.get("AI_API_KEY") and not self.engine.running:
                self.root.after(1000, self._auto_start_if_ready)
        else:
            self.lbl_conn_status.configure(text="Failed", foreground=ACCENT_RED)
            self._append_log(f"MT5 connection failed: {msg}")
            messagebox.showerror("Connection Error", msg)

    def _on_start_robot(self):
        # Switch connector: gunakan exchange jika dipilih dan connected (Fix G)
        connector_type = self.config.get("CONNECTOR_TYPE", "MT5")
        if connector_type != "MT5" and self.exchange_connector and self.exchange_connector.connected:
            self.engine.connector = self.exchange_connector
            self._append_log(f"Engine menggunakan {connector_type} exchange connector")
        else:
            self.engine.connector = self.connector  # MT5 default

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
        # Mirror Composer logs
        if "[COMPOSER]" in msg or "Composer" in msg:
            self._append_composer_log(msg.replace("[COMPOSER] ", ""))

    def _append_log_threadsafe(self, msg: str):
        self.root.after(0, lambda: self._append_log(msg))

    # ── Exchange Connector handlers ──

    def _on_connect_exchange(self):
        exchange_name = self.cmb_exchange.get()
        api_key = self.ent_ex_key.get().strip()
        api_secret = self.ent_ex_secret.get().strip()
        testnet = self.var_testnet.get()

        self.config.set("CONNECTOR_TYPE", exchange_name)
        self.config.set("EXCHANGE_API_KEY", api_key)
        self.config.set("EXCHANGE_API_SECRET", api_secret)
        self.config.set("EXCHANGE_TESTNET", str(testnet).lower())
        self.config.save()

        self.lbl_exchange_status.configure(text="Connecting...", foreground="#ffaa00")
        self._append_log(f"Menghubungkan ke {exchange_name}...")

        def do_connect():
            conn = ExchangeConnector(exchange_name, api_key, api_secret, testnet)
            ok, msg = conn.connect()
            if ok:
                self.exchange_connector = conn
            self.root.after(0, lambda: self._handle_exchange_connect(ok, msg, conn))

        threading.Thread(target=do_connect, daemon=True).start()

    def _handle_exchange_connect(self, ok, msg, conn):
        if ok:
            self.lbl_exchange_status.configure(text="Connected", foreground=ACCENT_GREEN)
            self._append_log(f"Exchange: {msg}")
            # Swap connector di engine jika user mau
            self._append_log("Exchange terhubung. Gunakan tombol 'Start Robot' untuk trading via exchange.")
        else:
            self.lbl_exchange_status.configure(text="Failed", foreground=ACCENT_RED)
            self._append_log(f"Exchange failed: {msg}")

    # ── News Agent handlers ──

    def _on_toggle_news(self):
        enabled = self.var_news_enabled.get()
        interval = int(self.spn_news_interval.get())
        self.config.set("NEWS_AGENT_ENABLED", str(enabled).lower())
        self.config.set("NEWS_AGENT_INTERVAL", str(interval))
        self.config.save()
        self.news_agent.set_interval(interval)
        self._append_log(f"News Agent {'AKTIF' if enabled else 'NONAKTIF'} (interval={interval} menit)")

    def _on_fetch_news(self):
        self._append_log("Mengambil berita pasar...")
        symbols_text = self.ent_symbols.get().strip()
        symbols = [s.strip() for s in symbols_text.replace(";", ",").split(",") if s.strip()]

        def do_fetch():
            news = self.news_agent.get_news(symbols, max_items=15)
            self.root.after(0, lambda: self._display_news(news))

        threading.Thread(target=do_fetch, daemon=True).start()

    def _display_news(self, news_items: list):
        self.txt_news.config(state=tk.NORMAL)
        self.txt_news.delete("1.0", tk.END)
        if not news_items:
            self.txt_news.insert(tk.END, "Tidak ada berita ditemukan.\n")
        else:
            for item in news_items:
                rel = item.get("relevance", 0)
                line = f"[{item.get('source','')}] {item.get('title','')}"
                if rel > 2:
                    line = "★ " + line
                self.txt_news.insert(tk.END, line + "\n")
                if item.get("summary"):
                    self.txt_news.insert(tk.END, f"  {item['summary'][:100]}\n\n")
        self.txt_news.config(state=tk.DISABLED)
        self._append_log(f"News Agent: {len(news_items)} berita diambil.")

    # ── Research Agent handlers ──

    def _on_toggle_research(self):
        enabled = self.var_research_enabled.get()
        interval = int(self.spn_research_interval.get())
        self.config.set("RESEARCH_AGENT_ENABLED", str(enabled).lower())
        self.config.set("RESEARCH_AGENT_INTERVAL", str(interval))
        self.config.save()
        self.research_agent.set_interval(interval)
        self._append_log(f"Research Agent {'AKTIF' if enabled else 'NONAKTIF'} (interval={interval} menit)")

    def _on_run_research(self):
        if not self.ai.enabled:
            from tkinter import messagebox
            messagebox.showwarning("Research Agent", "Aktifkan AI Agent terlebih dahulu.")
            return
        self._append_log("Menjalankan DeepResearch Agent...")

        symbols_text = self.ent_symbols.get().strip()
        symbols = [s.strip() for s in symbols_text.replace(";", ",").split(",") if s.strip()]

        def do_research():
            report = self.research_agent.run_full_research(symbols[:5], self.ai)
            self.root.after(0, lambda: self._display_research(report))

        threading.Thread(target=do_research, daemon=True).start()

    def _display_research(self, report: dict):
        self.txt_research.config(state=tk.NORMAL)
        self.txt_research.delete("1.0", tk.END)

        fg = report.get("fear_greed", {})
        if fg.get("available"):
            self.txt_research.insert(tk.END,
                f"Fear & Greed Index: {fg['value']}/100 — {fg['label']}\n\n")

        calendar = report.get("calendar", [])
        if calendar:
            self.txt_research.insert(tk.END, f"Economic Events ({len(calendar)} high/medium):\n")
            for ev in calendar[:5]:
                self.txt_research.insert(tk.END,
                    f"  [{ev['country']}] {ev['title']} — {ev.get('date','?')}\n")
            self.txt_research.insert(tk.END, "\n")

        for sym, rep in report.get("symbol_reports", {}).items():
            rec = rep.get("recommendation", "?")
            outlook = rep.get("fundamental_outlook", "")[:100]
            self.txt_research.insert(tk.END, f"{sym} [{rec}]: {outlook}\n")

        self.txt_research.config(state=tk.DISABLED)
        self._append_log("Research Agent selesai.")

    # ── Composer Agent handlers ──

    def _on_toggle_composer(self):
        enabled = self.var_composer_enabled.get()
        self.config.set("COMPOSER_ENABLED", str(enabled).lower())
        self.config.save()
        self._append_log(f"Composer Agent {'AKTIF' if enabled else 'NONAKTIF'}")

    def _on_save_composer(self):
        try:
            target = float(self.ent_profit_target.get())
            interval = int(self.spn_composer_interval.get())
        except ValueError:
            return
        self.config.set("PROFIT_TARGET_PERCENT", str(target))
        self.config.set("COMPOSER_INTERVAL", str(interval))
        self.config.save()
        self.composer._profit_target = target
        self.composer._interval_minutes = interval
        self._append_log(f"Composer: target={target}%, interval={interval} menit disimpan.")

    def _on_force_composer(self):
        if not self.ai.enabled:
            from tkinter import messagebox
            messagebox.showwarning("Composer", "Aktifkan AI Agent terlebih dahulu.")
            return
        self._append_log("Force Composer update...")
        self.composer._last_run = 0  # Reset timer supaya langsung jalan

    def _on_composer_mode_change(self, mode: str, mode_params: dict):
        """Callback saat Composer berganti mode."""
        color = mode_params.get("color", "#ffffff")
        desc = mode_params.get("description", "")
        self.root.after(0, lambda: self._update_composer_display(mode, color, desc))

    def _update_composer_display(self, mode: str, color: str, desc: str):
        self.lbl_composer_mode.configure(text=f"Mode: {mode}", foreground=color)
        self._append_composer_log(f"Mode → {mode}: {desc}")

    def _append_composer_log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        self.txt_composer_log.config(state=tk.NORMAL)
        self.txt_composer_log.insert(tk.END, line)
        self.txt_composer_log.see(tk.END)
        self.txt_composer_log.config(state=tk.DISABLED)

    def _on_close(self):
        if self.engine.running:
            self.engine.stop()
        self.connector.disconnect()
        if self.exchange_connector and self.exchange_connector.connected:
            self.exchange_connector.disconnect()
        self.chart_mgr.clear()
        self.root.destroy()
