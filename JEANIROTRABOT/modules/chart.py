"""
JEANIROTRABOT - Chart & Indicator Module
Renders candlestick charts with SMA and RSI using mplfinance + matplotlib.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for embedding in Tkinter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf

from modules.trading_engine import (
    compute_sma, compute_rsi, compute_macd, compute_bollinger_bands
)

logger = logging.getLogger("JEANIROTRABOT.chart")


# Custom dark style for mplfinance
DARK_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#00ff88", down="#ff4444",
        edge={"up": "#00ff88", "down": "#ff4444"},
        wick={"up": "#00ff88", "down": "#ff4444"},
        volume={"up": "#00ff8844", "down": "#ff444444"},
    ),
    facecolor="#1a1a2e",
    edgecolor="#1a1a2e",
    figcolor="#1a1a2e",
    gridcolor="#2d2d44",
    gridstyle="--",
    gridaxis="both",
    y_on_right=True,
    rc={
        "axes.labelcolor": "#aaaaaa",
        "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa",
    },
)


class ChartManager:
    """Creates and updates candlestick charts with indicators."""

    def __init__(self):
        self._figure: Optional[Figure] = None
        self._canvas: Optional[FigureCanvasTkAgg] = None
        self._show_sma20 = True
        self._show_sma50 = True
        self._show_rsi = True
        self._show_macd = True
        self._show_bb = True

    @property
    def figure(self) -> Optional[Figure]:
        return self._figure

    @property
    def canvas(self) -> Optional[FigureCanvasTkAgg]:
        return self._canvas

    def set_canvas(self, canvas: FigureCanvasTkAgg):
        self._canvas = canvas

    def toggle_sma20(self, value: bool):
        self._show_sma20 = value

    def toggle_sma50(self, value: bool):
        self._show_sma50 = value

    def toggle_rsi(self, value: bool):
        self._show_rsi = value

    def toggle_macd(self, value: bool):
        self._show_macd = value

    def toggle_bb(self, value: bool):
        self._show_bb = value

    def render(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> Optional[Figure]:
        """
        Render candlestick chart with optional SMA and RSI overlays.
        df must have DatetimeIndex and columns: Open, High, Low, Close, Volume.
        """
        if df is None or len(df) < 5:
            return None

        # Compute indicators
        df = df.copy()
        addplots = []

        if self._show_sma20:
            df["SMA20"] = compute_sma(df["Close"], 20)
            addplots.append(mpf.make_addplot(
                df["SMA20"], color="#00bfff", width=1.2, label="SMA 20"
            ))

        if self._show_sma50:
            df["SMA50"] = compute_sma(df["Close"], 50)
            addplots.append(mpf.make_addplot(
                df["SMA50"], color="#ffaa00", width=1.2, label="SMA 50"
            ))

        if self._show_bb:
            bb_upper, bb_mid, bb_lower, _ = compute_bollinger_bands(df["Close"])
            addplots.append(mpf.make_addplot(
                bb_upper, color="#ff880055", width=0.8, linestyle="--", label="BB Upper"
            ))
            addplots.append(mpf.make_addplot(
                bb_lower, color="#ff880055", width=0.8, linestyle="--", label="BB Lower"
            ))

        if self._show_rsi:
            df["RSI"] = compute_rsi(df["Close"], 14)
            addplots.append(mpf.make_addplot(
                df["RSI"], panel=2, color="#bb86fc", width=1.0,
                ylabel="RSI", ylim=(0, 100), label="RSI(14)"
            ))
            # RSI reference lines
            rsi_70 = pd.Series(70.0, index=df.index)
            rsi_30 = pd.Series(30.0, index=df.index)
            addplots.append(mpf.make_addplot(
                rsi_70, panel=2, color="#ff444488", width=0.5, linestyle="--"
            ))
            addplots.append(mpf.make_addplot(
                rsi_30, panel=2, color="#00ff8888", width=0.5, linestyle="--"
            ))

        if self._show_macd:
            macd_line, signal_line, histogram = compute_macd(df["Close"])
            macd_panel = 3 if self._show_rsi else 2
            addplots.append(mpf.make_addplot(
                macd_line, panel=macd_panel, color="#00bfff", width=1.0, label="MACD"
            ))
            addplots.append(mpf.make_addplot(
                signal_line, panel=macd_panel, color="#ff8800", width=1.0, label="Signal"
            ))
            hist_colors = ["#00ff8855" if v >= 0 else "#ff444455"
                           for v in histogram.fillna(0)]
            addplots.append(mpf.make_addplot(
                histogram, panel=macd_panel, type="bar",
                color=hist_colors, label="Hist"
            ))

        title = f"{symbol} {timeframe}" if symbol else "Chart"

        # Close previous figure
        if self._figure:
            plt.close(self._figure)

        if self._show_rsi and self._show_macd:
            panel_ratios = (4, 1, 2, 2)
        elif self._show_rsi:
            panel_ratios = (4, 1, 2)
        elif self._show_macd:
            panel_ratios = (4, 1, 2)
        else:
            panel_ratios = (4, 1)

        fig, axes = mpf.plot(
            df,
            type="candle",
            style=DARK_STYLE,
            volume=True,
            addplot=addplots if addplots else None,
            title=title,
            returnfig=True,
            figsize=(10, 6),
            panel_ratios=panel_ratios,
            tight_layout=True,
        )

        self._figure = fig

        if self._canvas:
            self._canvas.figure = fig
            self._canvas.draw_idle()

        return fig

    def clear(self):
        """Clear current chart."""
        if self._figure:
            plt.close(self._figure)
            self._figure = None
