"""
JEANIROTRABOT - Animation Manager
Satu loop root.after(100ms) untuk semua animasi GUI.
Tidak ada thread baru — aman untuk tkinter.
"""

import tkinter as tk
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger("JEANIROTRABOT.anim")

TICK_MS = 100  # satu tick = 100ms


@dataclass
class AnimTask:
    kind: str            # "COUNT_UP" | "PULSE" | "SPINNER" | "EASE_BAR" | "FLASH"
    widget: Any          # tk widget atau canvas item id
    data: dict = field(default_factory=dict)
    ticks_left: int = 0
    active: bool = True


class ToastWindow:
    """Pop-up notifikasi kecil fade-in/out di pojok kanan bawah."""

    def __init__(self, root: tk.Tk, action: str, symbol: str,
                 lot: float, price: float, pnl: float):
        self._root = root
        self._win: Optional[tk.Toplevel] = None
        self._alpha = 0.0
        self._phase = "in"   # "in" | "show" | "out" | "done"
        self._show_ticks = 20   # 2 detik
        self._show_elapsed = 0
        self.done = False

        bg = "#0a3d1f" if action == "BUY" else "#3d0a0a"
        border = "#00ff88" if action == "BUY" else "#ff4444"
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        text = f"[{action}] {symbol}  {lot:.2f} lot @ {price:.5g}  {pnl_str}"

        try:
            self._win = tk.Toplevel(root)
            self._win.overrideredirect(True)
            self._win.attributes("-alpha", 0.0)
            self._win.attributes("-topmost", True)
            self._win.configure(bg=border)

            inner = tk.Frame(self._win, bg=bg, padx=10, pady=6)
            inner.pack(padx=2, pady=2)
            tk.Label(inner, text=text, bg=bg, fg="#ffffff",
                     font=("Consolas", 9, "bold")).pack()

            self._win.update_idletasks()
        except Exception as e:
            logger.debug(f"ToastWindow create error: {e}")
            self.done = True

    def position(self, x: int, y: int):
        if self._win:
            try:
                self._win.geometry(f"+{x}+{y}")
            except Exception:
                pass

    def tick(self) -> bool:
        """Return True jika sudah selesai."""
        if self.done or self._win is None:
            return True
        try:
            if self._phase == "in":
                self._alpha = min(1.0, self._alpha + 0.5)
                self._win.attributes("-alpha", self._alpha)
                if self._alpha >= 1.0:
                    self._phase = "show"
            elif self._phase == "show":
                self._show_elapsed += 1
                if self._show_elapsed >= self._show_ticks:
                    self._phase = "out"
            elif self._phase == "out":
                self._alpha = max(0.0, self._alpha - 0.33)
                self._win.attributes("-alpha", self._alpha)
                if self._alpha <= 0.0:
                    self._phase = "done"
                    self.done = True
                    self._win.destroy()
                    self._win = None
        except Exception:
            self.done = True
        return self.done

    def destroy(self):
        if self._win:
            try:
                self._win.destroy()
            except Exception:
                pass
            self._win = None
        self.done = True


class AnimationManager:
    """Mengelola semua animasi GUI via satu root.after loop."""

    SPINNER_FRAMES = ["⟳", "↻", "↺", "⟲"]

    def __init__(self, root: tk.Tk):
        self._root = root
        self._running = False
        self._tasks: dict[str, AnimTask] = {}
        self._toasts: list = []
        self._after_id: Optional[int] = None

        self._pnl_labels: dict = {}
        self._gauge_canvas: Optional[tk.Canvas] = None
        self._gauge_label: Optional[tk.Label] = None
        self._badges: dict[str, tk.Canvas] = {}

        self._last_pnl: dict = {}
        self._gauge_current: float = 0.0
        self._gauge_target_pct: float = 70.0

    def start(self):
        self._running = True
        self._tick()

    def stop(self):
        self._running = False
        if self._after_id:
            try:
                self._root.after_cancel(self._after_id)
            except Exception:
                pass
        for t in list(self._toasts):
            try:
                t.destroy()
            except Exception:
                pass
        self._toasts.clear()

    def register_pnl_labels(self, acc_labels: dict):
        self._pnl_labels = acc_labels

    def register_gauge(self, canvas: tk.Canvas, label):
        self._gauge_canvas = canvas
        self._gauge_label = label

    def register_badge(self, name: str, canvas: tk.Canvas):
        self._badges[name] = canvas

    def _tick(self):
        if not self._running:
            return
        try:
            self._process_tasks()
        except Exception as e:
            logger.debug(f"Anim tick error: {e}")
        self._after_id = self._root.after(TICK_MS, self._tick)

    def _process_tasks(self):
        done = []
        for key, task in list(self._tasks.items()):
            if not task.active:
                done.append(key)
                continue
            try:
                self._step_task(task)
                task.ticks_left -= 1
                if task.ticks_left <= 0:
                    task.active = False
                    done.append(key)
            except Exception as e:
                logger.debug(f"Task {key} error: {e}")
                done.append(key)
        for key in done:
            self._tasks.pop(key, None)
        self._tick_toasts()

    def _step_task(self, task: AnimTask):
        kind = task.kind
        if kind == "COUNT_UP":
            self._step_count_up(task)
        elif kind == "FLASH":
            self._step_flash(task)
        elif kind == "PULSE":
            self._step_pulse(task)
        elif kind == "SPINNER":
            self._step_spinner(task)
        elif kind == "EASE_BAR":
            self._step_ease_bar(task)
        elif kind == "FLASH_BADGE":
            self._step_flash_badge(task)

    # ── Animation step handlers ──

    def _step_count_up(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        t = elapsed / total if total > 0 else 1.0
        t_eased = 1 - (1 - t) ** 3
        current = d["start"] + (d["end"] - d["start"]) * t_eased
        currency = d.get("currency", "")
        fmt = d.get("fmt", "{:,.2f}")
        try:
            task.widget.configure(text=f"{fmt.format(current)} {currency}".strip())
        except Exception:
            task.active = False

    def _step_flash(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        halfway = total // 2
        color = d["flash_color"] if elapsed < halfway else d["normal_color"]
        try:
            task.widget.configure(foreground=color)
        except Exception:
            task.active = False

    def _step_pulse(self, task: AnimTask):
        d = task.data
        half = d.get("half_period", 4)
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        phase = (elapsed // half) % 2
        color = d["color_bright"] if phase == 0 else d["color_dim"]
        canvas = task.widget
        item_id = d.get("item_id")
        if item_id and canvas:
            try:
                canvas.itemconfig(item_id, fill=color)
            except Exception:
                pass

    def _step_spinner(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        frame_idx = (elapsed // d.get("frame_ticks", 2)) % len(self.SPINNER_FRAMES)
        char = self.SPINNER_FRAMES[frame_idx]
        try:
            task.widget.configure(text=char)
        except Exception:
            task.active = False

    def _step_ease_bar(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        t = elapsed / total if total > 0 else 1.0
        t_eased = 1 - (1 - t) ** 3
        current = d["start_val"] + (d["end_val"] - d["start_val"]) * t_eased
        self._gauge_current = current
        self._draw_gauge(current, d.get("target_pct", 70.0))

    def _step_flash_badge(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        color = d["flash_color"] if (elapsed // 2) % 2 == 0 else d["normal_color"]
        canvas = d.get("canvas")
        item_id = d.get("item_id")
        if canvas and item_id:
            try:
                canvas.itemconfig(item_id, fill=color)
            except Exception:
                pass

    # ── Public API ──

    def update_pnl(self, balance: float, equity: float, profit: float, currency: str = ""):
        fields = {"Balance": balance, "Equity": equity, "Profit": profit}
        for name, new_val in fields.items():
            old_val = self._last_pnl.get(name, new_val)
            lbl = self._pnl_labels.get(name)
            if lbl is None:
                continue
            if abs(new_val - old_val) < 0.001:
                continue

            if name == "Profit":
                color = "#00ff88" if new_val > 0 else ("#ff4444" if new_val < 0 else "#888888")
            else:
                color = "#ffffff"

            try:
                lbl.configure(foreground=color)
            except Exception:
                pass

            total_ticks = 5
            self._tasks[f"count_{name}"] = AnimTask(
                kind="COUNT_UP", widget=lbl, ticks_left=total_ticks,
                data={"start": old_val, "end": new_val, "total_ticks": total_ticks,
                      "currency": currency, "fmt": "{:,.2f}"}
            )

            if abs(new_val - old_val) > 1.0 or (old_val != 0 and abs(new_val - old_val) / abs(old_val) > 0.001):
                self._tasks[f"flash_{name}"] = AnimTask(
                    kind="FLASH", widget=lbl, ticks_left=4,
                    data={"total_ticks": 4,
                          "flash_color": "#ffff00" if new_val > old_val else "#ff8888",
                          "normal_color": color}
                )

        self._last_pnl = {"Balance": balance, "Equity": equity, "Profit": profit}

    def set_agent_state(self, name: str, state: str):
        canvas = self._badges.get(name)
        if canvas is None:
            return

        for key in [f"pulse_{name}", f"spinner_{name}", f"flash_{name}_badge"]:
            self._tasks.pop(key, None)

        item_id = getattr(canvas, "_badge_oval", None)
        spinner_lbl = getattr(canvas, "_spinner_lbl", None)

        if state == "idle":
            if item_id:
                try:
                    canvas.itemconfig(item_id, fill="#444444")
                except Exception:
                    pass
            if spinner_lbl:
                try:
                    spinner_lbl.configure(text="")
                except Exception:
                    pass

        elif state == "active":
            self._tasks[f"pulse_{name}"] = AnimTask(
                kind="PULSE", widget=canvas, ticks_left=9999,
                data={"total_ticks": 9999, "item_id": item_id,
                      "color_bright": "#00ff88", "color_dim": "#004422", "half_period": 4}
            )

        elif state == "processing":
            if item_id:
                try:
                    canvas.itemconfig(item_id, fill="#ffaa00")
                except Exception:
                    pass
            if spinner_lbl:
                self._tasks[f"spinner_{name}"] = AnimTask(
                    kind="SPINNER", widget=spinner_lbl, ticks_left=9999,
                    data={"total_ticks": 9999, "frame_ticks": 2}
                )

        elif state == "error":
            self._tasks[f"flash_{name}_badge"] = AnimTask(
                kind="FLASH_BADGE", widget=None, ticks_left=20,
                data={"total_ticks": 20, "flash_color": "#ff4444",
                      "normal_color": "#444444", "canvas": canvas, "item_id": item_id}
            )

    def update_profit_gauge(self, current_pct: float, target_pct: float = 70.0):
        self._gauge_target_pct = target_pct
        old_val = self._gauge_current
        if abs(current_pct - old_val) < 0.05:
            return
        total_ticks = 10
        self._tasks["gauge_ease"] = AnimTask(
            kind="EASE_BAR", widget=self._gauge_canvas, ticks_left=total_ticks,
            data={"total_ticks": total_ticks, "start_val": old_val,
                  "end_val": current_pct, "target_pct": target_pct}
        )

    def _draw_gauge(self, current_pct: float, target_pct: float = 70.0):
        canvas = self._gauge_canvas
        if canvas is None:
            return
        try:
            w = canvas.winfo_width() or 200
            h = canvas.winfo_height() or 20
            canvas.delete("all")
            canvas.create_rectangle(0, 0, w, h, fill="#1a1a2e", outline="#333333")

            fill_pct = max(0.0, min(100.0, current_pct))
            fill_w = int(w * fill_pct / 100)

            if fill_pct < 30:
                bar_color = "#ff4444"
            elif fill_pct < 60:
                bar_color = "#ffaa00"
            elif fill_pct < 90:
                bar_color = "#00cc66"
            else:
                bar_color = "#00ff88"

            if fill_w > 0:
                canvas.create_rectangle(0, 0, fill_w, h, fill=bar_color, outline="")

            target_x = int(w * target_pct / 100)
            canvas.create_line(target_x, 0, target_x, h, fill="#ffffff", width=1, dash=(3, 2))

            txt = f"{current_pct:+.1f}% / {target_pct:.0f}%"
            canvas.create_text(w // 2, h // 2, text=txt, fill="#ffffff",
                                font=("Consolas", 8, "bold"))
        except Exception as e:
            logger.debug(f"draw_gauge error: {e}")

    def show_trade_toast(self, action: str, symbol: str,
                         lot: float, price: float, pnl: float = 0.0):
        if len(self._toasts) >= 3:
            oldest = self._toasts.pop(0)
            try:
                oldest.destroy()
            except Exception:
                pass
        toast = ToastWindow(self._root, action, symbol, lot, price, pnl)
        if toast.done:
            return
        self._toasts.append(toast)
        self._reposition_toasts()

    def _reposition_toasts(self):
        try:
            rx = self._root.winfo_x() + self._root.winfo_width()
            ry = self._root.winfo_y() + self._root.winfo_height()
        except Exception:
            return
        toast_h = 40
        margin = 10
        for i, toast in enumerate(self._toasts):
            x = rx - 340
            y = ry - (toast_h + margin) * (i + 1) - margin
            toast.position(x, y)

    def _tick_toasts(self):
        still_alive = []
        for toast in self._toasts:
            done = toast.tick()
            if done:
                try:
                    toast.destroy()
                except Exception:
                    pass
            else:
                still_alive.append(toast)
        self._toasts = still_alive
        if still_alive:
            self._reposition_toasts()

    @staticmethod
    def parse_trade_log(msg: str) -> Optional[dict]:
        m = re.search(
            r"AUTO-EXECUTE:\s+(BUY|SELL)\s+([\d.]+)\s+(\w+)",
            msg, re.IGNORECASE
        )
        if m:
            return {"action": m.group(1).upper(), "lot": float(m.group(2)),
                    "symbol": m.group(3), "price": 0.0, "pnl": 0.0}
        m2 = re.search(r"Order filled.*@\s*([\d.]+)", msg)
        if m2:
            return {"action": "FILL", "price": float(m2.group(1)),
                    "lot": 0.0, "symbol": "", "pnl": 0.0}
        return None
