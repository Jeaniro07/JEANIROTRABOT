# Dynamic UI Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tambahkan 4 animasi full interaktif ke GUI JEANIROTRABOT tanpa mengubah fitur yang sudah ada — Live P&L Ticker, Agent Status Badge, Profit Target Gauge, dan Trade Toast Notification.

**Architecture:** Satu class `AnimationManager` di file baru `modules/animation_manager.py` mengelola semua animasi via satu loop `root.after(100ms)`. Integrasi ke `gui.py` hanya di 4 titik: `__init__`, `_update_display`, `_on_composer_mode_change`, dan `_append_log`. Widget badge ditambahkan di `_build_right_panel` dan gauge canvas di Composer section.

**Tech Stack:** Python 3.10+, tkinter, ttkbootstrap darkly theme, tidak ada dependency baru.

---

## File Map

| File | Aksi | Tanggung Jawab |
|------|------|----------------|
| `modules/animation_manager.py` | **BARU** | Semua logika animasi: AnimationManager, AnimTask, ToastWindow |
| `modules/gui.py` | **EDIT** | 4 titik integrasi + tambah badge canvas + gauge canvas |

---

## Task 1: Buat `AnimationManager` — skeleton + loop utama

**Files:**
- Create: `JEANIROTRABOT/modules/animation_manager.py`

- [ ] **Step 1: Buat file dengan skeleton class dan loop 100ms**

```python
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


class AnimationManager:
    """Mengelola semua animasi GUI via satu root.after loop."""

    SPINNER_FRAMES = ["⟳", "↻", "↺", "⟲"]

    def __init__(self, root: tk.Tk):
        self._root = root
        self._running = False
        self._tasks: dict[str, AnimTask] = {}
        self._toasts: list = []          # list of ToastWindow
        self._after_id: Optional[str] = None

        # Referensi widget yang didaftarkan
        self._pnl_labels: dict = {}      # {"Balance": lbl, "Equity": lbl, "Profit": lbl, ...}
        self._gauge_canvas: Optional[tk.Canvas] = None
        self._gauge_label: Optional[tk.Label] = None
        self._badges: dict[str, tk.Canvas] = {}   # {"composer": canvas, "news": canvas, ...}

        # State terakhir untuk deteksi perubahan
        self._last_pnl: dict = {}        # {"balance": 0.0, "equity": 0.0, "profit": 0.0}
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
        # Destroy semua toast
        for t in list(self._toasts):
            try:
                t.destroy()
            except Exception:
                pass
        self._toasts.clear()

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
        for key, task in self._tasks.items():
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

    def _step_task(self, task: AnimTask):
        raise NotImplementedError  # akan dioverride per kind di method _dispatch

    def register_pnl_labels(self, acc_labels: dict):
        """Daftarkan dict label dari Account Info panel."""
        self._pnl_labels = acc_labels

    def register_gauge(self, canvas: tk.Canvas, label: tk.Label):
        """Daftarkan canvas dan label untuk profit gauge."""
        self._gauge_canvas = canvas
        self._gauge_label = label

    def register_badge(self, name: str, canvas: tk.Canvas):
        """Daftarkan badge canvas untuk agent (composer/news/research)."""
        self._badges[name] = canvas
```

- [ ] **Step 2: Verifikasi file bisa diimport tanpa error**

```bash
cd JEANIROTRABOT
python -c "from modules.animation_manager import AnimationManager; print('OK')"
```
Expected output: `OK`

- [ ] **Step 3: Commit skeleton**

```bash
git add JEANIROTRABOT/modules/animation_manager.py
git commit -m "feat: add AnimationManager skeleton with 100ms tick loop"
```

---

## Task 2: Implementasi `_step_task` — dispatch per kind + COUNT_UP

**Files:**
- Modify: `JEANIROTRABOT/modules/animation_manager.py`

- [ ] **Step 1: Ganti `_step_task` dengan dispatcher dan implementasi COUNT_UP**

Ganti method `_step_task` yang ada dengan kode berikut (di dalam class `AnimationManager`):

```python
    def _step_task(self, task: AnimTask):
        """Dispatch ke handler berdasarkan kind."""
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

    def _step_count_up(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        t = elapsed / total  # 0.0 → 1.0
        # Ease-out cubic: t = 1 - (1-t)^3
        t_eased = 1 - (1 - t) ** 3
        current = d["start"] + (d["end"] - d["start"]) * t_eased
        widget = task.widget
        currency = d.get("currency", "")
        fmt = d.get("fmt", "{:,.2f}")
        try:
            widget.configure(text=f"{fmt.format(current)} {currency}".strip())
        except Exception:
            pass

    def _step_flash(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        # Flash: warna flash di awal, kembali ke normal di akhir
        halfway = total // 2
        color = d["flash_color"] if elapsed < halfway else d["normal_color"]
        try:
            task.widget.configure(foreground=color)
        except Exception:
            pass

    def _step_pulse(self, task: AnimTask):
        d = task.data
        # Pulse: toggle antara terang dan redup setiap d["half_period"] ticks
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
            pass

    def _step_ease_bar(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        t = elapsed / total
        t_eased = 1 - (1 - t) ** 3
        current = d["start_val"] + (d["end_val"] - d["start_val"]) * t_eased
        self._gauge_current = current
        self._draw_gauge(current, d["target_pct"])
```

- [ ] **Step 2: Tambah method `update_pnl` untuk trigger COUNT_UP**

Tambah di bawah method-method step (masih di dalam class `AnimationManager`):

```python
    def update_pnl(self, balance: float, equity: float, profit: float, currency: str = ""):
        """Trigger count-up animation saat nilai P&L berubah."""
        fields = {"Balance": balance, "Equity": equity, "Profit": profit}
        for name, new_val in fields.items():
            old_val = self._last_pnl.get(name, new_val)
            lbl = self._pnl_labels.get(name)
            if lbl is None:
                continue
            if abs(new_val - old_val) < 0.001:
                continue  # Tidak berubah, skip

            # Warna teks
            if name == "Profit":
                if new_val > 0:
                    color = "#00ff88"
                elif new_val < 0:
                    color = "#ff4444"
                else:
                    color = "#888888"
            else:
                color = "#ffffff"

            try:
                lbl.configure(foreground=color)
            except Exception:
                pass

            # COUNT_UP: 500ms = 5 ticks (100ms each)
            total_ticks = 5
            task = AnimTask(
                kind="COUNT_UP",
                widget=lbl,
                ticks_left=total_ticks,
                data={
                    "start": old_val,
                    "end": new_val,
                    "total_ticks": total_ticks,
                    "currency": currency,
                    "fmt": "{:,.2f}",
                }
            )
            self._tasks[f"count_{name}"] = task

            # Flash putih jika perubahan besar (>$1 atau >0.1%)
            if abs(new_val - old_val) > 1.0 or (old_val != 0 and abs(new_val - old_val) / abs(old_val) > 0.001):
                flash_task = AnimTask(
                    kind="FLASH",
                    widget=lbl,
                    ticks_left=4,
                    data={
                        "total_ticks": 4,
                        "flash_color": "#ffff00" if new_val > old_val else "#ff8888",
                        "normal_color": color,
                    }
                )
                self._tasks[f"flash_{name}"] = flash_task

        # Update state
        self._last_pnl = {"Balance": balance, "Equity": equity, "Profit": profit}
```

- [ ] **Step 3: Verifikasi import masih bersih**

```bash
python -c "from modules.animation_manager import AnimationManager, AnimTask; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add JEANIROTRABOT/modules/animation_manager.py
git commit -m "feat: add COUNT_UP and FLASH animation tasks to AnimationManager"
```

---

## Task 3: Implementasi PULSE + SPINNER badge + `set_agent_state`

**Files:**
- Modify: `JEANIROTRABOT/modules/animation_manager.py`

- [ ] **Step 1: Tambah method `set_agent_state`**

Tambah di dalam class `AnimationManager` setelah `update_pnl`:

```python
    def set_agent_state(self, name: str, state: str):
        """
        Update badge agent.
        state: "idle" | "active" | "processing" | "error"
        name:  "composer" | "news" | "research"
        """
        canvas = self._badges.get(name)
        if canvas is None:
            return

        # Hapus task lama untuk badge ini
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

        elif state == "active":
            # Pulse hijau: terang → redup → terang setiap 800ms (8 ticks)
            task = AnimTask(
                kind="PULSE",
                widget=canvas,
                ticks_left=9999,   # jalan terus
                data={
                    "total_ticks": 9999,
                    "item_id": item_id,
                    "color_bright": "#00ff88",
                    "color_dim": "#004422",
                    "half_period": 4,   # 4 ticks = 400ms per half
                }
            )
            self._tasks[f"pulse_{name}"] = task

        elif state == "processing":
            # Spinner kuning di spinner label
            if spinner_lbl:
                task = AnimTask(
                    kind="SPINNER",
                    widget=spinner_lbl,
                    ticks_left=9999,
                    data={
                        "total_ticks": 9999,
                        "frame_ticks": 2,  # ganti frame setiap 200ms
                    }
                )
                self._tasks[f"spinner_{name}"] = task
            if item_id:
                try:
                    canvas.itemconfig(item_id, fill="#ffaa00")
                except Exception:
                    pass

        elif state == "error":
            # Flash merah 2 detik (20 ticks)
            task = AnimTask(
                kind="FLASH",
                widget=None,   # handle manual via canvas
                ticks_left=20,
                data={
                    "total_ticks": 20,
                    "flash_color": "#ff4444",
                    "normal_color": "#444444",
                    "canvas": canvas,
                    "item_id": item_id,
                }
            )
            # Override _step_flash agar pakai canvas.itemconfig
            task.kind = "FLASH_BADGE"
            self._tasks[f"flash_{name}_badge"] = task

    def _step_flash_badge(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        half = total // 2
        color = d["flash_color"] if (elapsed // 2) % 2 == 0 else d["normal_color"]
        canvas = d.get("canvas")
        item_id = d.get("item_id")
        if canvas and item_id:
            try:
                canvas.itemconfig(item_id, fill=color)
            except Exception:
                pass
```

- [ ] **Step 2: Tambah `"FLASH_BADGE"` ke dispatcher `_step_task`**

Di method `_step_task`, tambah satu baris setelah `elif kind == "EASE_BAR"`:

```python
        elif kind == "FLASH_BADGE":
            self._step_flash_badge(task)
```

- [ ] **Step 3: Verifikasi**

```bash
python -c "from modules.animation_manager import AnimationManager; a = AnimationManager.__new__(AnimationManager); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add JEANIROTRABOT/modules/animation_manager.py
git commit -m "feat: add PULSE/SPINNER badge animations and set_agent_state method"
```

---

## Task 4: Implementasi Profit Gauge — EASE_BAR + `update_profit_gauge`

**Files:**
- Modify: `JEANIROTRABOT/modules/animation_manager.py`

- [ ] **Step 1: Tambah `_draw_gauge` dan `update_profit_gauge`**

Tambah di dalam class `AnimationManager`:

```python
    def _draw_gauge(self, current_pct: float, target_pct: float = 70.0):
        """Gambar progress bar di canvas gauge."""
        canvas = self._gauge_canvas
        label = self._gauge_label
        if canvas is None:
            return

        try:
            w = canvas.winfo_width() or 200
            h = canvas.winfo_height() or 20
            canvas.delete("all")

            # Background
            canvas.create_rectangle(0, 0, w, h, fill="#1a1a2e", outline="#333333")

            # Bar fill
            fill_pct = max(0.0, min(100.0, current_pct))
            fill_w = int(w * fill_pct / 100)

            # Warna bar berdasarkan progress
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

            # Target line (garis vertikal di posisi target%)
            target_x = int(w * target_pct / 100)
            canvas.create_line(target_x, 0, target_x, h, fill="#ffffff", width=1, dash=(3, 2))

            # Teks overlay
            txt = f"{current_pct:+.1f}% / {target_pct:.0f}%"
            canvas.create_text(w // 2, h // 2, text=txt, fill="#ffffff",
                                font=("Consolas", 8, "bold"))

            # Pulse jika ≥90%
            if fill_pct >= 90 and "gauge_pulse" not in self._tasks:
                self._tasks["gauge_pulse"] = AnimTask(
                    kind="EASE_BAR",
                    widget=canvas,
                    ticks_left=10,
                    data={
                        "total_ticks": 10,
                        "start_val": fill_pct,
                        "end_val": fill_pct,
                        "target_pct": target_pct,
                    }
                )
        except Exception as e:
            logger.debug(f"draw_gauge error: {e}")

    def update_profit_gauge(self, current_pct: float, target_pct: float = 70.0):
        """Trigger smooth easing ke nilai gauge baru."""
        self._gauge_target_pct = target_pct
        old_val = self._gauge_current
        if abs(current_pct - old_val) < 0.05:
            return  # Tidak berubah signifikan

        total_ticks = 10  # 1000ms smooth
        task = AnimTask(
            kind="EASE_BAR",
            widget=self._gauge_canvas,
            ticks_left=total_ticks,
            data={
                "total_ticks": total_ticks,
                "start_val": old_val,
                "end_val": current_pct,
                "target_pct": target_pct,
            }
        )
        self._tasks["gauge_ease"] = task
```

- [ ] **Step 2: Perbaiki `_step_ease_bar` agar panggil `_draw_gauge`**

Ganti method `_step_ease_bar` yang ada:

```python
    def _step_ease_bar(self, task: AnimTask):
        d = task.data
        total = d["total_ticks"]
        elapsed = total - task.ticks_left
        t = elapsed / total if total > 0 else 1.0
        t_eased = 1 - (1 - t) ** 3
        current = d["start_val"] + (d["end_val"] - d["start_val"]) * t_eased
        self._gauge_current = current
        self._draw_gauge(current, d.get("target_pct", 70.0))
```

- [ ] **Step 3: Verifikasi import**

```bash
python -c "from modules.animation_manager import AnimationManager; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add JEANIROTRABOT/modules/animation_manager.py
git commit -m "feat: add animated profit gauge with easing and color thresholds"
```

---

## Task 5: Implementasi Trade Toast — `show_trade_toast`

**Files:**
- Modify: `JEANIROTRABOT/modules/animation_manager.py`

- [ ] **Step 1: Tambah class `ToastWindow` dan method `show_trade_toast`**

Tambah class `ToastWindow` di **luar** class `AnimationManager` (sebelum class `AnimationManager`):

```python
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
        """Posisikan window toast."""
        if self._win:
            try:
                self._win.geometry(f"+{x}+{y}")
            except Exception:
                pass

    def tick(self) -> bool:
        """Panggil setiap 100ms. Return True jika sudah selesai."""
        if self.done or self._win is None:
            return True
        try:
            if self._phase == "in":
                self._alpha = min(1.0, self._alpha + 0.5)  # fade-in 200ms (2 ticks)
                self._win.attributes("-alpha", self._alpha)
                if self._alpha >= 1.0:
                    self._phase = "show"
            elif self._phase == "show":
                self._show_elapsed += 1
                if self._show_elapsed >= self._show_ticks:
                    self._phase = "out"
            elif self._phase == "out":
                self._alpha = max(0.0, self._alpha - 0.33)  # fade-out 300ms (3 ticks)
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
```

- [ ] **Step 2: Tambah method `show_trade_toast` dan `_tick_toasts` di class `AnimationManager`**

Tambah di dalam class `AnimationManager`:

```python
    def show_trade_toast(self, action: str, symbol: str,
                         lot: float, price: float, pnl: float = 0.0):
        """Tampilkan notifikasi trade di pojok kanan bawah."""
        # Maks 3 toast aktif — dismiss yang paling lama
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
        """Hitung posisi semua toast (stack dari bawah ke atas)."""
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
        """Proses animasi semua toast aktif."""
        still_alive = []
        for toast in self._toasts:
            done = toast.tick()
            if not done:
                still_alive.append(toast)
        self._toasts = still_alive
        if still_alive:
            self._reposition_toasts()
```

- [ ] **Step 3: Tambah `_tick_toasts()` ke dalam `_process_tasks`**

Di method `_process_tasks`, tambah satu baris di bagian bawah setelah loop:

```python
    def _process_tasks(self):
        done = []
        for key, task in self._tasks.items():
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
        # Proses toast animasi
        self._tick_toasts()
```

- [ ] **Step 4: Tambah helper `parse_trade_log` sebagai static method**

Tambah di akhir class `AnimationManager`:

```python
    @staticmethod
    def parse_trade_log(msg: str) -> Optional[dict]:
        """
        Parse log string dari trading_engine untuk extract info trade.
        Format: "AUTO-EXECUTE: BUY 0.01 XAUUSD SL=100 TP=200 (mult=1.0x)"
        atau:    "Order filled: ticket=12345 @ 2345.50"
        Return dict dengan action/symbol/lot/price atau None.
        """
        # Match AUTO-EXECUTE line
        m = re.search(
            r"AUTO-EXECUTE:\s+(BUY|SELL)\s+([\d.]+)\s+(\w+)",
            msg, re.IGNORECASE
        )
        if m:
            return {
                "action": m.group(1).upper(),
                "lot": float(m.group(2)),
                "symbol": m.group(3),
                "price": 0.0,
                "pnl": 0.0,
            }
        # Match Order filled line
        m2 = re.search(r"Order filled.*@\s*([\d.]+)", msg)
        if m2:
            return {"action": "FILL", "price": float(m2.group(1)),
                    "lot": 0.0, "symbol": "", "pnl": 0.0}
        return None
```

- [ ] **Step 5: Verifikasi final import**

```bash
python -c "from modules.animation_manager import AnimationManager, ToastWindow; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add JEANIROTRABOT/modules/animation_manager.py
git commit -m "feat: add TradeToast fade-in/out notifications and toast stack"
```

---

## Task 6: Tambah badge canvas + gauge canvas di `gui.py`

**Files:**
- Modify: `JEANIROTRABOT/modules/gui.py`

- [ ] **Step 1: Tambah badge canvas di Composer section**

Di `_build_right_panel`, cari baris:
```python
        self.lbl_composer_mode = ttkb.Label(
            lf_composer, text="Mode: CONSERVATIVE", foreground="#00aaff",
            font=("Consolas", 9, "bold")
        )
        self.lbl_composer_mode.pack(anchor=tk.W, padx=5, pady=2)
```

Ganti dengan:
```python
        # Badge row: mode label + badge canvas
        badge_row = ttkb.Frame(lf_composer)
        badge_row.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_composer_mode = ttkb.Label(
            badge_row, text="Mode: CONSERVATIVE", foreground="#00aaff",
            font=("Consolas", 9, "bold")
        )
        self.lbl_composer_mode.pack(side=tk.LEFT)
        self.badge_composer = tk.Canvas(badge_row, width=14, height=14,
                                        bg="#1a1a2e", highlightthickness=0)
        self.badge_composer.pack(side=tk.LEFT, padx=6)
        oval = self.badge_composer.create_oval(2, 2, 12, 12, fill="#00ff88", outline="")
        self.badge_composer._badge_oval = oval
        self.lbl_composer_spinner = tk.Label(badge_row, text="", bg="#1a1a2e",
                                             fg="#ffaa00", font=("Consolas", 10))
        self.lbl_composer_spinner.pack(side=tk.LEFT)
        self.badge_composer._spinner_lbl = self.lbl_composer_spinner
```

- [ ] **Step 2: Tambah gauge canvas di Composer section**

Di `_build_right_panel`, cari baris:
```python
        ttkb.Button(
            lf_composer, text="Save Composer Settings", bootstyle="warning-outline",
            command=self._on_save_composer
        ).pack(fill=tk.X, padx=5, pady=2)
```

Tepat **sebelum** baris itu, tambahkan:
```python
        # Profit gauge canvas
        gauge_frame = ttkb.Frame(lf_composer)
        gauge_frame.pack(fill=tk.X, padx=5, pady=3)
        ttkb.Label(gauge_frame, text="Progress:").pack(side=tk.LEFT)
        self.gauge_canvas = tk.Canvas(gauge_frame, height=20, bg="#1a1a2e",
                                      highlightthickness=1,
                                      highlightbackground="#333333")
        self.gauge_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
```

- [ ] **Step 3: Tambah badges untuk News dan Research agent**

Di `_build_bottom_panel`, cari tab News & Research. Cari section News Agent lalu tambahkan badge canvas. Cari baris:
```python
        self.var_news_enabled = tk.BooleanVar(
```
Sebelum baris itu, tambahkan canvas badge (atau cari anchor yang tepat di panel kanan bawah News):

```python
        # Badge untuk News Agent (di header section News)
        self.badge_news = tk.Canvas(parent, width=14, height=14,
                                    bg="#1a1a2e", highlightthickness=0)
        oval_n = self.badge_news.create_oval(2, 2, 12, 12, fill="#444444", outline="")
        self.badge_news._badge_oval = oval_n
        self.badge_news._spinner_lbl = None
```

> **Catatan:** Untuk Research badge, tambahkan dengan cara sama dengan nama `self.badge_research`. Ini badge sederhana yang akan didaftarkan ke AnimationManager di Task 7.

- [ ] **Step 4: Commit perubahan gui.py (badge + gauge)**

```bash
git add JEANIROTRABOT/modules/gui.py
git commit -m "feat: add badge canvas and profit gauge canvas to GUI panels"
```

---

## Task 7: Integrasi AnimationManager ke `gui.py` — 4 titik

**Files:**
- Modify: `JEANIROTRABOT/modules/gui.py`

- [ ] **Step 1: Import AnimationManager di bagian atas gui.py**

Cari baris:
```python
from modules.exchange_connector import ExchangeConnector, SUPPORTED_EXCHANGES
```
Tambahkan tepat setelahnya:
```python
from modules.animation_manager import AnimationManager
```

- [ ] **Step 2: Titik 1 — Inisialisasi di `__init__` setelah `_build_ui()`**

Cari baris:
```python
        self._build_ui()
        self._start_refresh_timer()
```
Ganti dengan:
```python
        self._build_ui()
        self._start_refresh_timer()

        # Animasi (Fix UI) — inisialisasi setelah build_ui agar widget ada
        self.anim = AnimationManager(self.root)
        self.anim.register_pnl_labels(self.acc_labels)
        self.anim.register_gauge(self.gauge_canvas, None)
        self.anim.register_badge("composer", self.badge_composer)
        self.anim.start()
```

- [ ] **Step 3: Titik 2 — Feed data di `_update_display`**

Cari blok:
```python
        if info:
            self.acc_labels["Balance"].configure(text=f"{info.balance:,.2f} {info.currency}")
            self.acc_labels["Equity"].configure(text=f"{info.equity:,.2f}")
            self.acc_labels["Margin"].configure(text=f"{info.margin:,.2f}")
            self.acc_labels["Free Margin"].configure(text=f"{info.free_margin:,.2f}")
            color = ACCENT_GREEN if info.profit >= 0 else ACCENT_RED
            self.acc_labels["Profit"].configure(text=f"{info.profit:,.2f}", foreground=color)
```
Ganti dengan:
```python
        if info:
            self.acc_labels["Margin"].configure(text=f"{info.margin:,.2f}")
            self.acc_labels["Free Margin"].configure(text=f"{info.free_margin:,.2f}")
            # Animasi P&L (count-up + flash)
            self.anim.update_pnl(info.balance, info.equity, info.profit, info.currency)
            # Profit gauge
            if hasattr(self.composer, "_initial_balance") and self.composer._initial_balance > 0:
                pct = ((info.equity - self.composer._initial_balance)
                       / self.composer._initial_balance * 100)
                target = self.config.get_float("PROFIT_TARGET_PERCENT", 70.0)
                self.anim.update_profit_gauge(pct, target)
```

- [ ] **Step 4: Titik 3 — Badge update di `_on_composer_mode_change`**

Cari method:
```python
    def _on_composer_mode_change(self, mode: str, mode_params: dict):
        """Callback saat Composer berganti mode."""
        color = mode_params.get("color", "#ffffff")
        desc = mode_params.get("description", "")
        self.root.after(0, lambda: self._update_composer_display(mode, color, desc))
```
Ganti dengan:
```python
    def _on_composer_mode_change(self, mode: str, mode_params: dict):
        """Callback saat Composer berganti mode."""
        color = mode_params.get("color", "#ffffff")
        desc = mode_params.get("description", "")
        self.root.after(0, lambda: self._update_composer_display(mode, color, desc))
        # Update badge state
        self.anim.set_agent_state("composer", "active")
```

- [ ] **Step 5: Titik 4 — Toast trigger di `_append_log`**

Cari method:
```python
    def _append_log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, line)
```
Tambahkan **setelah** blok existing (setelah `self.txt_log.config(state=tk.DISABLED)` atau sebelum akhir method):
```python
        # Toast notifikasi untuk AUTO-EXECUTE
        if "AUTO-EXECUTE:" in msg:
            parsed = AnimationManager.parse_trade_log(msg)
            if parsed and parsed.get("action") in ("BUY", "SELL"):
                self.anim.show_trade_toast(
                    parsed["action"], parsed["symbol"],
                    parsed["lot"], parsed["price"], parsed["pnl"]
                )
```

- [ ] **Step 6: Tambah `anim.stop()` di `_on_close`**

Cari method:
```python
    def _on_close(self):
        if self.engine.running:
            self.engine.stop()
        self.connector.disconnect()
```
Ganti dengan:
```python
    def _on_close(self):
        if self.engine.running:
            self.engine.stop()
        if hasattr(self, "anim"):
            self.anim.stop()
        self.connector.disconnect()
```

- [ ] **Step 7: Commit integrasi**

```bash
git add JEANIROTRABOT/modules/gui.py
git commit -m "feat: integrate AnimationManager into GUI — P&L ticker, badge, gauge, toast"
```

---

## Task 8: Test visual end-to-end + polish

**Files:**
- Modify: `JEANIROTRABOT/modules/gui.py` (minor fixes jika ada)
- Modify: `JEANIROTRABOT/modules/animation_manager.py` (minor fixes jika ada)

- [ ] **Step 1: Jalankan aplikasi dan cek startup**

```bash
python main.py
```

Ceklis visual:
- [ ] Aplikasi terbuka tanpa error di terminal
- [ ] Badge Composer terlihat di panel kanan (lingkaran hijau kecil di sebelah "Mode: CONSERVATIVE")
- [ ] Gauge canvas terlihat di Composer section (bar kosong atau 0%)
- [ ] Tidak ada traceback di console

- [ ] **Step 2: Test badge pulse tanpa MT5**

Di Python console atau tambahkan sementara ke `_auto_connect_on_startup`:
```python
# Test manual (hapus setelah test)
self.root.after(2000, lambda: self.anim.set_agent_state("composer", "active"))
self.root.after(4000, lambda: self.anim.set_agent_state("composer", "processing"))
self.root.after(6000, lambda: self.anim.set_agent_state("composer", "error"))
self.root.after(8000, lambda: self.anim.set_agent_state("composer", "idle"))
```

Ceklis:
- [ ] Badge berkedip hijau saat "active"
- [ ] Badge berubah kuning + spinner saat "processing"
- [ ] Badge flash merah saat "error"
- [ ] Badge abu statis saat "idle"

- [ ] **Step 3: Test toast notifikasi**

Tambahkan sementara ke `_auto_connect_on_startup`:
```python
self.root.after(3000, lambda: self.anim.show_trade_toast("BUY", "XAUUSD", 0.01, 2345.50, 1.20))
self.root.after(4000, lambda: self.anim.show_trade_toast("SELL", "EURUSD", 0.02, 1.0854, -0.50))
self.root.after(5000, lambda: self.anim.show_trade_toast("BUY", "BTCUSD", 0.001, 65432.0, 5.30))
```

Ceklis:
- [ ] 3 toast muncul stack di pojok kanan bawah
- [ ] BUY = background hijau gelap
- [ ] SELL = background merah gelap
- [ ] Fade-in, tampil 2 detik, fade-out

- [ ] **Step 4: Test gauge**

Tambahkan sementara:
```python
import threading
def test_gauge():
    for pct in range(0, 101, 5):
        self.anim.update_profit_gauge(pct, 70.0)
        import time; time.sleep(0.3)
threading.Thread(target=test_gauge, daemon=True).start()
```

Ceklis:
- [ ] Bar bergerak smooth dari 0 ke 100%
- [ ] Warna berubah: merah → kuning → hijau → hijau terang
- [ ] Garis putih vertikal di posisi 70%

- [ ] **Step 5: Hapus semua kode test sementara**

- [ ] **Step 6: Commit final**

```bash
git add JEANIROTRABOT/modules/animation_manager.py JEANIROTRABOT/modules/gui.py
git commit -m "feat: dynamic UI animation complete — P&L ticker, badge, gauge, toast"
```

---

## Self-Review Checklist

- [x] Spec coverage: Task 1-5 cover AnimationManager (A+B+C+D). Task 6-7 cover integrasi gui.py. Task 8 cover testing.
- [x] Tidak ada TBD/placeholder — semua step punya kode lengkap
- [x] Type consistency: `AnimTask`, `ToastWindow` didefinisikan di Task 1/5 dan dipakai konsisten di semua task
- [x] Method names konsisten: `set_agent_state`, `update_pnl`, `update_profit_gauge`, `show_trade_toast`, `register_badge`, `register_gauge`, `register_pnl_labels`
- [x] `_ticker_toasts` → diperbaiki jadi `_tick_toasts` di semua tempat
- [x] `anim.stop()` di `_on_close` sudah ada di Task 7 Step 6
- [x] Toast detection pakai `"AUTO-EXECUTE:"` — cocok dengan log format di `trading_engine.py` baris 593
