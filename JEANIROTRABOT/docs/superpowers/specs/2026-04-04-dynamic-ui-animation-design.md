# JEANIROTRABOT — Dynamic UI Animation Design

**Date:** 2026-04-04
**Status:** Approved
**Scope:** Tambahkan animasi full interaktif ke GUI tanpa mengubah fitur yang sudah ada.

---

## Tujuan

Membuat GUI JEANIROTRABOT lebih dinamis dan interaktif dengan 4 komponen animasi:
- Live P&L Ticker (count-up angka)
- Agent Status Badge (pulse/spinner)
- Profit Target Gauge (animated progress bar)
- Trade Toast Notification (fade in/out)

Semua fitur trading, AI agent, dan koneksi **tidak berubah**.

---

## Arsitektur

### Pendekatan: `AnimationManager` Terpusat

File baru: `modules/animation_manager.py`

Satu class `AnimationManager` mengelola semua animasi via **satu loop `root.after(100ms)`**. Setiap efek didaftarkan sebagai `AnimTask` dengan state internal. Di setiap tick, manager iterasi semua task aktif dan update widget yang terdaftar.

```
AnimationManager
├── _tasks: dict[str, AnimTask]     # semua efek aktif
├── _toasts: list[ToastWindow]      # stack toast aktif (maks 3)
├── tick()                          # root.after(100) loop utama
├── update_pnl(balance, equity, profit, currency)
├── set_agent_state(name, state)    # "idle" | "active" | "processing" | "error"
├── update_profit_gauge(current_pct, target_pct)
└── show_trade_toast(action, symbol, lot, price, pnl)
```

**Keuntungan:**
- Satu timer untuk semua animasi → CPU ringan
- Tidak ada thread baru → tidak bisa crash GUI
- Mudah di-stop: `anim.stop()` di `_on_close()`

---

## Komponen Animasi Detail

### A. Live P&L Ticker — `CountUpLabel`

**Lokasi:** Panel kiri, section "Account Info" — label Balance, Equity, Profit

**Perilaku:**
- Saat nilai berubah, angka "count-up" dari nilai lama ke baru dalam **500ms, 10 step**
- Flash putih 1x saat perubahan > $1 (atau > 0.1% dari nilai)
- Warna dinamis:
  - Profit > 0 → `#00ff88` (hijau)
  - Profit < 0 → `#ff4444` (merah)
  - Profit = 0 → `#888888` (abu)
- Balance & Equity: selalu `#ffffff`, flash kuning saat naik

**Implementasi:** `AnimTask` type `COUNT_UP` menyimpan `start_val`, `end_val`, `steps_remaining`, `widget_ref`

---

### B. Agent Status Badge — `PulsingBadge`

**Lokasi:** Panel kanan — di sebelah label "Composer Agent", "News Agent", "Research Agent"

**State & Visual:**

| State | Warna | Animasi |
|-------|-------|---------|
| `idle` | `#444444` abu | Statis |
| `active` | `#00ff88` hijau | Pulse pelan 800ms (terang→redup→terang) |
| `processing` | `#ffaa00` kuning | Spinner karakter `⟳→↻` ganti frame 150ms |
| `error` | `#ff4444` merah | Flash cepat 200ms selama 2 detik |

**Implementasi:** Canvas oval 12x12px per badge. `AnimTask` type `PULSE` atau `SPINNER`.

---

### C. Profit Target Gauge — `AnimatedProgressBar`

**Lokasi:** Panel kanan, Composer Agent section — di bawah label "Target Profit"

**Perilaku:**
- `ttk.Progressbar` + overlay label teks `"X.X% / 70%"`
- Nilai bergerak smooth dengan **easing** (tidak langsung lompat)
- Warna bar berubah otomatis sesuai progress:

| Progress | Warna |
|----------|-------|
| 0–30% | `danger` merah |
| 30–60% | `warning` kuning |
| 60–90% | `success` hijau |
| ≥90% | `success` + pulse hijau terang |

- Target line (garis vertikal tipis di posisi 70%) sebagai referensi visual

**Implementasi:** `tk.Canvas` custom width=200, height=20. Draw rect + overlay text. `AnimTask` type `EASE_BAR`.

---

### D. Trade Toast Notification

**Lokasi:** Pojok kanan bawah window, stack vertikal ke atas

**Perilaku:**
- Muncul setiap BUY/SELL dieksekusi (deteksi dari log "Order filled")
- Animasi: fade-in 200ms → tampil 2 detik → fade-out 300ms
- Warna background:
  - BUY → `#0a3d1f` hijau gelap + border `#00ff88`
  - SELL → `#3d0a0a` merah gelap + border `#ff4444`
- Konten: `[BUY] XAUUSD  0.01 lot @ 2345.50  +$1.20`
- Maksimal **3 toast** stack sekaligus (yang paling lama di-dismiss dulu)

**Implementasi:** `tk.Toplevel` tanpa border (`overrideredirect=True`), posisi dihitung dari `root.winfo_x/y/width/height`. Opacity via `wm_attributes("-alpha", value)`.

---

## Integrasi ke `gui.py`

Hanya **4 titik perubahan** di `gui.py`, tidak ada perubahan di file lain:

### 1. `__init__` — Inisialisasi AnimationManager
```python
from modules.animation_manager import AnimationManager
# ... setelah self.root dibuat dan self._build_ui() dipanggil:
self.anim = AnimationManager(self.root)
self.anim.register_pnl_labels(self.acc_labels)
self.anim.register_composer_gauge(self.profit_gauge_canvas)
self.anim.start()
```

### 2. `_update_display()` — Feed data ke animator
```python
if info:
    self.anim.update_pnl(info.balance, info.equity, info.profit, info.currency)
    # Hitung profit % dari initial balance
    if self.composer._initial_balance > 0:
        pct = ((info.equity - self.composer._initial_balance)
               / self.composer._initial_balance * 100)
        target = self.config.get_float("PROFIT_TARGET_PERCENT", 70.0)
        self.anim.update_profit_gauge(pct, target)
```

### 3. `_on_composer_mode_change()` — Update badge state
```python
def _on_composer_mode_change(self, mode: str):
    self.anim.set_agent_state("composer", "active")
    # ... existing code
```

### 4. `_append_log()` — Deteksi trade untuk toast
```python
def _append_log(self, msg: str):
    # ... existing scroll text code ...
    if "Order filled" in msg or "AUTO-EXECUTE" in msg:
        self._trigger_trade_toast(msg)
```

---

## File yang Dibuat/Diubah

| File | Aksi | Keterangan |
|------|------|-----------|
| `modules/animation_manager.py` | **BARU** | Semua logika animasi |
| `modules/gui.py` | **EDIT** | 4 titik integrasi + tambah badge widget |

**Tidak ada perubahan pada:**
- `trading_engine.py`
- `composer_agent.py`
- `ai_agent.py`
- `config.py`
- Semua modul lainnya

---

## Error Handling

- Semua animasi wrapped `try/except` — jika widget sudah dihapus (app closing), task otomatis di-cancel
- `anim.stop()` dipanggil di `_on_close()` untuk bersihkan semua timer
- Toast window di-destroy jika parent root closing

---

## Testing

```bash
# Jalankan aplikasi
python main.py

# Test tanpa MT5 (visual check):
# - Badge Composer berkedip hijau jika COMPOSER_ENABLED=true
# - Progress gauge terlihat di panel Composer
# - Simulasi toast via console: anim.show_trade_toast("BUY","XAUUSD",0.01,2345.50,1.20)

# Test dengan MT5:
# - Balance/Equity count-up saat posisi berubah
# - Toast muncul setiap order filled
# - Badge spinner saat Composer sedang call AI
```
