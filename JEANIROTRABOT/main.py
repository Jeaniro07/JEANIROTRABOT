"""
JEANIROTRABOT - Robot Trading Otomatis untuk Windows
Plug & Play: jalankan .exe, isi credentials di GUI, langsung trading.
"""

import sys
import os
import logging
from pathlib import Path

# App directory (works for both script and frozen .exe)
if getattr(sys, 'frozen', False):
    APP_DIR = Path(sys.executable).parent
else:
    APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Setup logging
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "jeanirotrabot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("JEANIROTRABOT")


def main():
    logger.info("=== JEANIROTRABOT Starting ===")

    from modules.gui import JeaniroTrabotApp
    app = JeaniroTrabotApp()
    app.run()

    logger.info("=== JEANIROTRABOT Exiting ===")


if __name__ == "__main__":
    main()
