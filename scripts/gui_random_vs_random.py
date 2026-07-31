"""Launch the desktop GUI to watch random-vs-random self-play live.

Usage:
    python scripts/gui_random_vs_random.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.app import ChessWatcherApp


def main():
    app = ChessWatcherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
