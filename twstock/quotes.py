"""模組進入點：``python -m twstock.quotes``（等同 ``python -m twstock.cli quotes``）。

實作在 :mod:`twstock.data.quotes`；此檔案只提供穩定的模組路徑，避免
把 CLI 綁在資料層的檔案位置上。
"""
from __future__ import annotations

import sys

from .data.quotes import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
