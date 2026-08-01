"""tests/conftest.py —— pytest 全局配置。

确保项目根目录在 sys.path 中，以便正确导入 src 包。
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
