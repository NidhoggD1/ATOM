# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM determinism test harness.

Modules
-------
- ``compare``  Pure-Python metrics + merge CLI:
               ``python -m atom.utils.determinism.compare run*.json``.
- ``d1``       Offline-engine D1 (run-to-run) runner:
               ``python -m atom.utils.determinism.d1 --model ... --stage d1.0``.

``compare`` deliberately imports neither torch nor any ATOM engine module, so
its metrics are unit-tested on CPU in ``tests/test_determinism_compare.py``.

Methodology and the D1 stage ladder: ``docs/determinism_testing.md``.
"""

from atom.utils.determinism.compare import (
    D1Report,
    PromptVerdict,
    RunSample,
    build_report,
    compare_repeats,
    diagnose,
    load_run,
    render_text,
    report_to_dict,
    save_run,
)

__all__ = [
    "D1Report",
    "PromptVerdict",
    "RunSample",
    "build_report",
    "compare_repeats",
    "diagnose",
    "load_run",
    "render_text",
    "report_to_dict",
    "save_run",
]
