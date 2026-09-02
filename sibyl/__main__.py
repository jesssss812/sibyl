"""``python -m sibyl`` — the single entry point for a forecasting run.

The template this fork descends from ran through a top-level ``main.py``. A
``__main__`` module replaces it so the package is the entry point: there is no
importable-vs-runnable split to keep in sync, and the command works from any
directory without depending on the checkout being the working directory.
"""

from sibyl.cli import main

if __name__ == "__main__":
    main()
