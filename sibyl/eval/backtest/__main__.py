"""Entry point so `python -m sibyl.eval.backtest` runs the backtest driver.

Mirrors `sibyl/eval/performance/__main__.py`, which is the convention this
package was the odd one out against: `ablation/` and `performance_analysis/` already kept
their drivers inside the package while `backtest.py` sat at the repo root purely because
the Metaculus starter template put it there.
"""

from sibyl.eval.backtest.cli import main

if __name__ == "__main__":
    main()
