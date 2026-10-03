"""Run two offline fixture ensembles against one frozen local input set."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from sibyl.eval.backtest.cli import _build_parser, run_backtest
from sibyl.eval.compare import render_report


async def main() -> None:
    root = Path("backtests/fixture_aa") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = root / "manifest.json"
    parser = _build_parser()
    await run_backtest(parser.parse_args(["--fixture", "--num-questions", "8", "--prepare-only", str(manifest)]))
    paths = []
    for arm in ("a", "b"):
        folder = root / arm
        await run_backtest(parser.parse_args(["--fixture", "--manifest", str(manifest), "--output-dir", str(folder)]))
        paths.append(next(folder.glob("backtest_data_*.json")))
    report = render_report(*paths)
    (root / "aa_comparison.md").write_text(report)
    print(f"Fixture A/A results: {root}")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
