# drawdown-constrained-portfolio-sim

Drawdown-constrained portfolio sims for Polymarket (median target at max-DD bound): ensemble-zone logistic, Kelly, mean-reversion sizing.

## Why it exists

These are offline research simulators that answer one question for the Polymarket 5-minute crypto up/down book: **starting from a $9 bankroll, what is the maximum achievable MEDIAN per-market PnL while keeping max cumulative drawdown under $5?** Each script attacks that drawdown-constrained target with a different sizing/selection family, sweeps its config grid, and reports the honest Pareto frontier (most families prove the +$1 median is structurally infeasible at real coverage — the book is efficient, so cheap entries that pay multiples come only with ~50% win rate).

All sims are **leak-free, real-time first-trigger**: per market, ticks are scanned `cd` high→low; the first qualifying tick is the fire; only ticks at `cd >= fire_cd` feed features; the winner is used *only* to score PnL at settle. Drawdown distributions come from Monte-Carlo shuffles (2000 runs) of fired-market order.

## What's inside

| Script | Family | What it does |
| --- | --- | --- |
| `sim_target_meanrev_sized.py` | meanrev-sized | The core validated edge: fade the side that fell over the last 20s in the 0.40–0.60 coin-flip zone. Sweeps move-threshold, cd-window, tight-spread gate, and $1–$3 sizing. |
| `sim_target_kelly_dd_constrained.py` | kelly-dd-constrained | Fixes that mean-rev edge and studies pure sizing math: fractional-Kelly stake with a convex drawdown-throttle that hard-clamps `stake <= headroom`, making maxDD < budget a guarantee (not a hope). |
| `sim_target_ensemble_zone_logistic.py` | ensemble-zone-logistic | Fits an offline, leak-free L2 logistic over 4 orthogonal in-zone signals (crowd, 30s slope, velocity, tight-spread) + level term; deterministic slug-hash TRAIN/TEST split (~50/50); fires on model confidence. Requires `numpy`. |
| `sim_target_tight_spread_highconf.py` | tight-spread-highconf | Rides the favored (higher-mid) side only when both spreads are tight inside a cd-window, sweeping spread/mid floors and $1–$5 size for high win-rate. |
| `sim_target_existing_cheapEV_portfolio.py` | existing-cheapEV-portfolio | Parses the bot's own SHADOW recap lines from replay logs, ranks positive-EV cheap (entry ≤ $0.50, cd > 15) strategy-sides, then searches baskets × per-market/side caps × fire size for the best DD-constrained median. |

Each script prints a tuning table + best/Pareto config and writes a machine-readable summary JSON into the data directory (e.g. `.meanrev_sized_summary.json`, `.kelly_dd_constrained_summary.json`, `.ensemble_zone_logistic_summary.json`, `sim_target_existing_cheapEV_portfolio_results.json`).

## Requirements

- Python 3.8+
- `numpy` (only `sim_target_ensemble_zone_logistic.py`); the other four are stdlib-only (`json`, `re`, `glob`, `random`, `statistics`, `math`, `hashlib`).
- No wallet, key, or network access — these are pure offline simulators and handle **no real funds**.

```bash
python3 -m pip install numpy
```

## Usage

Each script is a no-argument entry point; just run it:

```bash
python3 sim_target_meanrev_sized.py
python3 sim_target_kelly_dd_constrained.py
python3 sim_target_ensemble_zone_logistic.py
python3 sim_target_tight_spread_highconf.py
python3 sim_target_existing_cheapEV_portfolio.py
```

## Data

These sims read inputs from the **private `polymarket-data` repo** (paths are currently hard-coded to `/home/polybot/polymarket-bot/data/`):

- Four families load `market_panel.json` (per-market chronological tick panels with winners).
- `sim_target_existing_cheapEV_portfolio.py` instead parses `race_test_btc-updown-5m-*.log` replay logs from `quant_bots_logs_replay/`.

Place those files at the expected paths (or edit the `PANEL_PATH` / `LOG_DIR` constants) before running. Output summary JSONs are written back into `data/` and are git-ignored.

> Private research software. No warranty; trades/handles real funds at your own risk.
