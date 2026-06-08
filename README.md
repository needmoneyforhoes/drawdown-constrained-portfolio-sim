# drawdown-constrained-portfolio-sim

Offline simulators for the Polymarket 5-minute crypto up/down book. Each one searches for the maximum median per-market PnL from a $9 bankroll while holding max cumulative drawdown under $5, using a different sizing or selection family.

Sims are leak-free and real-time first-trigger: per market, ticks scan `cd` high to low, the first qualifying tick is the fire, only ticks at `cd >= fire_cd` feed features, and the winner is used only to score PnL at settle. Drawdown distributions come from 2000-run Monte-Carlo shuffles of fired-market order.

## Scripts

- `sim_target_meanrev_sized.py`: fades the side that fell over the last 20s in the 0.40-0.60 zone. Sweeps move-threshold, cd-window, tight-spread gate, and $1-$3 sizing.
- `sim_target_kelly_dd_constrained.py`: fixes the mean-rev edge and studies sizing. Fractional-Kelly stake with a convex drawdown-throttle that clamps `stake <= headroom`, so maxDD stays under budget.
- `sim_target_ensemble_zone_logistic.py`: fits an offline L2 logistic over 4 in-zone signals (crowd, 30s slope, velocity, tight-spread) plus a level term. Deterministic slug-hash TRAIN/TEST split (~50/50); fires on model confidence. Needs numpy.
- `sim_target_tight_spread_highconf.py`: rides the higher-mid side when both spreads are tight inside a cd-window. Sweeps spread/mid floors and $1-$5 size.
- `sim_target_existing_cheapEV_portfolio.py`: parses the bot's SHADOW recap lines from replay logs, ranks positive-EV cheap strategy-sides (entry <= $0.50, cd > 15), then searches baskets, per-market/side caps, and fire size.

Each script prints a tuning table and best/Pareto config, then writes a summary JSON to the data dir: `.meanrev_sized_summary.json`, `.kelly_dd_constrained_summary.json`, `.ensemble_zone_logistic_summary.json`, `sim_target_existing_cheapEV_portfolio_results.json`.

## Requirements

Python 3.8+. Only `sim_target_ensemble_zone_logistic.py` needs numpy; the rest are stdlib-only.

```bash
python3 -m pip install numpy
```

## Usage

No arguments; run each directly.

```bash
python3 sim_target_meanrev_sized.py
python3 sim_target_kelly_dd_constrained.py
python3 sim_target_ensemble_zone_logistic.py
python3 sim_target_tight_spread_highconf.py
python3 sim_target_existing_cheapEV_portfolio.py
```

## Data

Reads `$DATA_DIR` from the private polymarket-data repo. Four families load `market_panel.json` (per-market tick panels with winners). `sim_target_existing_cheapEV_portfolio.py` parses `race_test_btc-updown-5m-*.log` from `quant_bots_logs_replay/`. Paths are hard-coded to `./data`; edit `PANEL_PATH` / `LOG_DIR` to relocate. Output JSONs write back to `./data` and are git-ignored.

Offline only; no wallet, key, or network access, and no real funds.
