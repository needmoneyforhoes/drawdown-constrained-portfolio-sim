#!/usr/bin/env python3
"""
sim_target_existing_cheapEV_portfolio.py

FAMILY: existing-cheapEV-portfolio

Goal: Build a portfolio from the bot's EXISTING positive-EV cheap strategies
(parsed straight from the replay-log SHADOW/FIRE recap lines), restricted to
entry<=$0.50 and cd>15, and find the basket + per-market cap + fire size that
maximizes MEDIAN per-market PnL subject to maxDD<$5 (running balance from $9).

Leak-free / no look-ahead:
  - Each fire is a REAL-TIME first-trigger decision the bot made during replay at
    a stated cd; the market winner is used ONLY to score PnL at settle.
  - We parse (strategy, side, entry_price, cd, won) directly from the recap
    '🔵 SHADOW' lines, where won/lost is the actual settle outcome vs winner=.
  - PnL for a $1 fire on a cheap binary @ entry p:
        win  -> (1/p - 1) * fire_dollars
        loss -> -1        * fire_dollars
    (Polymarket pays $1/share; entry p buys 1/p shares per $1. min fire $1.)

Portfolio model:
  - Pick a basket of N strategies.
  - For each market (chronological order, as logged), collect that market's fires
    from the basket strategies that satisfy entry<=ENTRY_MAX and cd>CD_MIN.
  - Apply per-market cap (max fires/market) and per-side cap if configured.
  - Each selected fire bets FIRE_DOLLARS. Sum PnL = market PnL.
  - Running balance starts at $9; track peak-to-trough drawdown.
  - Median is the median of PER-MARKET PnL across fired markets.

Monte Carlo:
  - Markets are time-ordered, but a single fixed ordering's drawdown is fragile.
    We bootstrap-shuffle the per-market PnL vector many times to get a robust
    distribution of maxDD (and report the realized chronological maxDD too, and
    a conservative high-percentile maxDD from the MC).
"""

import os
import re
import glob
import json
import random
import statistics
from collections import defaultdict, Counter

LOG_DIR = "/home/polybot/polymarket-bot/quant_bots_logs_replay"
LOG_GLOB = os.path.join(LOG_DIR, "race_test_btc-updown-5m-*.log")

START_BALANCE = 9.0
ENTRY_MAX = 0.50          # cheap-only
CD_MIN = 15               # exclude last 15s (unfillable) -> require cd > 15
MC_ITERS = 2000
RNG_SEED = 12345

# ---- regexes -------------------------------------------------------------
RE_RECAP = re.compile(r"MARKET RECAP\s+—\s+(\S+)\s+\(winner=([A-Z]+)\)")
# recap shadow line:
#   1. depth_stable_mid       🔵 SHADOW UP @ $0.360 5sh cd=285  real=$0.00  hypo=$-1.85 ✅ [settle]
RE_SHADOW = re.compile(
    r"\d+\.\s+([A-Za-z0-9_]+)\s+🔵 SHADOW\s+(UP|DN)\s+@\s+\$([0-9.]+)\s+[\d.]+sh\s+cd=(\d+)"
)
RE_STATUS_WIN = "✅"
RE_STATUS_LOSS = "❌"


def parse_logs():
    """Return list of markets in log order:
       [{slug, winner, fires:[{strat, side, entry, cd, won}]}]
       won is None when no settle status emoji present (excluded)."""
    markets = []
    files = sorted(glob.glob(LOG_GLOB))
    for fp in files:
        cur = None
        with open(fp, "r", errors="ignore") as fh:
            for line in fh:
                mr = RE_RECAP.search(line)
                if mr:
                    if cur is not None:
                        markets.append(cur)
                    cur = {"slug": mr.group(1), "winner": mr.group(2), "fires": []}
                    continue
                if cur is None:
                    continue
                if "🔵 SHADOW" not in line:
                    continue
                ms = RE_SHADOW.search(line)
                if not ms:
                    continue
                strat = ms.group(1)
                side = ms.group(2)
                entry = float(ms.group(3))
                cd = int(ms.group(4))
                if RE_STATUS_WIN in line:
                    won = True
                elif RE_STATUS_LOSS in line:
                    won = False
                else:
                    won = None  # excluded / SL-only-no-status -> skip later
                # cross-check won vs winner header (should agree for [settle] holds)
                cur["fires"].append({
                    "strat": strat, "side": side, "entry": entry,
                    "cd": cd, "won": won,
                })
        if cur is not None:
            markets.append(cur)
    return markets


def fire_pnl(entry, won, fire_dollars):
    if won:
        return (1.0 / entry - 1.0) * fire_dollars
    return -1.0 * fire_dollars


def strat_universe(markets):
    """Per-strategy stats over cheap, cd>15, status-known fires. Used to rank
       which strategies are positive-EV (the 'existing positive-EV cheap' set)."""
    agg = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "mkts": set()})
    for mi, m in enumerate(markets):
        for f in m["fires"]:
            if f["won"] is None:
                continue
            if f["entry"] > ENTRY_MAX:
                continue
            if f["cd"] <= CD_MIN:
                continue
            key = (f["strat"], f["side"])
            a = agg[key]
            a["n"] += 1
            a["wins"] += 1 if f["won"] else 0
            a["pnl"] += fire_pnl(f["entry"], f["won"], 1.0)
            a["mkts"].add(mi)
    rows = []
    for (strat, side), a in agg.items():
        n = a["n"]
        rows.append({
            "key": (strat, side),
            "strat": strat, "side": side,
            "n": n,
            "wr": a["wins"] / n if n else 0.0,
            "ev": a["pnl"] / n if n else 0.0,
            "total": a["pnl"],
            "nmkts": len(a["mkts"]),
        })
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows


def simulate(markets, basket_keys, fire_dollars, per_mkt_cap, per_side_cap=None,
             cd_min=CD_MIN, entry_max=ENTRY_MAX):
    """Return per-market PnL list (only for markets that fired), chronological."""
    bset = set(basket_keys)
    per_mkt_pnls = []
    fired_mkts = 0
    total_mkts = len(markets)
    for m in markets:
        # collect candidate fires from basket, sorted by cd DESC (earliest first
        # = real first-trigger order within the market)
        cands = []
        for f in m["fires"]:
            if f["won"] is None:
                continue
            if (f["strat"], f["side"]) not in bset:
                continue
            if f["entry"] > entry_max:
                continue
            if f["cd"] <= cd_min:
                continue
            cands.append(f)
        if not cands:
            continue
        cands.sort(key=lambda x: -x["cd"])  # earliest (high cd) first
        # dedup: one fire per (strat,side) per market (first-trigger only)
        seen = set()
        chosen = []
        side_count = Counter()
        for f in cands:
            k = (f["strat"], f["side"])
            if k in seen:
                continue
            if per_side_cap is not None and side_count[f["side"]] >= per_side_cap:
                continue
            seen.add(k)
            side_count[f["side"]] += 1
            chosen.append(f)
            if len(chosen) >= per_mkt_cap:
                break
        if not chosen:
            continue
        pnl = sum(fire_pnl(f["entry"], f["won"], fire_dollars) for f in chosen)
        per_mkt_pnls.append(pnl)
        fired_mkts += 1
    return per_mkt_pnls, fired_mkts, total_mkts


def max_drawdown(pnls, start=START_BALANCE):
    bal = start
    peak = start
    mdd = 0.0
    for p in pnls:
        bal += p
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > mdd:
            mdd = dd
    return mdd


def wipe_check(pnls, start=START_BALANCE):
    """fraction of MC paths that hit balance <= 0 at any point."""
    bal = start
    for p in pnls:
        bal += p
        if bal <= 0:
            return True
    return False


def mc_metrics(per_mkt_pnls, iters=MC_ITERS, seed=RNG_SEED):
    """Bootstrap-shuffle the per-market PnL vector. Return dict of robust stats."""
    rng = random.Random(seed)
    n = len(per_mkt_pnls)
    if n == 0:
        return {"p50_dd": float("inf"), "p95_dd": float("inf"),
                "max_dd_mc": float("inf"), "wipe_rate": 100.0,
                "chrono_dd": float("inf")}
    chrono_dd = max_drawdown(per_mkt_pnls)
    dds = []
    wipes = 0
    base = list(per_mkt_pnls)
    for _ in range(iters):
        rng.shuffle(base)
        dds.append(max_drawdown(base))
        if wipe_check(base):
            wipes += 1
    dds.sort()
    p50 = dds[int(0.50 * (len(dds) - 1))]
    p95 = dds[int(0.95 * (len(dds) - 1))]
    return {
        "p50_dd": p50,
        "p95_dd": p95,
        "max_dd_mc": dds[-1],
        "wipe_rate": 100.0 * wipes / iters,
        "chrono_dd": chrono_dd,
    }


def main():
    print("Parsing logs ...")
    markets = parse_logs()
    print(f"  parsed {len(markets)} market recaps")

    uni = strat_universe(markets)
    pos = [r for r in uni if r["ev"] > 0 and r["n"] >= 8]
    print(f"\nTop positive-EV cheap (entry<=${ENTRY_MAX}, cd>{CD_MIN}, n>=8) strategy-sides:")
    for r in pos[:25]:
        print(f"  {r['strat']:<28} {r['side']}  n={r['n']:>4}  wr={r['wr']:.3f}  "
              f"ev/$={r['ev']:+.4f}  total=${r['total']:+.2f}  mkts={r['nmkts']}")

    # candidate basket pool = positive EV strategy-sides (ranked by total)
    pool = [r["key"] for r in pos]

    # ---- search over basket size, fire size, caps -------------------------
    best = None
    results = []
    fire_sizes = [1.0, 2.0, 3.0]
    caps = [1, 2, 3, 4, 5, 8]
    side_caps = [None, 1, 2]
    basket_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, len(pool)]

    for bn in basket_sizes:
        basket = pool[:bn]
        for fd in fire_sizes:
            for cap in caps:
                for sc in side_caps:
                    pnls, fired, total = simulate(
                        markets, basket, fd, cap, per_side_cap=sc)
                    if fired < 30:
                        continue
                    med = statistics.median(pnls)
                    mean = statistics.mean(pnls)
                    mc = mc_metrics(pnls)
                    rec = {
                        "basket_size": bn,
                        "basket": [f"{k[0]}|{k[1]}" for k in basket],
                        "fire_dollars": fd,
                        "per_mkt_cap": cap,
                        "per_side_cap": sc,
                        "median": med,
                        "mean": mean,
                        "fired": fired,
                        "total_mkts": total,
                        "pct_fired": 100.0 * fired / total,
                        "chrono_dd": mc["chrono_dd"],
                        "p95_dd": mc["p95_dd"],
                        "max_dd_mc": mc["max_dd_mc"],
                        "wipe_rate": mc["wipe_rate"],
                    }
                    results.append(rec)

    # Primary objective: median >= +1 AND maxDD (use MC p95 as robust DD) < 5
    # Rank by: target_met first, then median desc, then DD asc.
    def meets(rec):
        return rec["median"] >= 1.0 and rec["p95_dd"] < 5.0

    results.sort(key=lambda r: (meets(r), r["median"], -r["p95_dd"]), reverse=True)

    print("\n=== TOP 15 CONFIGS (ranked: target_met, median desc, dd asc) ===")
    for r in results[:15]:
        flag = "✅TARGET" if meets(r) else ""
        print(f"  bsz={r['basket_size']:>3} fire=${r['fire_dollars']:.0f} cap={r['per_mkt_cap']} "
              f"scap={r['per_side_cap']}  med={r['median']:+.3f} mean={r['mean']:+.3f} "
              f"fired={r['fired']}/{r['total_mkts']} ({r['pct_fired']:.0f}%)  "
              f"chronoDD=${r['chrono_dd']:.2f} p95DD=${r['p95_dd']:.2f} "
              f"wipe={r['wipe_rate']:.1f}% {flag}")

    # Best config that meets target; else best median with p95DD<5; else Pareto.
    met = [r for r in results if meets(r)]
    if met:
        best = met[0]
    else:
        # Pareto: among DD-feasible (p95DD<5), the max median.
        feas = [r for r in results if r["p95_dd"] < 5.0]
        feas.sort(key=lambda r: r["median"], reverse=True)
        best = feas[0] if feas else results[0]

    print("\n=== BEST CONFIG ===")
    print(json.dumps({k: v for k, v in best.items() if k != "basket"}, indent=2))
    print("basket:", best["basket"])

    # write full results
    out = {
        "n_markets": len(markets),
        "positive_ev_pool": [
            {"strat": r["strat"], "side": r["side"], "n": r["n"], "wr": r["wr"],
             "ev_per_$": r["ev"], "total": r["total"]} for r in pos
        ],
        "best": best,
        "top15": results[:15],
    }
    with open("/home/polybot/polymarket-bot/data/sim_target_existing_cheapEV_portfolio_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results json.")
    return best


if __name__ == "__main__":
    main()
