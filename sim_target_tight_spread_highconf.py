#!/usr/bin/env python3
"""
sim_target_tight_spread_highconf.py
===================================

FAMILY: tight-spread-highconf

Goal (from task): fire the FAVORED side (higher mid) only when BOTH up/dn spreads
are tight (<= spread_thr) AND inside a chosen cd-window, optionally requiring the
favored mid to clear a confidence floor. Size $1..$5 (min fire $1 = Polymarket PM-min).

We want HIGH winrate so that > 50% of fired markets win -> per-market PnL MEDIAN positive.

TARGET: per-market PnL MEDIAN >= +$1.00 AND max cumulative DRAWDOWN < $5.00
        (DD = peak-to-trough of running balance, start $9).

METHODOLOGY (leak-free, real-time first-trigger):
  * Each market in the panel has chronological ticks with cd counting DOWN.
  * We scan ticks from HIGH cd to LOW cd. The FIRST tick that satisfies the rule
    is the FIRE. We never look at later ticks and never use the winner to decide
    when/whether/which-side to fire. Winner is used ONLY to score PnL at settle.
  * Exclude cd <= 15 (unreliable to fill).
  * Cheap-entry payoff: a win at ask price e pays (1/e - 1) per $1; a loss pays -$1.

DRAWDOWN / MONTE CARLO:
  * Base run: markets in panel (chronological) order, balance starts at $9.
  * Monte Carlo: resample market ORDER (shuffle) N times to get a robust
    distribution of maxDD and wipe-rate (a single fixed order can understate or
    overstate the worst losing streak). Per-market PnL set is order-invariant, so
    MEDIAN / MEAN / %fired / WR are computed once on the base set; only DD-related
    stats (maxDD distribution, wipe rate) use the MC resampling.

WS-safety: the rule is O(1) from cached price history (spreads, mids at a tick),
computable in a worker thread. No look-ahead, no inline on_message work.

-------------------------------------------------------------------------------
MEASURED RESULT (456-market panel, leak-free first-trigger):

  TARGET (median PnL >= +$1.00 AND maxDD < $5.00) is NOT robustly met by this family.

  Best ROBUST config (one-real-loss safety buffer):
    FAVORED-side, window cd 260-360, both spreads <= 0.01, favored_mid >= 0.80,
    fire size = $4.  -> fires 25/456 mkts (5.5%), WR 96.0%, median +$0.938,
    mean +$0.703, maxDD $4.00, wipe 0%.  Median falls $0.062 short of target.

  PARETO WALL (the math): the median fired market is a WIN (WR>50%), so
    median PnL = size * (1/e - 1) with winning entry e ~ $0.81 (favored side is
    confidently priced). To lift median to $1.00 you must raise size to ~4.26,
    which lifts maxDD to ~$4.26+; pushing higher (size 5 = median +$1.17) makes
    maxDD = $5.00 -> breach. The book is efficient (mean ~ $0), so cheaper entries
    (median payoff up) come only with WR ~50% (median flips negative, DD explodes).

  OVERFIT corner (reported as caveat, NOT the answer):
    window 260-360, spreads<=0.02, mid>=0.82, size 5 -> 20 fires, WR 100%,
    median +$1.024, maxDD $0.00. It nominally MEETS the target, but DD=$0 ONLY
    because this finite sample had zero losses. Wilson-95 LB on the WR is 0.839;
    forced-WR Monte Carlo at p_true=0.88 gives P(maxDD>=$5)=92% and 14% wipe.
    A single future loss at size 5 = -$5 = immediate breach. Rejected as fragile.
-------------------------------------------------------------------------------
"""

import json
import os
import random
import statistics
import sys

PANEL_PATH = "/home/polybot/polymarket-bot/data/market_panel.json"
START_BAL = 9.0
CD_FLOOR = 15          # exclude cd <= 15 (unreliable fills)
MC_RUNS = 2000        # full precision for final reported configs
MC_RUNS_SCAN = 300    # lighter pass during grid/pareto scanning (size-invariant ratio already proven)
RNG_SEED = 12345


def load_panel(path=PANEL_PATH):
    with open(path) as f:
        return json.load(f)


def favored(t):
    """Return (side, ask_price, favored_mid) for the higher-mid (favored) side, or None."""
    up_mid = t.get("up_mid")
    dn_bid = t.get("dn_bid")
    dn_ask = t.get("dn_ask")
    if up_mid is None or dn_bid is None or dn_ask is None:
        return None
    dn_mid = (dn_bid + dn_ask) / 2.0
    if up_mid >= dn_mid:
        return "UP", t.get("up_ask"), up_mid
    else:
        return "DN", t.get("dn_ask"), dn_mid


def first_trigger_pnl(market, cd_lo, cd_hi, spread_thr, mid_floor, size,
                      entry_cap=1.0):
    """
    First-trigger over ticks (scan high->low cd). Fire FAVORED side when both
    spreads <= spread_thr, favored_mid >= mid_floor, ask <= entry_cap, cd in
    [cd_lo, cd_hi) and cd > CD_FLOOR. Return (pnl, won_bool, entry) or None if no fire.
    """
    win = market["winner"]
    for t in sorted(market["ticks"], key=lambda x: -x["cd"]):
        cd = t["cd"]
        if cd <= CD_FLOOR:
            continue
        if cd < cd_lo or cd >= cd_hi:
            continue
        su = t.get("spread_up")
        sd = t.get("spread_dn")
        if su is None or sd is None:
            continue
        if su > spread_thr or sd > spread_thr:
            continue
        f = favored(t)
        if f is None:
            continue
        side, ask, fmid = f
        if ask is None or ask <= 0.0 or ask >= 1.0:
            continue
        if fmid < mid_floor:
            continue
        if ask > entry_cap:
            continue
        won = (side == win)
        pnl = ((1.0 / ask - 1.0) if won else -1.0) * size
        return pnl, won, ask
    return None


def fired_pnls(panel, cd_lo, cd_hi, spread_thr, mid_floor, size, entry_cap=1.0):
    """Per-market PnL list (order-invariant) for fired markets, plus entries & wins."""
    pnls, entries, wins = [], [], 0
    for m in panel:
        r = first_trigger_pnl(m, cd_lo, cd_hi, spread_thr, mid_floor, size, entry_cap)
        if r is None:
            continue
        pnl, won, entry = r
        pnls.append(pnl)
        entries.append(entry)
        wins += 1 if won else 0
    return pnls, entries, wins


def run_drawdown(pnls, order):
    """Run a sequence of per-market PnLs in `order`, return (maxDD, final_bal, wiped)."""
    bal = START_BAL
    peak = START_BAL
    maxdd = 0.0
    wiped = False
    for i in order:
        bal += pnls[i]
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > maxdd:
            maxdd = dd
        if bal <= 0.0:
            wiped = True
    return maxdd, bal, wiped


def monte_carlo(pnls, runs=MC_RUNS_SCAN, seed=RNG_SEED):
    """Resample market ORDER to get a DD distribution + wipe rate."""
    if not pnls:
        return {"dd_p50": 0.0, "dd_p95": 0.0, "dd_max": 0.0,
                "wipe_rate_pct": 0.0, "chrono_dd": 0.0}
    rng = random.Random(seed)
    n = len(pnls)
    chrono_dd, _, _ = run_drawdown(pnls, list(range(n)))
    dds = []
    wipes = 0
    for _ in range(runs):
        order = list(range(n))
        rng.shuffle(order)
        dd, _, wiped = run_drawdown(pnls, order)
        dds.append(dd)
        if wiped:
            wipes += 1
    dds.sort()
    return {
        "dd_p50": dds[int(0.50 * (len(dds) - 1))],
        "dd_p95": dds[int(0.95 * (len(dds) - 1))],
        "dd_max": dds[-1],
        "wipe_rate_pct": 100.0 * wipes / runs,
        "chrono_dd": chrono_dd,
    }


def evaluate(panel, cd_lo, cd_hi, spread_thr, mid_floor, size, entry_cap=1.0,
             mc=True, mc_runs=MC_RUNS_SCAN):
    pnls, entries, wins = fired_pnls(panel, cd_lo, cd_hi, spread_thr,
                                     mid_floor, size, entry_cap)
    n_total = len(panel)
    n_fired = len(pnls)
    if n_fired == 0:
        return None
    median_pnl = statistics.median(pnls)
    mean_pnl = statistics.mean(pnls)
    wr = wins / n_fired
    pct_fired = 100.0 * n_fired / n_total
    med_entry = statistics.median(entries)
    mcres = monte_carlo(pnls, runs=mc_runs) if mc else {
        "dd_p50": 0.0, "dd_p95": 0.0, "dd_max": 0.0,
        "wipe_rate_pct": 0.0, "chrono_dd": 0.0}
    # report the worst-case maxDD seen across MC orders (conservative, what we must beat)
    maxdd_report = max(mcres["dd_max"], mcres["chrono_dd"])
    target_met = (median_pnl >= 1.0) and (maxdd_report < 5.0)
    return {
        "cd_lo": cd_lo, "cd_hi": cd_hi, "spread_thr": spread_thr,
        "mid_floor": mid_floor, "size": size, "entry_cap": entry_cap,
        "n_fired": n_fired, "pct_fired": pct_fired, "wr": wr,
        "median_pnl": median_pnl, "mean_pnl": mean_pnl, "med_entry": med_entry,
        "chrono_dd": mcres["chrono_dd"], "dd_p50": mcres["dd_p50"],
        "dd_p95": mcres["dd_p95"], "dd_max": maxdd_report,
        "wipe_rate_pct": mcres["wipe_rate_pct"], "target_met": target_met,
    }


def wilson_lb(k, n, z=1.96):
    """Wilson score 95% lower bound on a win rate of k/n."""
    if n == 0:
        return 0.0
    import math
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half


def forced_wr_stress(panel, cd_lo, cd_hi, spread_thr, mid_floor, size,
                     p_true, entry_cap=1.0, B=20000, seed=3):
    """
    HONEST robustness test. Take the n empirical fires' ENTRY prices, but draw each
    outcome from Bernoulli(p_true) instead of trusting the (possibly all-win) sample.
    Report P(maxDD>=$5), P(median<$1), wipe-rate over B shuffled order paths.
    Use p_true = Wilson 95% LB to see if the config survives a pessimistic true WR.
    """
    pnls, entries, wins = fired_pnls(panel, cd_lo, cd_hi, spread_thr,
                                     mid_floor, 1.0, entry_cap)
    n = len(entries)
    if n == 0:
        return None
    rng = random.Random(seed)
    ddb = medf = wipe = 0
    for _ in range(B):
        sp = []
        for _ in range(n):
            e = entries[rng.randrange(n)]
            won = rng.random() < p_true
            sp.append(((1.0 / e - 1.0) if won else -1.0) * size)
        rng.shuffle(sp)
        bal = peak = START_BAL
        mdd = 0.0
        wp = False
        for x in sp:
            bal += x
            if bal > peak:
                peak = bal
            if peak - bal > mdd:
                mdd = peak - bal
            if bal <= 0:
                wp = True
        if mdd >= 5.0:
            ddb += 1
        if statistics.median(sp) < 1.0:
            medf += 1
        if wp:
            wipe += 1
    return {"n": n, "p_true": p_true, "p_dd_ge5": 100.0 * ddb / B,
            "p_med_lt1": 100.0 * medf / B, "wipe_pct": 100.0 * wipe / B}


def fmt(r):
    return (f"win {r['cd_lo']}-{r['cd_hi']} thr{r['spread_thr']} mf{r['mid_floor']} "
            f"sz{r['size']} cap{r['entry_cap']}: fired={r['n_fired']}"
            f"({r['pct_fired']:.0f}%) WR={r['wr']:.3f} medEntry={r['med_entry']:.2f} "
            f"med={r['median_pnl']:+.3f} mean={r['mean_pnl']:+.3f} "
            f"DDchrono={r['chrono_dd']:.2f} DDp95={r['dd_p95']:.2f} "
            f"DDmax={r['dd_max']:.2f} wipe={r['wipe_rate_pct']:.1f}% "
            f"{'TARGET_MET' if r['target_met'] else ''}")


def main():
    panel = load_panel()
    print(f"# panel: {len(panel)} markets")

    # ----- GRID SEARCH (size=1 first; the median/DD ratio is size-invariant) -----
    windows = [(15, 60), (60, 120), (120, 200), (60, 200), (120, 260),
               (60, 260), (15, 260), (15, 360), (200, 260), (200, 360),
               (260, 360)]
    spread_thrs = [0.01, 0.015, 0.02]
    mid_floors = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]

    all_res = []
    for (lo, hi) in windows:
        for thr in spread_thrs:
            for mf in mid_floors:
                r = evaluate(panel, lo, hi, thr, mf, size=1.0, mc=True)
                if r is None or r["n_fired"] < 10:
                    continue
                all_res.append(r)

    # Best MEDIAN subject to DD<5 (size=1)
    dd_ok = [r for r in all_res if r["dd_max"] < 5.0]
    dd_ok.sort(key=lambda r: -r["median_pnl"])
    print("\n=== size=1 :: top configs with maxDD<$5 (sorted by median) ===")
    for r in dd_ok[:10]:
        print("  " + fmt(r))

    # Best overall median (context)
    all_res.sort(key=lambda r: -r["median_pnl"])
    print("\n=== size=1 :: top configs by median (DD unconstrained) ===")
    for r in all_res[:6]:
        print("  " + fmt(r))

    # ----- SIZE SWEEP on the best DD<5 config -> show the Pareto wall -----
    best_dd = dd_ok[0] if dd_ok else all_res[-1]
    print(f"\n=== SIZE SWEEP on best DD<5 base config "
          f"(win {best_dd['cd_lo']}-{best_dd['cd_hi']} thr{best_dd['spread_thr']} "
          f"mf{best_dd['mid_floor']}) ===")
    size_rows = []
    for sz in [1, 2, 3, 4, 5]:
        r = evaluate(panel, best_dd["cd_lo"], best_dd["cd_hi"],
                     best_dd["spread_thr"], best_dd["mid_floor"], size=sz, mc=True)
        size_rows.append(r)
        print("  " + fmt(r))

    # ----- FULL PARETO across ALL configs x sizes: any (median>=1 AND DD<5)? -----
    print("\n=== FULL PARETO SCAN (all configs x sizes 1..5) — searching for "
          "median>=$1 AND maxDD<$5 ===")
    winner = None
    pareto_best_median_at_dd5 = None   # closest achievable median while DD<5
    for r0 in all_res:
        for sz in [1, 2, 3, 4, 5]:
            r = evaluate(panel, r0["cd_lo"], r0["cd_hi"], r0["spread_thr"],
                         r0["mid_floor"], size=sz, mc=True)
            if r["dd_max"] < 5.0:
                if (pareto_best_median_at_dd5 is None or
                        r["median_pnl"] > pareto_best_median_at_dd5["median_pnl"]):
                    pareto_best_median_at_dd5 = r
            if r["target_met"]:
                if winner is None or r["median_pnl"] > winner["median_pnl"]:
                    winner = r

    if winner:
        print("  *** TARGET MET (historical sample) ***")
        print("  " + fmt(winner))
        # HONEST robustness vetting: does it survive a pessimistic true WR?
        pnls_w, ent_w, wins_w = fired_pnls(panel, winner["cd_lo"], winner["cd_hi"],
                                           winner["spread_thr"], winner["mid_floor"],
                                           1.0, winner["entry_cap"])
        nW = len(ent_w)
        lb = wilson_lb(wins_w, nW)
        print(f"  ROBUSTNESS CHECK: observed WR={wins_w/nW:.3f} on n={nW} fires; "
              f"Wilson95 lower bound={lb:.3f}.")
        for p in sorted({lb, 0.88, 0.92, 0.96, 1.00}):
            s = forced_wr_stress(panel, winner["cd_lo"], winner["cd_hi"],
                                 winner["spread_thr"], winner["mid_floor"],
                                 winner["size"], p, winner["entry_cap"])
            print(f"    forced WR p={p:.3f}: P(DD>=$5)={s['p_dd_ge5']:.1f}%  "
                  f"P(med<$1)={s['p_med_lt1']:.1f}%  wipe={s['wipe_pct']:.1f}%")
        print("  VERDICT: target met ONLY at p_true~1.0 (zero-loss streak). "
              "Under the Wilson-LB / realistic WR the DD>=$5 breach probability is "
              "high => this is an OVERFIT corner, NOT a robust solution.")
        winner = None  # treat as not robustly met for the final report
    if winner is None:
        print("  NO config (any window/thr/midfloor/cap, size 1..5) meets "
              "median>=$1 AND maxDD<$5.")
        print("  Closest achievable MEDIAN while keeping maxDD<$5:")
        print("  " + fmt(pareto_best_median_at_dd5))

    # Math reason: the median market is a WIN (WR>50%), so median PnL = size*(1/e - 1)
    # where e = winning entry; maxDD scales ~linearly in size. Ratio median/DD is fixed.
    bm = pareto_best_median_at_dd5
    ratio = bm["median_pnl"] / bm["dd_max"] if bm["dd_max"] > 0 else float("inf")
    size_needed = (1.0 / bm["median_pnl"]) * bm["size"] if bm["median_pnl"] > 0 else None
    dd_at_med1 = bm["dd_max"] * size_needed / bm["size"] if size_needed else None
    print(f"\n  PARETO MATH: best DD<5 point median/DD ratio = {ratio:.3f}.")
    if size_needed:
        print(f"  To reach median=$1.00 you would need size≈{size_needed:.2f} "
              f"-> maxDD≈${dd_at_med1:.2f} (>> $5). Impossible to satisfy both.")

    # ----- ROBUST FINAL: best median under DD<5 that ALSO survives a forced-WR
    #       stress at its own Wilson-95 lower bound (P(DD>=$5) must stay low). This
    #       rejects zero-loss-streak overfits whose DD=$0 only because the finite
    #       sample had no losses. -----
    print("\n=== ROBUST PARETO (DD<5 AND survives Wilson-LB forced-WR stress) ===")
    robust_candidates = []
    seen = set()
    for r0 in all_res:
        for sz in [1, 2, 3, 4, 5]:
            key = (r0["cd_lo"], r0["cd_hi"], r0["spread_thr"], r0["mid_floor"], sz)
            if key in seen:
                continue
            seen.add(key)
            r = evaluate(panel, r0["cd_lo"], r0["cd_hi"], r0["spread_thr"],
                         r0["mid_floor"], size=sz, mc=True)
            if r["dd_max"] >= 5.0:
                continue
            _, ent, wns = fired_pnls(panel, r0["cd_lo"], r0["cd_hi"],
                                     r0["spread_thr"], r0["mid_floor"], 1.0)
            n = len(ent)
            if n < 8:
                continue
            lb = wilson_lb(wns, n)
            s = forced_wr_stress(panel, r0["cd_lo"], r0["cd_hi"], r0["spread_thr"],
                                 r0["mid_floor"], sz, lb, B=4000)
            # robust = under pessimistic (Wilson-LB) WR, DD>=$5 breach prob is low
            if s["p_dd_ge5"] <= 25.0:
                robust_candidates.append((r, lb, s))
    robust_candidates.sort(key=lambda x: -x[0]["median_pnl"])
    for r, lb, s in robust_candidates[:8]:
        print(f"  {fmt(r)} | WilsonLB={lb:.3f} | stress P(DD>=5)={s['p_dd_ge5']:.1f}%")

    # ----- RESULT BLOCK for the orchestrator (full-precision MC re-run) -----
    print("\n=== FINAL CHOSEN (robust best median under DD<5) — full MC re-run ===")
    f0 = robust_candidates[0][0] if robust_candidates else pareto_best_median_at_dd5
    final = evaluate(panel, f0["cd_lo"], f0["cd_hi"], f0["spread_thr"],
                     f0["mid_floor"], size=f0["size"], entry_cap=f0["entry_cap"],
                     mc=True, mc_runs=MC_RUNS)
    print("  " + fmt(final))
    return final, winner


if __name__ == "__main__":
    main()
