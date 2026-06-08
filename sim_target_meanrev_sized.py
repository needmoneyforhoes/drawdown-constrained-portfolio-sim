#!/usr/bin/env python3
"""
sim_target_meanrev_sized.py
===========================
FAMILY: meanrev-sized

Searches MEAN-REVERSION-FADE configurations (the one validated orthogonal alpha:
short-horizon mean reversion in the coin-flip zone) for a config whose
PER-MARKET PnL has MEDIAN >= +$1.00 AND whose max cumulative DRAWDOWN
(peak-to-trough of running balance, start $9) is < $5.00.

RULE (per market, leak-free, real-time first-trigger):
  Scan ticks cd HIGH -> LOW (chronological; cd counts DOWN ~300->0).
  At the FIRST tick where, using ONLY ticks with cd >= K (the fire cd):
      * fire-window:  cd in [cd_lo, cd_hi]
      * UP_mid in [0.40, 0.60]  (coin-flip / "mid" zone)
      * |20s up_mid move| > move_threshold
  -> FADE the side that FELL over the last 20s (mean-revert):
       move>0 (UP rose, DN fell)  -> buy DN at dn_ask
       move<0 (UP fell)           -> buy UP at up_ask
  Optional CONFIDENCE GATE (tight-spread): require spread_up<=0.02 AND
       spread_dn<=0.02 at the fire tick. Two uses:
         gate="required" -> only fire when tight (selectivity)
         gate="sizeup"   -> always fire base size; size UP when tight
  Never fire at cd <= 15 (last 15s unreliable to fill).

SIZING (real-money, Polymarket PM-min = $1):
  base_size in {1,2,3}.  With gate="sizeup": stake = sizeup_size if tight else
  base_size.  PnL for a fired market = stake * ((1/entry - 1) if faded side wins
  else -1).  Cheap entries pay (1/entry - 1):1.

NO LEAK:
  - 20s move at cd=K uses up_mid(K) - up_mid(K+20); cd+20 is strictly earlier.
  - First-trigger: stop at the first qualifying tick scanning high->low.
  - The winner is used ONLY to score PnL at settle, never in the decision.

MONTE CARLO:
  For each config, the per-market realized PnL vector (one entry per FIRED market)
  is fixed (deterministic given the rule). We bootstrap-resample the ORDER of
  fired markets (2000 runs), build the running balance starting at $9, and record:
    - max drawdown  = max over the run of (running_peak - running_balance)
    - wipe          = balance hit <= $0 at any point
  We aggregate across runs:
    - median per-market PnL (median of the per-market PnL vector; order-invariant,
      so reported directly from the realized vector)
    - mean per-market PnL
    - max drawdown = the MAX drawdown observed across all 2000 bootstrap runs
      (worst-case peak-to-trough an adversarial market order could produce)
    - wipe rate    = fraction of the 2000 runs that wiped
    - % markets fired = fired / total markets

We report the config closest to (median>=+$1 AND maxDD<$5), and — being honest
that cheap fades pay little per win — the Pareto-best median achievable under
DD<$5.
"""
import json
import random
import statistics
from collections import defaultdict

PANEL_PATH = "./data/market_panel.json"

MID_LO, MID_HI = 0.40, 0.60
CD_FLOOR = 16          # never fire at cd <= 15
LOOKBACK = 20          # 20-second mean-rev horizon
TIGHT_SPREAD = 0.02    # confidence-gate threshold
START_BAL = 9.0
N_BOOT = 2000
SEED = 1337

# Mathematical reality discovered in the broad sweep: every config in the base
# grid fires on >=224 markets (~49-57%). Over 200+ bets a +5% edge produces a
# structural peak-to-trough drawdown of $40-$160 — DD<$5 from a $9 start is
# impossible at that fire count. The ONLY route to DD<$5 is to fire on a HANDFUL
# of markets (short series can't drift far below the $9 start). So we add a
# SELECTIVE tier that stacks a confidence stack to cut fires to single/low-double
# digits, then re-tests the target. This is the Pareto frontier exploration.

# ---- search grid ----
MOVE_THRESHOLDS = [0.008, 0.012, 0.016, 0.020, 0.024]   # 0.8c .. 2.4c
# fire-windows (cd_hi inclusive, cd_lo inclusive). Edge concentrated 360-180 & 180-120.
FIRE_WINDOWS = [
    ("full",     10_000, CD_FLOOR),
    ("360-180",     360,      181),
    ("360-120",     360,      121),   # the two positive contiguous windows joined
    ("200-120",     200,      121),   # validated scan-start cd<=200 region
    ("180-120",     180,      121),
]
GATES = ["none", "required", "sizeup"]
BASE_SIZES = [1, 2, 3]
SIZEUP_SIZES = [2, 3]   # only used when gate == "sizeup"


# --------------------------------------------------------------------------- #
# Tick-series construction (last-wins per cd second)                          #
# --------------------------------------------------------------------------- #
def build_cd_series(ticks):
    cd_to_row = {}
    cd_to_mid = {}
    for t in ticks:
        cd = t["cd"]
        um = t.get("up_mid")
        if um is None:
            continue
        cd_to_row[cd] = t      # last-wins => freshest state at that cd second
        cd_to_mid[cd] = um
    ordered = sorted(cd_to_row.keys(), reverse=True)   # cd descending = chronological
    series = []
    for cd in ordered:
        t = cd_to_row[cd]
        series.append((
            cd, t.get("up_mid"), t.get("up_ask"), t.get("dn_ask"),
            t.get("spread_up"), t.get("spread_dn"),
        ))
    return series, cd_to_mid


def mid_move_20s(cd, up_mid_now, cd_to_mid):
    """up_mid(now) - up_mid(as of cd+20). cd+20 is strictly earlier => no leak."""
    ref = cd + LOOKBACK
    if ref in cd_to_mid:
        return up_mid_now - cd_to_mid[ref]
    for delta in (1, 2):
        for r in (ref + delta, ref - delta):
            if r in cd_to_mid:
                return up_mid_now - cd_to_mid[r]
    best, best_d = None, 99
    for r in range(ref - 5, ref + 6):
        if r in cd_to_mid:
            d = abs(r - ref)
            if d < best_d:
                best_d, best = d, r
    if best is not None:
        return up_mid_now - cd_to_mid[best]
    return None


def first_trigger_fire_selective(market, threshold, cd_hi, cd_lo,
                                 spread_cap, mid_lo, mid_hi):
    """
    High-selectivity variant: require BOTH spreads <= spread_cap (a confidence
    gate tighter than the base 0.02), a narrowed coin-flip zone [mid_lo,mid_hi],
    and a (typically larger) move threshold. First-trigger, leak-free, same fade.
    Purpose: drive fire-count toward single/low-double digits so the running
    balance can't drift past a $5 drawdown from $9.
    """
    series, cd_to_mid = build_cd_series(market["ticks"])
    for cd, up_mid, up_ask, dn_ask, su, sd in series:
        if cd <= CD_FLOOR:
            break
        if not (cd_lo <= cd <= cd_hi):
            continue
        if up_mid is None or not (mid_lo <= up_mid <= mid_hi):
            continue
        if su is None or sd is None or su > spread_cap or sd > spread_cap:
            continue
        move = mid_move_20s(cd, up_mid, cd_to_mid)
        if move is None or abs(move) <= threshold:
            continue
        if move > 0:
            side, entry = "DN", dn_ask
        else:
            side, entry = "UP", up_ask
        if entry is None or entry <= 0 or entry >= 1.0:
            continue
        return {"cd": cd, "side": side, "entry": entry, "move": move, "tight": True}
    return None


def first_trigger_fire(market, threshold, cd_hi, cd_lo, gate):
    """
    First qualifying fire scanning cd high->low. Returns fire dict or None.
    gate="required" -> only fire when both spreads <= TIGHT_SPREAD.
    gate in ("none","sizeup") -> fire regardless of spread (sizeup affects only
    staking, computed later via the 'tight' flag).
    """
    series, cd_to_mid = build_cd_series(market["ticks"])
    for cd, up_mid, up_ask, dn_ask, su, sd in series:
        if cd <= CD_FLOOR:
            break
        if not (cd_lo <= cd <= cd_hi):
            continue
        if up_mid is None or not (MID_LO <= up_mid <= MID_HI):
            continue
        move = mid_move_20s(cd, up_mid, cd_to_mid)
        if move is None or abs(move) <= threshold:
            continue
        tight = (su is not None and sd is not None
                 and su <= TIGHT_SPREAD and sd <= TIGHT_SPREAD)
        if gate == "required" and not tight:
            continue
        if move > 0:                 # UP rose => DN fell => fade buys DN
            side, entry = "DN", dn_ask
        else:                        # UP fell => fade buys UP
            side, entry = "UP", up_ask
        if entry is None or entry <= 0 or entry >= 1.0:
            continue
        return {"cd": cd, "side": side, "entry": entry, "move": move, "tight": tight}
    return None


def per_market_pnls(panel, threshold, cd_hi, cd_lo, gate, base_size, sizeup_size):
    """
    Return (pnls, n_fired, n_total, n_tight). pnls = list of realized per-market
    dollar PnL for FIRED markets (one entry each).
    """
    pnls = []
    n_tight = 0
    for m in panel:
        fire = first_trigger_fire(m, threshold, cd_hi, cd_lo, gate)
        if fire is None:
            continue
        if gate == "sizeup":
            stake = sizeup_size if fire["tight"] else base_size
        else:
            stake = base_size
        if fire["tight"]:
            n_tight += 1
        won = (fire["side"] == m["winner"])
        unit = (1.0 / fire["entry"] - 1.0) if won else -1.0
        pnls.append(stake * unit)
    return pnls, len(pnls), len(panel), n_tight


# --------------------------------------------------------------------------- #
# Monte Carlo: bootstrap-resample market ORDER, running balance from $9       #
# --------------------------------------------------------------------------- #
def monte_carlo(pnls, n_boot=N_BOOT, start=START_BAL, rng=None):
    """
    Returns dict with worst-case max drawdown across runs, mean of per-run max
    drawdown, wipe rate, and the median/mean per-market PnL (order-invariant).
    Bootstrap resamples WITH replacement the same number of fired markets, in a
    random order, to probe the distribution of running-balance drawdowns.
    """
    if not pnls:
        return None
    if rng is None:
        rng = random.Random(SEED)
    k = len(pnls)
    worst_dd = 0.0
    dd_samples = []
    wipes = 0
    final_bals = []
    for _ in range(n_boot):
        sample = [pnls[rng.randrange(k)] for _ in range(k)]
        bal = start
        peak = start
        run_dd = 0.0
        wiped = False
        for x in sample:
            bal += x
            if bal > peak:
                peak = bal
            dd = peak - bal
            if dd > run_dd:
                run_dd = dd
            if bal <= 0.0:
                wiped = True
        dd_samples.append(run_dd)
        if run_dd > worst_dd:
            worst_dd = run_dd
        if wiped:
            wipes += 1
        final_bals.append(bal)
    return {
        "median_pnl": statistics.median(pnls),
        "mean_pnl": statistics.fmean(pnls),
        "max_dd_worst": worst_dd,                       # adversarial worst order
        "max_dd_mean": statistics.fmean(dd_samples),    # typical order
        "max_dd_p95": sorted(dd_samples)[int(0.95 * (n_boot - 1))],
        "wipe_rate": wipes / n_boot,
        "mean_final_bal": statistics.fmean(final_bals),
        "n_fired": k,
    }


def evaluate(panel):
    results = []
    for thr in MOVE_THRESHOLDS:
        for wlabel, cd_hi, cd_lo in FIRE_WINDOWS:
            for gate in GATES:
                for base in BASE_SIZES:
                    sizeups = SIZEUP_SIZES if gate == "sizeup" else [None]
                    for su in sizeups:
                        if su is not None and su <= base:
                            continue   # sizeup must exceed base to mean anything
                        pnls, n_fired, n_total, n_tight = per_market_pnls(
                            panel, thr, cd_hi, cd_lo, gate, base, su)
                        if n_fired == 0:
                            continue
                        mc = monte_carlo(pnls)
                        wr = sum(1 for p in pnls if p > 0) / n_fired
                        cfg = {
                            "family": "meanrev-sized",
                            "move_thr_c": round(thr * 100, 1),
                            "window": wlabel,
                            "cd_hi": cd_hi, "cd_lo": cd_lo,
                            "gate": gate,
                            "base_size": base,
                            "sizeup_size": su,
                            "n_fired": n_fired,
                            "pct_fired": 100.0 * n_fired / n_total,
                            "n_tight": n_tight,
                            "win_rate_pct": round(100 * wr, 1),
                            "median_pnl": mc["median_pnl"],
                            "mean_pnl": mc["mean_pnl"],
                            "max_dd_worst": mc["max_dd_worst"],
                            "max_dd_mean": mc["max_dd_mean"],
                            "max_dd_p95": mc["max_dd_p95"],
                            "wipe_rate": mc["wipe_rate"],
                            "mean_final_bal": mc["mean_final_bal"],
                            "total_pnl": sum(pnls),
                        }
                        results.append(cfg)
    return results


def evaluate_selective(panel):
    """
    Stack confidence gates to cut fire-count toward single digits. Vary spread_cap
    (ultra-tight), narrowed mid zone, large move threshold, tight window. Test
    fire sizes 1..3. These are the only candidates that can plausibly hold DD<$5.
    """
    spread_caps = [0.02, 0.01]
    mid_zones = [(0.45, 0.55), (0.47, 0.53)]   # narrowed coin-flip cores
    thrs = [0.020, 0.030, 0.040]               # larger moves only
    windows = [("360-180", 360, 181), ("360-240", 360, 241), ("300-200", 300, 201)]
    sizes = [1, 2, 3]
    results = []
    for sc in spread_caps:
        for (mlo, mhi) in mid_zones:
            for thr in thrs:
                for wlabel, hi, lo in windows:
                    pnls_unit = []
                    for m in panel:
                        fire = first_trigger_fire_selective(m, thr, hi, lo, sc, mlo, mhi)
                        if fire is None:
                            continue
                        won = (fire["side"] == m["winner"])
                        unit = (1.0 / fire["entry"] - 1.0) if won else -1.0
                        pnls_unit.append(unit)
                    if not pnls_unit:
                        continue
                    for size in sizes:
                        pnls = [size * u for u in pnls_unit]
                        mc = monte_carlo(pnls)
                        wr = sum(1 for p in pnls if p > 0) / len(pnls)
                        results.append({
                            "family": "meanrev-sized",
                            "tier": "selective",
                            "move_thr_c": round(thr * 100, 1),
                            "window": wlabel, "cd_hi": hi, "cd_lo": lo,
                            "gate": f"required(spread<= {sc})",
                            "spread_cap": sc,
                            "mid_zone": [mlo, mhi],
                            "base_size": size, "sizeup_size": None,
                            "n_fired": len(pnls),
                            "pct_fired": 100.0 * len(pnls) / len(panel),
                            "win_rate_pct": round(100 * wr, 1),
                            "median_pnl": mc["median_pnl"],
                            "mean_pnl": mc["mean_pnl"],
                            "max_dd_worst": mc["max_dd_worst"],
                            "max_dd_mean": mc["max_dd_mean"],
                            "max_dd_p95": mc["max_dd_p95"],
                            "wipe_rate": mc["wipe_rate"],
                            "mean_final_bal": mc["mean_final_bal"],
                            "total_pnl": sum(pnls),
                        })
    return results


def cfg_str(c):
    if c.get("tier") == "selective":
        mlo, mhi = c["mid_zone"]
        return (f"FADE coin-flip[{mlo},{mhi}], move>|{c['move_thr_c']}c| over 20s, "
                f"first-trigger cd-window {c['window']}[{c['cd_hi']}..{c['cd_lo']}], "
                f"fire ${c['base_size']}, gate=required(spread_up<= {c['spread_cap']} "
                f"AND spread_dn<= {c['spread_cap']}), exclude cd<=15")
    g = c["gate"]
    gpart = (f"gate=sizeup(tight<= {TIGHT_SPREAD}: ${c['sizeup_size']} else ${c['base_size']})"
             if g == "sizeup" else
             f"gate=required(spread<= {TIGHT_SPREAD})" if g == "required" else
             "gate=none")
    return (f"FADE coin-flip[0.40,0.60], move>|{c['move_thr_c']}c| over 20s, "
            f"first-trigger cd-window {c['window']}[{c['cd_hi']}..{c['cd_lo']}], "
            f"fire ${c['base_size']}, {gpart}, exclude cd<=15")


def main():
    with open(PANEL_PATH) as f:
        panel = json.load(f)
    n_up = sum(1 for m in panel if m["winner"] == "UP")
    n_dn = sum(1 for m in panel if m["winner"] == "DN")
    print(f"Loaded {len(panel)} markets ({n_up} UP / {n_dn} DN)\n")
    print("FAMILY = meanrev-sized | MC: 2000 bootstrap runs, start $9, "
          "DD=peak-to-trough of running balance\n")

    results = evaluate(panel)
    sel = evaluate_selective(panel)
    results = results + sel

    print("=" * 100)
    print(f"SELECTIVE TIER ({len(sel)} configs) — lowest-fire-count candidates "
          "(the only route to DD<$5):")
    for c in sorted(sel, key=lambda c: (c["max_dd_worst"]))[:14]:
        print(f"  n={c['n_fired']:>2} fired={c['pct_fired']:.1f}% "
              f"med=${c['median_pnl']:+.3f} mean=${c['mean_pnl']:+.3f} "
              f"ddWorst=${c['max_dd_worst']:.2f} ddMean=${c['max_dd_mean']:.2f} "
              f"wipe={c['wipe_rate']*100:.1f}% WR={c['win_rate_pct']}% "
              f"| {c['window']} {c['move_thr_c']}c spr<= {c['spread_cap']} "
              f"mid{c['mid_zone']} ${c['base_size']}")
    print()

    # Honest target test: median>=+$1 AND worst-case maxDD<$5.
    target = [c for c in results
              if c["median_pnl"] >= 1.0 and c["max_dd_worst"] < 5.0]
    # Also a softer test using the TYPICAL (mean) drawdown, for context.
    target_typ = [c for c in results
                  if c["median_pnl"] >= 1.0 and c["max_dd_mean"] < 5.0]

    # Pareto-best median achievable under worst-case DD<$5.
    under_dd_worst = [c for c in results if c["max_dd_worst"] < 5.0]
    under_dd_mean = [c for c in results if c["max_dd_mean"] < 5.0]
    best_median_under_dd_worst = max(under_dd_worst, key=lambda c: c["median_pnl"]) \
        if under_dd_worst else None
    best_median_under_dd_mean = max(under_dd_mean, key=lambda c: c["median_pnl"]) \
        if under_dd_mean else None

    # Score for "closest to target": maximize median while keeping DD low; we rank
    # by a combined objective that prefers DD<5 and high median, positive mean.
    def closeness(c):
        # primary: meets worst-DD<5 ; secondary: median ; tertiary: mean ; penalize wipe
        meets = 1 if c["max_dd_worst"] < 5.0 else 0
        return (meets, c["median_pnl"], c["mean_pnl"], -c["wipe_rate"])

    best_overall = max(results, key=closeness)

    print("=" * 100)
    print("TARGET HITS (median>=+$1.00 AND worst-case maxDD<$5.00):", len(target))
    for c in sorted(target, key=lambda c: (-c["median_pnl"], c["max_dd_worst"]))[:10]:
        print(f"  med=${c['median_pnl']:+.3f} mean=${c['mean_pnl']:+.3f} "
              f"ddW=${c['max_dd_worst']:.2f} ddMean=${c['max_dd_mean']:.2f} "
              f"wipe={c['wipe_rate']*100:.1f}% fired={c['pct_fired']:.0f}% "
              f"WR={c['win_rate_pct']}%  | {cfg_str(c)}")

    print("\nTARGET HITS using TYPICAL (mean-order) maxDD<$5:", len(target_typ))
    for c in sorted(target_typ, key=lambda c: (-c["median_pnl"], c["max_dd_mean"]))[:10]:
        print(f"  med=${c['median_pnl']:+.3f} mean=${c['mean_pnl']:+.3f} "
              f"ddMean=${c['max_dd_mean']:.2f} ddWorst=${c['max_dd_worst']:.2f} "
              f"wipe={c['wipe_rate']*100:.1f}% fired={c['pct_fired']:.0f}%  | {cfg_str(c)}")

    print("\n" + "=" * 100)
    print("PARETO-BEST MEDIAN under worst-case DD<$5:")
    if best_median_under_dd_worst:
        c = best_median_under_dd_worst
        print(f"  med=${c['median_pnl']:+.3f} mean=${c['mean_pnl']:+.3f} "
              f"ddWorst=${c['max_dd_worst']:.2f} wipe={c['wipe_rate']*100:.1f}% "
              f"fired={c['pct_fired']:.0f}% WR={c['win_rate_pct']}%\n    {cfg_str(c)}")
    print("PARETO-BEST MEDIAN under TYPICAL (mean-order) DD<$5:")
    if best_median_under_dd_mean:
        c = best_median_under_dd_mean
        print(f"  med=${c['median_pnl']:+.3f} mean=${c['mean_pnl']:+.3f} "
              f"ddMean=${c['max_dd_mean']:.2f} ddWorst=${c['max_dd_worst']:.2f} "
              f"wipe={c['wipe_rate']*100:.1f}% fired={c['pct_fired']:.0f}% "
              f"WR={c['win_rate_pct']}%\n    {cfg_str(c)}")

    # Best by MEAN PnL overall (the metric the alpha actually optimizes), DD<5 worst
    if under_dd_worst:
        bm = max(under_dd_worst, key=lambda c: c["mean_pnl"])
        print("\nBEST MEAN PnL under worst-case DD<$5:")
        print(f"  med=${bm['median_pnl']:+.3f} mean=${bm['mean_pnl']:+.3f} "
              f"ddWorst=${bm['max_dd_worst']:.2f} wipe={bm['wipe_rate']*100:.1f}% "
              f"fired={bm['pct_fired']:.0f}% total=${bm['total_pnl']:+.2f}\n    {cfg_str(bm)}")

    print("\n" + "=" * 100)
    print("TOP 15 by MEAN PnL (all configs):")
    for c in sorted(results, key=lambda c: -c["mean_pnl"])[:15]:
        print(f"  mean=${c['mean_pnl']:+.3f} med=${c['median_pnl']:+.3f} "
              f"ddW=${c['max_dd_worst']:.2f} ddMean=${c['max_dd_mean']:.2f} "
              f"wipe={c['wipe_rate']*100:.1f}% fired={c['pct_fired']:.0f}% "
              f"n={c['n_fired']} WR={c['win_rate_pct']}% | {c['window']} "
              f"{c['move_thr_c']}c {c['gate']} ${c['base_size']}"
              + (f"/${c['sizeup_size']}" if c['sizeup_size'] else ""))

    # Persist full results + the chosen answer
    answer = {
        "best_overall": best_overall,
        "target_hits_worst": len(target),
        "target_hits_typical": len(target_typ),
        "best_median_under_dd_worst": best_median_under_dd_worst,
        "best_median_under_dd_mean": best_median_under_dd_mean,
        "n_markets": len(panel),
    }
    with open("./data/.meanrev_sized_summary.json", "w") as f:
        json.dump({"answer": answer, "all_results": results}, f, indent=2)
    print("\n[wrote ./.meanrev_sized_summary.json]")

    return answer


if __name__ == "__main__":
    main()
