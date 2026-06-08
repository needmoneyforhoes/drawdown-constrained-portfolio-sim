#!/usr/bin/env python3
"""
sim_target_kelly_dd_constrained.py
==================================
FAMILY: kelly-dd-constrained

GOAL
----
Take the SINGLE validated edge (short-horizon MEAN-REVERSION FADE in the
coin-flip zone, spread<=0.02 confidence gate, first-trigger scan-start cd<=200,
|20s up_mid move|>2.0c) and focus PURELY on the SIZING math:

  1. Compute the fractional-Kelly stake from the realized per-$1 edge.
  2. Apply a DRAWDOWN-CONSTRAINED sizing rule that shrinks the stake as the
     running drawdown approaches the $5 budget (the classic "drawdown control"
     / vol-target overlay). Floor at the Polymarket $1 minimum: if the
     DD-scaled stake falls below $1 we SKIP the bet rather than over-bet.
  3. Ask: with optimal DD-aware sizing, what is the MAX MEDIAN per-market PnL
     achievable while GUARANTEEING maxDD<$5 across 2000 Monte-Carlo runs?
     This is the true math ceiling.

THE EDGE (fixed, leak-free, reproduced from market_panel.json)
--------------------------------------------------------------
Per market, scan ticks cd HIGH->LOW (chronological; cd counts DOWN ~300->0).
At the FIRST tick (using ONLY ticks with cd >= that cd; cd+20 is strictly
earlier so the 20s move never peeks forward; winner never used in the decision):
    * cd in [16, 200]                 (scan-start cd<=200, exclude last 15s)
    * up_mid in [0.40, 0.60]          (coin-flip / "mid" zone)
    * spread_up <= 0.02 AND spread_dn <= 0.02   (tight-spread confidence gate)
    * |up_mid(cd) - up_mid(cd+20)| > 0.020      (>2.0c, 20s mean-rev horizon)
  -> FADE the side that FELL over the last 20s (mean revert):
       move>0 (UP rose, DN fell) -> buy DN at dn_ask
       move<0 (UP fell)          -> buy UP at up_ask
Realized on the 456-market panel: 298 fires (65.4%), WR 54.4%,
mean per-$1 unit +$0.0451, median per-$1 unit +$0.667, std 0.975,
total +$13.45/$1. (Matches the CONFIRMED study.)

PER-$1 PAYOFF (unit): win -> (1/entry - 1)   [cheap entries pay multiples]
                       loss -> -1            [lose the full $1 staked]

KELLY MATH
----------
Full-Kelly fraction f* maximizes E[log(1 + f*u)] over the realized unit vector
u_i.  Numerically f* ~= 0.047 (bet ~4.7% of bankroll).  At a $9 bankroll that
is ~$0.42 -- BELOW the $1 Polymarket minimum.  This is the central reality:
the edge is real but THIN, so "true Kelly" is sub-minimum.  We therefore study
FRACTIONAL Kelly with a fixed DOLLAR base size (>= $1) and overlay the DD
constraint, and ALSO report the pure-fractional-of-bankroll Kelly path for
completeness.

DRAWDOWN-CONSTRAINED SIZING RULE (the heart of this family)
-----------------------------------------------------------
Let DD_now = peak - balance be the running drawdown, DD_BUDGET the cap we are
defending (we sweep budgets <5 so the worst-case stays <$5 with the largest
single loss = full stake), and headroom = DD_BUDGET - DD_now.

    stake = base_size * kelly_frac * clamp(headroom / DD_BUDGET, 0, 1) ** dd_power

  - kelly_frac in {0.25, 0.5, 1.0} = fractional-Kelly multiplier on base_size.
  - dd_power >=1 makes sizing back off faster as DD approaches the budget
    (convex throttle); dd_power=0 disables the throttle (flat sizing) as a
    control.
  - HARD GUARANTEE: a single bet can lose at most `stake` (unit=-1).  If we
    require  stake <= headroom  at fire time, the balance can NEVER drop more
    than DD_BUDGET below the peak => maxDD <= DD_BUDGET < $5 BY CONSTRUCTION.
    We enforce this hard clamp (stake = min(stake, headroom)) on TOP of the
    smooth throttle.  That clamp is what makes maxDD<$5 a GUARANTEE, not a hope.
  - $1 floor: if the clamped stake < $1 we SKIP (cannot place a sub-min order).
    Skipping is the only DD-safe response -- rounding UP to $1 could breach the
    budget on a loss.

So the realized maxDD is bounded by DD_BUDGET by construction; we sweep
DD_BUDGET up to 4.99 and pick the sizing that MAXIMIZES median per-market PnL
(over markets where we actually fired) subject to the across-2000-run worst-case
maxDD < $5 and wipe-rate == 0.

MONTE CARLO
-----------
The per-market unit vector is fixed by the rule.  Because the DD-constrained
stake DEPENDS ON PATH (current DD), PnL is NOT order-invariant, so we MUST
simulate sequences.  We run 2000 bootstrap sequences (resample the fired-market
unit outcomes WITH replacement, random order) and for each:
    walk balance from $9, applying the DD-constrained stake at every step,
    track peak, running DD, per-bet realized $ PnL, wipe (bal<=0), and whether
    any bet was SKIPPED (stake<$1).
We aggregate:
    median per-market PnL = median over ALL placed bets across all runs of the
        realized $ PnL per placed bet  (the per-market $ outcome)
    mean   per-market PnL = mean of the same
    maxDD  = MAX running drawdown observed across all 2000 runs (worst case)
    wipe   = fraction of runs that ever hit balance<=0
    %fired = placed-bet fraction = (avg placed bets per run) / fired-market count
             * (fired-market count / total markets)  -> reported as placed/total
We also report the deterministic in-sample path (markets in true chronological
panel order) for reference.

OUTPUT: the sizing config with the HIGHEST median per-market PnL among all that
GUARANTEE worst-case maxDD<$5 and wipe==0, plus the Pareto note.
"""
import json
import math
import random
import statistics

PANEL_PATH = "./data/market_panel.json"

# ---- fixed edge parameters (the CONFIRMED deployable config) ----
CD_FLOOR = 16          # never fire at cd <= 15
CD_HI    = 200         # scan-start cd <= 200
CD_LO    = CD_FLOOR
MID_LO, MID_HI = 0.40, 0.60
TIGHT    = 0.02        # spread confidence gate (both sides)
THR      = 0.020       # |20s up_mid move| > 2.0c
LOOKBACK = 20

START_BAL = 9.0
N_BOOT    = 2000
SEED      = 1337


# --------------------------------------------------------------------------- #
# Edge extraction (leak-free first-trigger fade)                              #
# --------------------------------------------------------------------------- #
def build_cd_series(ticks):
    cd_to_row, cd_to_mid = {}, {}
    for t in ticks:
        cd = t["cd"]
        um = t.get("up_mid")
        if um is None:
            continue
        cd_to_row[cd] = t          # last-wins => freshest state at that cd second
        cd_to_mid[cd] = um
    ordered = sorted(cd_to_row.keys(), reverse=True)   # cd descending = chronological
    series = []
    for cd in ordered:
        t = cd_to_row[cd]
        series.append((cd, t.get("up_mid"), t.get("up_ask"), t.get("dn_ask"),
                       t.get("spread_up"), t.get("spread_dn")))
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
    return (up_mid_now - cd_to_mid[best]) if best is not None else None


def first_trigger_fade(market):
    """First qualifying fade fire scanning cd high->low. Returns dict or None."""
    series, cd_to_mid = build_cd_series(market["ticks"])
    for cd, up_mid, up_ask, dn_ask, su, sd in series:
        if cd <= CD_FLOOR:
            break
        if not (CD_LO <= cd <= CD_HI):
            continue
        if up_mid is None or not (MID_LO <= up_mid <= MID_HI):
            continue
        if su is None or sd is None or su > TIGHT or sd > TIGHT:
            continue
        move = mid_move_20s(cd, up_mid, cd_to_mid)
        if move is None or abs(move) <= THR:
            continue
        if move > 0:                 # UP rose => DN fell => fade buys DN
            side, entry = "DN", dn_ask
        else:                        # UP fell => fade buys UP
            side, entry = "UP", up_ask
        if entry is None or entry <= 0 or entry >= 1.0:
            continue
        return {"cd": cd, "side": side, "entry": entry, "move": move}
    return None


def extract_units(panel):
    """Realized per-$1 unit outcome for each FIRED market (one per market)."""
    units, entries, wins = [], [], 0
    for m in panel:
        f = first_trigger_fade(m)
        if f is None:
            continue
        won = (f["side"] == m["winner"])
        if won:
            wins += 1
        u = (1.0 / f["entry"] - 1.0) if won else -1.0
        units.append(u)
        entries.append(f["entry"])
    return units, entries, wins


# --------------------------------------------------------------------------- #
# Kelly math                                                                   #
# --------------------------------------------------------------------------- #
def full_kelly_fraction(units):
    """f* maximizing mean log(1 + f*u) over realized units (grid + refine)."""
    n = len(units)
    def growth(f):
        s = 0.0
        for u in units:
            v = 1.0 + f * u
            if v <= 0:
                return -1e18
            s += math.log(v)
        return s / n
    # coarse grid then golden-ish refine
    best_f, best_g = 0.0, growth(0.0)
    for i in range(1, 1001):
        f = i / 1000.0
        g = growth(f)
        if g > best_g:
            best_g, best_f = g, f
        if g == -1e18:
            break
    lo, hi = max(0.0, best_f - 0.001), best_f + 0.001
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if growth(m1) < growth(m2):
            lo = m1
        else:
            hi = m2
    f = (lo + hi) / 2
    return f, growth(f)


# --------------------------------------------------------------------------- #
# DD-constrained stake function                                                #
# --------------------------------------------------------------------------- #
def dd_constrained_stake(base_size, kelly_frac, dd_power,
                         balance, peak, dd_budget, min_fire=1.0):
    """
    Smooth DD throttle * fractional-Kelly base, then HARD-clamp to headroom so a
    single loss can never breach dd_budget.  Returns 0.0 to SKIP (sub-min).
    """
    dd_now = peak - balance
    headroom = dd_budget - dd_now
    if headroom <= 0:
        return 0.0
    frac = headroom / dd_budget
    if frac < 0:
        frac = 0.0
    elif frac > 1:
        frac = 1.0
    throttle = frac ** dd_power if dd_power > 0 else 1.0
    stake = base_size * kelly_frac * throttle
    # HARD guarantee: never risk more than current headroom.
    if stake > headroom:
        stake = headroom
    # also never risk more than the balance itself (can't go below 0 on a loss)
    if stake > balance:
        stake = balance
    if stake < min_fire:
        return 0.0          # sub-min -> SKIP (the only DD-safe response)
    return stake


# --------------------------------------------------------------------------- #
# Monte-Carlo path simulation of a DD-constrained sizing config               #
# --------------------------------------------------------------------------- #
def simulate_config(units, base_size, kelly_frac, dd_power, dd_budget,
                    n_boot=N_BOOT, start=START_BAL, seed=SEED, min_fire=1.0):
    rng = random.Random(seed)
    k = len(units)
    worst_dd = 0.0
    dd_run_samples = []
    wipes = 0
    final_bals = []
    all_bet_pnls = []          # per PLACED bet $ PnL, pooled across runs
    placed_counts = []
    skipped_counts = []
    for _ in range(n_boot):
        seq = [units[rng.randrange(k)] for _ in range(k)]
        bal = start
        peak = start
        run_dd = 0.0
        placed = 0
        skipped = 0
        wiped = False
        for u in seq:
            stake = dd_constrained_stake(base_size, kelly_frac, dd_power,
                                         bal, peak, dd_budget, min_fire)
            if stake <= 0.0:
                skipped += 1
                continue
            pnl = stake * u            # win: stake*(1/entry-1); loss: -stake
            bal += pnl
            placed += 1
            all_bet_pnls.append(pnl)
            if bal > peak:
                peak = bal
            dd = peak - bal
            if dd > run_dd:
                run_dd = dd
            if bal <= 0.0:
                wiped = True
        dd_run_samples.append(run_dd)
        if run_dd > worst_dd:
            worst_dd = run_dd
        if wiped:
            wipes += 1
        final_bals.append(bal)
        placed_counts.append(placed)
        skipped_counts.append(skipped)
    if not all_bet_pnls:
        return None
    return {
        "median_pnl": statistics.median(all_bet_pnls),
        "mean_pnl": statistics.fmean(all_bet_pnls),
        "max_dd_worst": worst_dd,
        "max_dd_mean": statistics.fmean(dd_run_samples),
        "max_dd_p95": sorted(dd_run_samples)[int(0.95 * (n_boot - 1))],
        "wipe_rate": wipes / n_boot,
        "mean_final_bal": statistics.fmean(final_bals),
        "avg_placed": statistics.fmean(placed_counts),
        "avg_skipped": statistics.fmean(skipped_counts),
    }


# --------------------------------------------------------------------------- #
# Deterministic in-sample path (true chronological panel order)               #
# --------------------------------------------------------------------------- #
def deterministic_path(units, base_size, kelly_frac, dd_power, dd_budget,
                       start=START_BAL, min_fire=1.0):
    bal = start
    peak = start
    run_dd = 0.0
    placed = 0
    bet_pnls = []
    for u in units:
        stake = dd_constrained_stake(base_size, kelly_frac, dd_power,
                                     bal, peak, dd_budget, min_fire)
        if stake <= 0.0:
            continue
        pnl = stake * u
        bal += pnl
        placed += 1
        bet_pnls.append(pnl)
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > run_dd:
            run_dd = dd
    return {
        "final_bal": bal,
        "max_dd": run_dd,
        "placed": placed,
        "total_pnl": sum(bet_pnls),
        "median_pnl": statistics.median(bet_pnls) if bet_pnls else 0.0,
        "mean_pnl": statistics.fmean(bet_pnls) if bet_pnls else 0.0,
    }


# --------------------------------------------------------------------------- #
# Main sweep                                                                   #
# --------------------------------------------------------------------------- #
def main():
    with open(PANEL_PATH) as f:
        panel = json.load(f)
    n_total = len(panel)
    units, entries, wins = extract_units(panel)
    n_fired = len(units)
    wr = wins / n_fired

    print(f"Loaded {n_total} markets")
    print(f"EDGE: FADE coin-flip[0.40,0.60], spread<=0.02 both, |20s move|>2.0c, "
          f"first-trigger cd<=200, exclude cd<=15")
    print(f"  fired={n_fired} ({100*n_fired/n_total:.1f}%)  WR={100*wr:.1f}%  "
          f"mean_unit=${statistics.fmean(units):+.4f}  "
          f"median_unit=${statistics.median(units):+.4f}  "
          f"std_unit={statistics.pstdev(units):.4f}  "
          f"total_unit=${sum(units):+.2f}/$1")

    fk, fk_growth = full_kelly_fraction(units)
    print(f"\nKELLY: full-Kelly f* = {fk:.4f}  (bet {fk*100:.2f}% of bankroll; "
          f"log-growth/bet={fk_growth:.5f})")
    print(f"  At ${START_BAL:.0f} bankroll, full Kelly = ${fk*START_BAL:.3f} "
          f"-> BELOW $1 PM-min. Edge is real but THIN; true Kelly is sub-minimum.")
    print(f"  => We study fractional-Kelly with a fixed $base + DD throttle, "
          f"$1 floor (skip if sub-min).\n")

    # ---- sweep the DD-constrained sizing space ----
    base_sizes  = [1, 2, 3, 4]
    kelly_fracs = [0.25, 0.5, 1.0]      # fractional-Kelly multiplier on base
    dd_powers   = [0.0, 1.0, 2.0, 3.0]  # 0 = flat (control); >=1 = convex throttle
    dd_budgets  = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.99]

    results = []
    for base in base_sizes:
        for kf in kelly_fracs:
            for dp in dd_powers:
                for budget in dd_budgets:
                    mc = simulate_config(units, base, kf, dp, budget)
                    if mc is None:
                        continue
                    det = deterministic_path(units, base, kf, dp, budget)
                    placed_frac = mc["avg_placed"] / n_fired if n_fired else 0.0
                    cfg = {
                        "family": "kelly-dd-constrained",
                        "base_size": base,
                        "kelly_frac": kf,
                        "dd_power": dp,
                        "dd_budget": budget,
                        "median_pnl": mc["median_pnl"],
                        "mean_pnl": mc["mean_pnl"],
                        "max_dd_worst": mc["max_dd_worst"],
                        "max_dd_mean": mc["max_dd_mean"],
                        "max_dd_p95": mc["max_dd_p95"],
                        "wipe_rate": mc["wipe_rate"],
                        "mean_final_bal": mc["mean_final_bal"],
                        "avg_placed": mc["avg_placed"],
                        "avg_skipped": mc["avg_skipped"],
                        "placed_of_fired_frac": placed_frac,
                        # % of ALL markets that actually got a placed bet (avg)
                        "pct_markets_fired": 100.0 * (mc["avg_placed"] / n_total),
                        "det_final_bal": det["final_bal"],
                        "det_max_dd": det["max_dd"],
                        "det_placed": det["placed"],
                        "det_total_pnl": det["total_pnl"],
                        "det_median": det["median_pnl"],
                    }
                    results.append(cfg)

    # GUARANTEE filter: worst-case maxDD across 2000 runs < $5 AND no wipes.
    safe = [c for c in results if c["max_dd_worst"] < 5.0 and c["wipe_rate"] == 0.0]
    # target = also median per-market PnL >= +$1
    target = [c for c in safe if c["median_pnl"] >= 1.0]

    # ------------------------------------------------------------------ #
    # COVERAGE GUARD against a gamed median.                              #
    # ------------------------------------------------------------------ #
    # The median is order/coverage-sensitive: a config that places only a
    # handful of bets can show a high median purely because the tiny early
    # sequence was win-heavy. To report an HONEST ceiling we distinguish:
    #   * best_metric : pure max-median (may be a low-coverage artifact)
    #   * best_safe   : the honest deployable ceiling = the config meeting the
    #                   target (median>=$1) with the MOST coverage; if none
    #                   meet the target, the max-coverage DD-safe config (the
    #                   most bets you can place at all while guaranteeing DD<5).
    best_metric = max(safe, key=lambda c: (c["median_pnl"], c["mean_pnl"])) if safe else None
    if target:
        # honest: among target-meeting configs, maximize coverage (avg_placed),
        # tie-break on mean PnL then median.
        best_safe = max(target, key=lambda c: (c["avg_placed"], c["mean_pnl"],
                                               c["median_pnl"]))
    else:
        best_safe = max(safe, key=lambda c: (c["avg_placed"], c["median_pnl"])) \
            if safe else None
    # The single highest-coverage DD-safe config (the structural fire ceiling).
    max_cov = max(safe, key=lambda c: c["avg_placed"]) if safe else None

    print("=" * 100)
    print(f"DD-GUARANTEED-SAFE configs (worst-case maxDD<$5 AND wipe==0): "
          f"{len(safe)} of {len(results)}")
    print(f"TARGET hits (also median>=+$1): {len(target)}\n")

    print("TOP 12 DD-safe configs by MEDIAN per-market PnL:")
    for c in sorted(safe, key=lambda c: (-c["median_pnl"], -c["mean_pnl"]))[:12]:
        print(f"  med=${c['median_pnl']:+.3f} mean=${c['mean_pnl']:+.3f} "
              f"ddWorst=${c['max_dd_worst']:.2f} ddMean=${c['max_dd_mean']:.2f} "
              f"wipe={c['wipe_rate']*100:.1f}% placed/mkt={c['pct_markets_fired']:.0f}% "
              f"| base=${c['base_size']} kf={c['kelly_frac']} pow={c['dd_power']} "
              f"budget=${c['dd_budget']}")

    print("\nTOP 12 DD-safe configs by MEAN per-market PnL:")
    for c in sorted(safe, key=lambda c: -c["mean_pnl"])[:12]:
        print(f"  mean=${c['mean_pnl']:+.3f} med=${c['median_pnl']:+.3f} "
              f"ddWorst=${c['max_dd_worst']:.2f} wipe={c['wipe_rate']*100:.1f}% "
              f"placed/mkt={c['pct_markets_fired']:.0f}% finalBal=${c['mean_final_bal']:.2f} "
              f"| base=${c['base_size']} kf={c['kelly_frac']} pow={c['dd_power']} "
              f"budget=${c['dd_budget']}")

    print("\n" + "=" * 100)
    print("STRUCTURAL FIRE CEILING (max-coverage DD-safe config):")
    if max_cov:
        c = max_cov
        print(f"  avg_placed={c['avg_placed']:.1f}/{n_fired} fired "
              f"({100*c['avg_placed']/n_fired:.1f}% of edge fires, "
              f"{c['pct_markets_fired']:.1f}% of all markets)  "
              f"median=${c['median_pnl']:+.3f}  mean=${c['mean_pnl']:+.3f}  "
              f"maxDD=${c['max_dd_worst']:.2f}")
        print("  => DD<$5 from $9 with $1-min and full-stake loss caps placed bets")
        print("     at ~5 consecutive losses before headroom<$1 -> structural skip.")

    print("\n" + "=" * 100)
    if best_metric:
        c = best_metric
        print("PURE METRIC-MAX MEDIAN (CAUTION: may be a low-coverage artifact):")
        print(f"  median=${c['median_pnl']:+.3f}  mean=${c['mean_pnl']:+.3f}  "
              f"maxDD=${c['max_dd_worst']:.3f}  avg_placed={c['avg_placed']:.1f} "
              f"| base=${c['base_size']} kf={c['kelly_frac']} pow={c['dd_power']} "
              f"budget=${c['dd_budget']}")

    print("\n" + "=" * 100)
    if best_safe:
        c = best_safe
        hdr = ("HONEST CEILING (highest-coverage config that MEETS median>=$1 "
               "with GUARANTEED maxDD<$5):" if target else
               "HONEST CEILING (no config meets median>=$1; reporting "
               "max-coverage DD-safe config):")
        print(hdr)
        print(f"  median=${c['median_pnl']:+.3f}  mean=${c['mean_pnl']:+.3f}  "
              f"maxDD(worst of 2000)=${c['max_dd_worst']:.3f}  "
              f"wipe={c['wipe_rate']*100:.1f}%")
        print(f"  placed/mkt={c['pct_markets_fired']:.1f}%  "
              f"avg_placed={c['avg_placed']:.1f}/{n_fired}  "
              f"det_final_bal=${c['det_final_bal']:.2f} det_maxDD=${c['det_max_dd']:.2f}")
        print(f"  SIZING: stake = ${c['base_size']} * {c['kelly_frac']}(Kelly-frac) "
              f"* clamp((budget-DD)/budget,0,1)^{c['dd_power']}, "
              f"budget=${c['dd_budget']}, hard-clamp stake<=headroom, "
              f"skip if stake<$1")

    target_met = bool(target)
    answer = {
        "family": "kelly-dd-constrained",
        "n_markets": n_total,
        "n_fired_edge": n_fired,
        "edge_wr_pct": round(100 * wr, 2),
        "edge_mean_unit": statistics.fmean(units),
        "edge_median_unit": statistics.median(units),
        "edge_total_unit": sum(units),
        "full_kelly_f": fk,
        "full_kelly_dollars_at_9": fk * START_BAL,
        "best_safe": best_safe,        # honest, coverage-aware ceiling
        "best_metric": best_metric,    # pure max-median (may be artifact)
        "max_cov": max_cov,            # structural fire ceiling
        "target_met": target_met,
        "n_safe": len(safe),
        "n_target": len(target),
    }
    with open("./data/.kelly_dd_constrained_summary.json", "w") as fh:
        json.dump({"answer": answer, "all_results": results}, fh, indent=2)
    print("\n[wrote ./.kelly_dd_constrained_summary.json]")
    return answer, best_safe, units, n_total, n_fired


if __name__ == "__main__":
    main()
