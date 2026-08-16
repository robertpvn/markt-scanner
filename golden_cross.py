#!/usr/bin/env python3
"""
Golden-cross scanner — S&P 500 + Nasdaq-100, weekchart
Zoekt aandelen waarvan het 50-weeks gemiddelde op het punt staat het 200-weeks
gemiddelde van onderaf te kruisen.

Twee categorieën:
  1) NADERT      — MA50 ligt nog onder MA200, maar het gat krimpt en de projectie
                   op basis van de huidige helling komt binnen 13 weken uit.
  2) NET GEKRUIST — de kruising is de afgelopen 8 weken daadwerkelijk gebeurd.

Output: GOLDENCROSS_REPORT.md en data/goldencross/signals.json
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import scanner  # hergebruikt build_universe() en download_weekly()

# ---------------------------- instellingen ----------------------------
MA_FAST = 50             # weken
MA_SLOW = 200            # weken
PROJECT_MAX_WEEKS = 13   # projectie moet binnen een kwartaal uitkomen
SLOPE_WEEKS = 8          # venster waarover de helling van het gat gemeten wordt
CONFIRM_WEEKS = 8        # hoe lang een verse kruising nog gemeld wordt
MIN_BARS = MA_SLOW + SLOPE_WEEKS

DATA_DIR = os.path.join("data", "goldencross")
REPORT = "GOLDENCROSS_REPORT.md"

STATE_NEAR = "nadert"
STATE_CROSSED = "net gekruist"


# ---------------------------- analyse ---------------------------------
def analyse(df: pd.DataFrame) -> dict | None:
    """Beoordeelt één aandeel. Retourneert None als er niets te melden valt."""
    df = df.dropna(subset=["Close"])
    if len(df) < MIN_BARS:
        return None

    close = df["Close"]
    fast = close.rolling(MA_FAST).mean()
    slow = close.rolling(MA_SLOW).mean()
    diff = (fast - slow).dropna()
    if len(diff) < SLOPE_WEEKS + 1:
        return None

    ma_f, ma_s = float(fast.iloc[-1]), float(slow.iloc[-1])
    if not (np.isfinite(ma_f) and np.isfinite(ma_s)) or ma_s <= 0:
        return None

    price = float(close.iloc[-1])
    gap_pct = (ma_f - ma_s) / ma_s * 100
    base = {
        "close": round(price, 2),
        "ma_fast": round(ma_f, 2),
        "ma_slow": round(ma_s, 2),
        "gap_pct": round(gap_pct, 2),
        "week_end": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        "price_above_fast": bool(price > ma_f),
        "price_above_slow": bool(price > ma_s),
    }

    # 1) is de kruising recent al gebeurd?
    recent = diff.iloc[-(CONFIRM_WEEKS + 1):]
    signs = np.sign(recent.to_numpy(dtype=float))
    if signs[-1] > 0:
        neg = np.where(signs <= 0)[0]
        if len(neg):                      # ergens in dit venster stond hij nog onder nul
            weeks_since = len(signs) - 1 - int(neg[-1])
            base.update({"state": STATE_CROSSED, "weeks_since_cross": weeks_since,
                         "weeks_to_cross": None})
            return base
        return None                       # al langer boven: oud nieuws

    # 2) nadert hij? helling van het gat over de laatste SLOPE_WEEKS weken
    window = diff.iloc[-(SLOPE_WEEKS + 1):].to_numpy(dtype=float)
    x = np.arange(len(window), dtype=float)
    slope = float(np.polyfit(x, window, 1)[0])        # verandering van het gat per week
    diff_now = float(diff.iloc[-1])
    if slope <= 0 or diff_now >= 0:
        return None                                   # loopt niet naar elkaar toe

    weeks = -diff_now / slope
    if not np.isfinite(weeks) or weeks <= 0 or weeks > PROJECT_MAX_WEEKS:
        return None

    base.update({"state": STATE_NEAR,
                 "weeks_to_cross": round(weeks, 1),
                 "weeks_since_cross": None,
                 "gap_narrowing_per_week": round(slope / ma_s * 100, 3)})
    return base


# ---------------------------- rapportage ------------------------------
def write_outputs(hits: list[dict], universe_size: int, scanned: int, warnings: list[str]):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d")
    os.makedirs(os.path.join(DATA_DIR, "history"), exist_ok=True)

    near = sorted([h for h in hits if h["state"] == STATE_NEAR],
                  key=lambda h: h["weeks_to_cross"])
    crossed = sorted([h for h in hits if h["state"] == STATE_CROSSED],
                     key=lambda h: h["weeks_since_cross"])

    result = {
        "scan_date_utc": now.isoformat(timespec="seconds"),
        "market": "aandelen",
        "criteria": {
            "paar": f"{MA_FAST}-weeks SMA kruist boven {MA_SLOW}-weeks SMA",
            "nadert": f"gat krimpt en projectie op basis van de helling over "
                      f"{SLOPE_WEEKS} weken valt binnen {PROJECT_MAX_WEEKS} weken",
            "net_gekruist": f"kruising vond plaats in de afgelopen {CONFIRM_WEEKS} weken",
        },
        "universe_size": universe_size,
        "tickers_scanned": scanned,
        "approaching_count": len(near),
        "crossed_count": len(crossed),
        "warnings": warnings,
        "approaching": near,
        "recently_crossed": crossed,
    }
    with open(os.path.join(DATA_DIR, "signals.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(DATA_DIR, "history", f"{stamp}.json"), "w") as f:
        json.dump(result, f, indent=1)

    def positie(h):
        if h["price_above_fast"]:
            return "boven beide MA's" if h["price_above_slow"] else "boven MA50"
        return "boven MA200" if h["price_above_slow"] else "onder beide MA's"

    def table(rows, kind):
        kolom = "Verwacht over" if kind == STATE_NEAR else "Gekruist"
        lines = [
            f"| Ticker | Bedrijf | Index | Slot | MA50 | MA200 | Gat | {kolom} | Koers staat |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for h in rows:
            wanneer = (f"{h['weeks_to_cross']} wk" if kind == STATE_NEAR
                       else f"{h['weeks_since_cross']} wk geleden")
            lines.append(
                f"| **{h['ticker']}** | {h['name']} | {h['index']} | {h['close']} "
                f"| {h['ma_fast']} | {h['ma_slow']} | {h['gap_pct']:+.1f}% | {wanneer} "
                f"| {positie(h)} |"
            )
        return "\n".join(lines)

    md = [
        f"# Golden-cross scan — {stamp}",
        "",
        f"*{MA_FAST}-weeks tegen {MA_SLOW}-weeks SMA op de weekchart · "
        f"universum: S&P 500 + Nasdaq-100 ({universe_size} tickers, {scanned} gescand) · "
        f"projectie op basis van de helling van het gat over de laatste {SLOPE_WEEKS} weken.*",
        "",
    ]
    if warnings:
        md += ["> ⚠️ " + " | ".join(warnings), ""]

    if near:
        md += [f"## {len(near)} naderen een golden cross", "", table(near, STATE_NEAR), ""]
    else:
        md += ["## Geen naderende golden crosses", "",
               f"Bij geen enkel aandeel komt de projectie binnen {PROJECT_MAX_WEEKS} weken uit.", ""]

    if crossed:
        md += [f"## {len(crossed)} zijn net gekruist", "",
               f"Bij deze aandelen is de golden cross de afgelopen {CONFIRM_WEEKS} weken "
               "daadwerkelijk voltooid.", "", table(crossed, STATE_CROSSED), ""]

    md += [
        "*De verwachte kruising is een rechttoe-rechtaan doortrekking van de huidige helling. "
        "Draait de koers, dan verschuift die datum mee — het is een waarschuwing dat het "
        "eraan zit te komen, geen voorspelling. Een golden cross op de weekchart is een "
        "traag trendsignaal: het 50-weeks gemiddelde beslaat een jaar, het 200-weeks bijna vier.*",
        "",
        "_Databron: Yahoo Finance (koersen split-adjusted). Geen beleggingsadvies._",
        "",
    ]
    with open(REPORT, "w") as f:
        f.write("\n".join(md))

    alert_md = os.path.join(DATA_DIR, "alert.md")
    alert_title = os.path.join(DATA_DIR, "alert_title.txt")
    for p in (alert_md, alert_title):
        if os.path.exists(p):
            os.remove(p)
    if near or crossed:
        stukken = []
        if near:
            stukken.append(f"{len(near)} naderend")
        if crossed:
            stukken.append(f"{len(crossed)} net gekruist")
        with open(alert_title, "w") as f:
            f.write(f"✨ Golden-cross scan {stamp}: " + ", ".join(stukken))
        body = [f"Scan van {stamp} op de weekchart ({MA_FAST}/{MA_SLOW}).", ""]
        if near:
            body += [f"**{len(near)} naderen een golden cross:**", "", table(near[:40], STATE_NEAR), ""]
        if crossed:
            body += [f"**{len(crossed)} net gekruist:**", "", table(crossed[:40], STATE_CROSSED), ""]
        body.append("_Geen beleggingsadvies._")
        with open(alert_md, "w") as f:
            f.write("\n".join(body))


# ---------------------------- main ------------------------------------
def main():
    universe, warnings = scanner.build_universe()
    tickers = sorted(universe)
    print(f"Universum: {len(tickers)} tickers")

    frames = scanner.download_weekly(tickers)
    print(f"Data ontvangen voor {len(frames)} tickers")
    if len(frames) < len(tickers) * 0.5:
        warnings.append(f"slechts {len(frames)}/{len(tickers)} tickers met data")

    hits = []
    for t, df in frames.items():
        try:
            res = analyse(df)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {t}: {e}", file=sys.stderr)
            continue
        if res:
            res.update({"ticker": t, "name": universe[t]["name"],
                        "index": " + ".join(universe[t]["indices"])})
            hits.append(res)

    write_outputs(hits, len(tickers), len(frames), warnings)
    n_near = sum(1 for h in hits if h["state"] == STATE_NEAR)
    print(f"Klaar: {n_near} naderend, {len(hits) - n_near} net gekruist. Zie {REPORT}")


if __name__ == "__main__":
    main()
