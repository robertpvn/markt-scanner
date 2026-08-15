#!/usr/bin/env python3
"""
Wekelijkse marktscanner — S&P 500 + Nasdaq-100
Signalen op de WEEKCHART:
  1) Koers tikt de 200-weeks SMA aan (week-range omsluit de MA, of slotkoers binnen ±2%)
  2) RSI(14, weekly, Wilder) < 30

Output:
  data/signals.json        — machineleesbaar resultaat van de laatste scan
  data/history/<datum>.json— archief per scan
  REPORT.md                — leesbaar rapport (Nederlands)
  data/alert.md            — alleen aanwezig als er signalen zijn (trigger voor GitHub-issue)
  data/alert_title.txt     — titel voor het issue
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from io import StringIO

import pandas as pd

# ---------------------------- instellingen ----------------------------
MA_PERIOD = 200          # weken
MA_TOUCH_PCT = 0.02      # ±2% telt als "aantikken"
RSI_PERIOD = 14
RSI_THRESHOLD = 30.0
MIN_BARS_RSI = 30        # minimaal aantal weekbars voor betrouwbare RSI
DATA_DIR = "data"
UNIVERSE_CACHE = os.path.join(DATA_DIR, "universe.json")

UA = {"User-Agent": "Mozilla/5.0 (compatible; markt-scanner/1.0)"}


# ---------------------------- universum -------------------------------
def _wiki_table(url: str):
    import requests
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def fetch_sp500():
    tables = _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    for t in tables:
        if "Symbol" in t.columns:
            name_col = "Security" if "Security" in t.columns else t.columns[1]
            return {str(r["Symbol"]).strip(): str(r[name_col]).strip() for _, r in t.iterrows()}
    raise RuntimeError("S&P 500-tabel niet gevonden op Wikipedia")


def fetch_ndx():
    tables = _wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100")
    for t in tables:
        cols = [str(c) for c in t.columns]
        tick_col = next((c for c in ("Ticker", "Symbol") if c in cols), None)
        if tick_col and len(t) > 50:
            name_col = next((c for c in ("Company", "Security") if c in cols), cols[0])
            return {str(r[tick_col]).strip(): str(r[name_col]).strip() for _, r in t.iterrows()}
    raise RuntimeError("Nasdaq-100-tabel niet gevonden op Wikipedia")


def build_universe():
    """{ticker: {"name":..., "indices":[...]}}, met cache als vangnet."""
    universe, errors = {}, []
    for fetch, label in ((fetch_sp500, "S&P 500"), (fetch_ndx, "Nasdaq-100")):
        try:
            for tick, name in fetch().items():
                yahoo = tick.replace(".", "-")  # BRK.B -> BRK-B
                entry = universe.setdefault(yahoo, {"name": name, "indices": []})
                entry["indices"].append(label)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label}: {e}")

    if universe and not errors:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(UNIVERSE_CACHE, "w") as f:
            json.dump(universe, f, indent=1)
        return universe, []

    # vangnet: eerder gecachete lijst
    if os.path.exists(UNIVERSE_CACHE):
        with open(UNIVERSE_CACHE) as f:
            cached = json.load(f)
        if universe:  # deels gelukt: aanvullen met cache
            for tick, info in cached.items():
                universe.setdefault(tick, info)
            return universe, errors
        return cached, errors + ["universum volledig uit cache geladen"]

    if universe:
        return universe, errors
    raise RuntimeError("Geen universum beschikbaar: " + "; ".join(errors))


# ---------------------------- indicatoren -----------------------------
def rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)  # geen verliezen in de periode -> RSI 100
    return rsi


def evaluate(df: pd.DataFrame) -> dict | None:
    """df: weekbars met kolommen Open/High/Low/Close. Retourneert signaalinfo of None."""
    df = df.dropna(subset=["Close"])
    if len(df) < MIN_BARS_RSI:
        return None

    close = df["Close"]
    last = df.iloc[-1]
    price = float(last["Close"])

    out = {"close": round(price, 2), "week_end": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]), "signals": []}

    # RSI
    rsi = rsi_wilder(close)
    rsi_now = float(rsi.iloc[-1])
    out["rsi"] = round(rsi_now, 1)
    if rsi_now < RSI_THRESHOLD:
        out["signals"].append("RSI<30")

    # 200-weeks SMA
    if len(df) >= MA_PERIOD:
        ma = float(close.rolling(MA_PERIOD).mean().iloc[-1])
        dist = (price - ma) / ma
        out["sma200w"] = round(ma, 2)
        out["dist_pct"] = round(dist * 100, 2)
        lo = float(last["Low"]) if "Low" in last and pd.notna(last["Low"]) else price
        hi = float(last["High"]) if "High" in last and pd.notna(last["High"]) else price
        touched = (lo <= ma <= hi) or (abs(dist) <= MA_TOUCH_PCT)
        if touched:
            out["signals"].append("200WMA")
    else:
        out["sma200w"] = None
        out["dist_pct"] = None

    return out if out["signals"] else None


# ---------------------------- data ophalen ----------------------------
def download_weekly(tickers: list[str], chunk: int = 100) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch, period="6y", interval="1wk",
                    auto_adjust=False, group_by="ticker",
                    threads=True, progress=False,
                )
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(15)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            for t in batch:
                if t in raw.columns.get_level_values(0):
                    sub = raw[t].dropna(how="all")
                    if not sub.empty:
                        frames[t] = sub
        else:  # één ticker
            frames[batch[0]] = raw.dropna(how="all")
        time.sleep(2)
    return frames


# ---------------------------- rapportage ------------------------------
def write_outputs(hits: list[dict], universe_size: int, scanned: int, warnings: list[str]):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d")
    os.makedirs(os.path.join(DATA_DIR, "history"), exist_ok=True)

    hits.sort(key=lambda h: (h["rsi"] if "RSI<30" in h["signals"] else 100, abs(h.get("dist_pct") or 99)))

    result = {
        "scan_date_utc": now.isoformat(timespec="seconds"),
        "criteria": {
            "ma": f"koers raakt 200-weeks SMA (week-range of ±{MA_TOUCH_PCT:.0%})",
            "rsi": f"RSI({RSI_PERIOD}) weekly < {RSI_THRESHOLD:.0f}",
        },
        "universe_size": universe_size,
        "tickers_scanned": scanned,
        "signal_count": len(hits),
        "warnings": warnings,
        "signals": hits,
    }
    with open(os.path.join(DATA_DIR, "signals.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(DATA_DIR, "history", f"{stamp}.json"), "w") as f:
        json.dump(result, f, indent=1)

    # REPORT.md
    def table(rows):
        lines = [
            "| Ticker | Bedrijf | Index | Signaal | Slot | RSI (w) | 200WMA | Afstand |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
        for h in rows:
            ma = f"{h['sma200w']}" if h.get("sma200w") else "–"
            dist = f"{h['dist_pct']:+.1f}%" if h.get("dist_pct") is not None else "–"
            lines.append(
                f"| **{h['ticker']}** | {h['name']} | {h['index']} | {' + '.join(h['signals'])} "
                f"| {h['close']} | {h['rsi']} | {ma} | {dist} |"
            )
        return "\n".join(lines)

    md = [
        f"# Wekelijkse marktscan — {stamp}",
        "",
        f"*Universum: S&P 500 + Nasdaq-100 ({universe_size} tickers, {scanned} gescand) · weekchart · "
        f"criteria: 200-weeks SMA aangetikt (week-range of ±2%) en/of RSI(14) < 30.*",
        "",
    ]
    if warnings:
        md += ["> ⚠️ " + " | ".join(warnings), ""]
    if hits:
        md += [f"## {len(hits)} signalen", "", table(hits)]
    else:
        md += ["## Geen signalen deze week", "", "Geen enkel aandeel voldeed aan de criteria."]
    md += ["", "_Databron: Yahoo Finance (koersen split-adjusted). Geen beleggingsadvies._", ""]
    with open("REPORT.md", "w") as f:
        f.write("\n".join(md))

    # alert-bestanden (alleen bij signalen) -> GitHub-issue + e-mail
    alert_md = os.path.join(DATA_DIR, "alert.md")
    alert_title = os.path.join(DATA_DIR, "alert_title.txt")
    for p in (alert_md, alert_title):
        if os.path.exists(p):
            os.remove(p)
    if hits:
        both = [h for h in hits if len(h["signals"]) == 2]
        title = f"📉 Marktscan {stamp}: {len(hits)} signalen" + (f" (waarvan {len(both)} dubbel)" if both else "")
        body = [
            f"De wekelijkse scan van {stamp} vond **{len(hits)} aandelen** die aan de criteria voldoen.",
            "",
            table(hits[:50]),
        ]
        if len(hits) > 50:
            body.append(f"\n… en nog {len(hits) - 50} meer — zie REPORT.md in de repository.")
        body.append("\n_Geen beleggingsadvies._")
        with open(alert_md, "w") as f:
            f.write("\n".join(body))
        with open(alert_title, "w") as f:
            f.write(title)


# ---------------------------- main -------------------------------------
def main():
    universe, warnings = build_universe()
    tickers = sorted(universe)
    print(f"Universum: {len(tickers)} tickers")

    frames = download_weekly(tickers)
    print(f"Data ontvangen voor {len(frames)} tickers")
    if len(frames) < len(tickers) * 0.5:
        warnings.append(f"slechts {len(frames)}/{len(tickers)} tickers met data — resultaat mogelijk onvolledig")

    hits = []
    for t, df in frames.items():
        try:
            res = evaluate(df)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {t}: {e}", file=sys.stderr)
            continue
        if res:
            res.update({
                "ticker": t,
                "name": universe[t]["name"],
                "index": " + ".join(universe[t]["indices"]),
            })
            hits.append(res)

    write_outputs(hits, len(tickers), len(frames), warnings)
    print(f"Klaar: {len(hits)} signalen. Zie REPORT.md en data/signals.json")


if __name__ == "__main__":
    main()
