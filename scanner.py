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
import re
import sys
import time
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd

# ---------------------------- instellingen ----------------------------
MA_PERIOD = 200          # weken
MA_TOUCH_PCT = 0.02      # ±2% telt als "aantikken"
RSI_PERIOD = 14
RSI_THRESHOLD = 30.0
MIN_BARS_RSI = 30        # minimaal aantal weekbars voor betrouwbare RSI

# --- bullish divergentie (weekchart) ---
PIVOT_L = 3              # weken links van een bodem
PIVOT_R = 3              # weken rechts (bodem is pas na 3 weken bevestigd)
DIV_LOOKBACK = 60        # zoekvenster in weken
DIV_MAX_AGE = 8          # tweede bodem mag max zo oud zijn
DIV_MIN_SEP = 4          # minimale afstand tussen de twee bodems
DIV_MAX_SEP = 40         # maximale afstand
DIV_RSI_MAX_REG = 32.0   # regulier: minstens één bodem in oversold-gebied
DIV_RSI_MAX_HID = 42.0   # verborgen: minstens één bodem in pullback-zone
DIV_MIN_PRICE_DELTA = 0.02   # bodems moeten ≥2% verschillen
DIV_MIN_RSI_DELTA = 5.0      # RSI-bodems moeten ≥5 punten verschillen
# Deze drempels zijn geijkt op willekeurige koersreeksen: ~2% valse treffers,
# oftewel ruwweg 10-20 divergenties per week op een universum van 600 aandelen.

SIG_RSI = "RSI<30"
SIG_MA = "200WMA"            # slotkoers binnen ±2% van de 200-weeks SMA
SIG_MA_INTRA = "200WMA-intraweek"  # week-range raakte de MA, slot verder weg
SIG_DIV = "Bull.div"         # reguliere bullish divergentie
SIG_DIV_HID = "Verborgen bull.div"
# signalen die een melding waard zijn (SIG_MA_INTRA is alleen ter info)
CORE = {SIG_RSI, SIG_MA, SIG_DIV, SIG_DIV_HID}

DATA_DIR = "data"
UNIVERSE_CACHE = os.path.join(DATA_DIR, "universe.json")

UA = {"User-Agent": "Mozilla/5.0 (compatible; markt-scanner/1.0)"}


# ---------------------------- universum -------------------------------
def _wiki_table(url: str):
    import requests
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def _looks_like_tickers(col: pd.Series) -> float:
    """Aandeel van de waarden dat op een ticker lijkt (0..1)."""
    vals = [str(v).strip() for v in col.dropna()]
    if not vals:
        return 0.0
    return sum(bool(TICKER_RE.match(v)) for v in vals) / len(vals)


def _pick_constituents(tables, min_rows: int, max_rows: int) -> dict[str, str] | None:
    """Zoek de constituententabel op INHOUD (werkt ook als de kopregel niet herkend is)."""
    best = None
    for t in tables:
        if not (min_rows <= len(t) <= max_rows) or t.shape[1] < 2:
            continue
        t = t.copy()
        t.columns = [str(c) for c in t.columns]
        # kopregel niet herkend? (kolomnamen 0,1,2,...) -> eerste rij als kop gebruiken
        if all(c.isdigit() for c in t.columns) and len(t) > min_rows:
            first = [str(v).strip() for v in t.iloc[0]]
            if any(v.lower() in ("ticker", "symbol") for v in first):
                t.columns = first
                t = t.iloc[1:]
        scores = {c: _looks_like_tickers(t[c]) for c in t.columns}
        tick_col = max(scores, key=scores.get)
        if scores[tick_col] < 0.9:
            continue
        # naamkolom: eerst op naam zoeken, anders de langste tekstkolom
        named = [c for c in t.columns if c.strip().lower() in ("company", "security", "company name", "name")]
        if named:
            name_col = named[0]
        else:
            others = [c for c in t.columns if c != tick_col]
            name_col = max(others, key=lambda c: t[c].astype(str).str.len().mean()) if others else tick_col
        mapping = {}
        for _, r in t.iterrows():
            tick = str(r[tick_col]).strip()
            if TICKER_RE.match(tick):
                mapping[tick] = str(r[name_col]).strip()
        if len(mapping) >= min_rows and (best is None or len(mapping) > len(best)):
            best = mapping
    return best


def fetch_sp500():
    tables = _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    found = _pick_constituents(tables, 400, 600)
    if found:
        return found
    raise RuntimeError("S&P 500-tabel niet gevonden op Wikipedia")


def fetch_ndx():
    try:
        tables = _wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100")
        found = _pick_constituents(tables, 90, 115)
        if found:
            return found
        cols = [list(map(str, t.columns))[:5] for t in tables if len(t) > 50]
        print(f"  ! Nasdaq-100: geen tabel herkend. Kandidaten: {cols}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"  ! Nasdaq-100 via Wikipedia mislukt: {e}", file=sys.stderr)

    # reservebron
    tables = _wiki_table("https://www.slickcharts.com/nasdaq100")
    found = _pick_constituents(tables, 90, 115)
    if found:
        return found
    raise RuntimeError("Nasdaq-100-tabel niet gevonden (Wikipedia noch slickcharts)")


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


def pivot_lows(low: pd.Series, left: int = PIVOT_L, right: int = PIVOT_R) -> list[int]:
    """Indexposities van bevestigde bodems (laagste punt binnen ±N weken)."""
    vals = low.to_numpy(dtype=float)
    out: list[int] = []
    for i in range(left, len(vals) - right):
        window = vals[i - left: i + right + 1]
        if not np.isfinite(window).all():
            continue
        # eerste voorkomen van het minimum -> geen dubbele bodems bij gelijke koersen
        if i - left + int(window.argmin()) == i:
            out.append(i)
    return out


def detect_divergence(df: pd.DataFrame, rsi: pd.Series, ma: float | None) -> dict | None:
    """Bullish divergentie tussen de laatste twee koersbodems op de weekchart.

    Regulier : koers lagere bodem, RSI hogere bodem (omkeersignaal na een daling).
    Verborgen: koers hogere bodem, RSI lagere bodem (voortzetting in een stijgende trend).
    """
    if len(df) < DIV_MIN_SEP + PIVOT_L + PIVOT_R + RSI_PERIOD:
        return None

    low = df["Low"] if "Low" in df.columns else df["Close"]
    n = len(df)
    pivots = [i for i in pivot_lows(low) if n - 1 - i <= DIV_LOOKBACK and np.isfinite(rsi.iloc[i])]
    if len(pivots) < 2:
        return None

    i2 = pivots[-1]
    if n - 1 - i2 > DIV_MAX_AGE:
        return None

    price_now = float(df["Close"].iloc[-1])
    if ma is not None:
        uptrend = price_now > ma
    elif n >= 26:
        uptrend = price_now > float(df["Close"].iloc[-26])
    else:
        uptrend = False

    l2, r2 = float(low.iloc[i2]), float(rsi.iloc[i2])

    for i1 in reversed(pivots[:-1]):
        sep = i2 - i1
        if sep < DIV_MIN_SEP:
            continue
        if sep > DIV_MAX_SEP:
            break
        l1, r1 = float(low.iloc[i1]), float(rsi.iloc[i1])
        if not (np.isfinite(l1) and np.isfinite(r1)):
            continue

        lower_low = l2 < l1 * (1 - DIV_MIN_PRICE_DELTA)
        higher_low = l2 > l1 * (1 + DIV_MIN_PRICE_DELTA)
        rsi_up = r2 > r1 + DIV_MIN_RSI_DELTA
        rsi_down = r2 < r1 - DIV_MIN_RSI_DELTA

        kind = None
        if lower_low and rsi_up and min(r1, r2) < DIV_RSI_MAX_REG:
            kind = "regulier"
        elif higher_low and rsi_down and uptrend and min(r1, r2) < DIV_RSI_MAX_HID:
            kind = "verborgen"
        if kind:
            def stamp(i):
                v = df.index[i]
                return str(v.date()) if hasattr(v, "date") else str(v)
            return {
                "type": kind,
                "weeks_apart": sep,
                "age_weeks": n - 1 - i2,
                "low1": round(l1, 2), "low2": round(l2, 2),
                "rsi1": round(r1, 1), "rsi2": round(r2, 1),
                "date1": stamp(i1), "date2": stamp(i2),
            }
    return None


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
        out["signals"].append(SIG_RSI)

    # 200-weeks SMA
    ma = None
    if len(df) >= MA_PERIOD:
        ma = float(close.rolling(MA_PERIOD).mean().iloc[-1])
        dist = (price - ma) / ma
        out["sma200w"] = round(ma, 2)
        out["dist_pct"] = round(dist * 100, 2)
        lo = float(last["Low"]) if "Low" in last and pd.notna(last["Low"]) else price
        hi = float(last["High"]) if "High" in last and pd.notna(last["High"]) else price
        if abs(dist) <= MA_TOUCH_PCT:
            out["signals"].append(SIG_MA)          # slotkoers binnen ±2% van de MA
        elif lo <= ma <= hi:
            out["signals"].append(SIG_MA_INTRA)    # alleen intraweek aangeraakt
    else:
        out["sma200w"] = None
        out["dist_pct"] = None

    # bullish divergentie
    div = detect_divergence(df, rsi, ma)
    out["divergence"] = div
    if div:
        out["signals"].append(SIG_DIV if div["type"] == "regulier" else SIG_DIV_HID)

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

    def is_core(h):
        return bool(CORE.intersection(h["signals"]))

    hits.sort(key=lambda h: (
        0 if is_core(h) else 1,
        -len(CORE.intersection(h["signals"])),          # combinaties bovenaan
        h["rsi"] if SIG_RSI in h["signals"] else 100,
        abs(h.get("dist_pct") if h.get("dist_pct") is not None else 99),
    ))
    core = [h for h in hits if is_core(h)]
    intra = [h for h in hits if not is_core(h)]

    result = {
        "scan_date_utc": now.isoformat(timespec="seconds"),
        "criteria": {
            "ma": f"slotkoers binnen ±{MA_TOUCH_PCT:.0%} van de 200-weeks SMA",
            "ma_intraweek": "week-range (high/low) raakte de 200-weeks SMA, slot verder weg",
            "rsi": f"RSI({RSI_PERIOD}) weekly < {RSI_THRESHOLD:.0f}",
            "divergentie": "koers lagere bodem + RSI hogere bodem (regulier), of "
                           "koers hogere bodem + RSI lagere bodem in stijgende trend (verborgen)",
        },
        "universe_size": universe_size,
        "tickers_scanned": scanned,
        "signal_count": len(core),
        "intraweek_count": len(intra),
        "divergence_count": sum(1 for h in core if h.get("divergence")),
        "warnings": warnings,
        "signals": core,
        "signals_intraweek": intra,
    }
    with open(os.path.join(DATA_DIR, "signals.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(DATA_DIR, "history", f"{stamp}.json"), "w") as f:
        json.dump(result, f, indent=1)

    # REPORT.md
    def div_cell(h):
        d = h.get("divergence")
        if not d:
            return "–"
        pijl = "↗" if d["type"] == "regulier" else "↘"
        return (f"{d['type']} {pijl} RSI {d['rsi1']}→{d['rsi2']} "
                f"({d['date1']} → {d['date2']})")

    def table(rows):
        lines = [
            "| Ticker | Bedrijf | Index | Signaal | Slot | RSI (w) | 200WMA | Afstand | Divergentie |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
        for h in rows:
            ma = f"{h['sma200w']}" if h.get("sma200w") else "–"
            dist = f"{h['dist_pct']:+.1f}%" if h.get("dist_pct") is not None else "–"
            lines.append(
                f"| **{h['ticker']}** | {h['name']} | {h['index']} | {' + '.join(h['signals'])} "
                f"| {h['close']} | {h['rsi']} | {ma} | {dist} | {div_cell(h)} |"
            )
        return "\n".join(lines)

    md = [
        f"# Wekelijkse marktscan — {stamp}",
        "",
        f"*Universum: S&P 500 + Nasdaq-100 ({universe_size} tickers, {scanned} gescand) · weekchart · "
        f"criteria: slotkoers binnen ±{MA_TOUCH_PCT:.0%} van de 200-weeks SMA, "
        f"RSI({RSI_PERIOD}) < {RSI_THRESHOLD:.0f} en/of een bullish divergentie.*",
        "",
    ]
    if warnings:
        md += ["> ⚠️ " + " | ".join(warnings), ""]
    if core:
        n_div = sum(1 for h in core if h.get("divergence"))
        md += [f"## {len(core)} signalen", "", table(core), ""]
        md += [
            f"*Divergentie: bij {n_div} van deze {len(core)} aandelen. "
            "**regulier ↗** = koers zette een lagere bodem terwijl de RSI een hógere bodem maakte "
            "(verkoopdruk neemt af, klassiek omkeersignaal). "
            "**verborgen ↘** = koers zette een hógere bodem terwijl de RSI lager ging, in een "
            "stijgende trend (meestal voortzetting van die trend). "
            "De datums zijn de twee weken waarin die bodems lagen.*",
            "",
        ]
    else:
        md += ["## Geen signalen deze week", "",
               "Geen enkel aandeel voldeed aan de hoofdcriteria.", ""]
    if intra:
        md += [
            f"## Ter info: {len(intra)} intraweek aangeraakt",
            "",
            "Deze aandelen raakten de 200-weeks MA tijdens de week wel aan (high/low), "
            "maar sloten er verder dan 2% vandaan.",
            "",
            table(intra),
            "",
        ]
    md += ["_Databron: Yahoo Finance (koersen split-adjusted). Geen beleggingsadvies._", ""]
    with open("REPORT.md", "w") as f:
        f.write("\n".join(md))

    # alert-bestanden (alleen bij signalen) -> GitHub-issue + e-mail
    alert_md = os.path.join(DATA_DIR, "alert.md")
    alert_title = os.path.join(DATA_DIR, "alert_title.txt")
    for p in (alert_md, alert_title):
        if os.path.exists(p):
            os.remove(p)
    if core:
        both = [h for h in core if len(CORE.intersection(h["signals"])) > 1]
        n_div = sum(1 for h in core if h.get("divergence"))
        title = f"📉 Marktscan {stamp}: {len(core)} signalen" + (f" (waarvan {len(both)} dubbel)" if both else "")
        body = [
            f"De wekelijkse scan van {stamp} vond **{len(core)} aandelen** die aan de criteria voldoen"
            + (f", waarvan **{n_div}** met een bullish divergentie" if n_div else "") + ".",
            "",
            table(core[:50]),
        ]
        if len(core) > 50:
            body.append(f"\n… en nog {len(core) - 50} meer — zie REPORT.md in de repository.")
        if intra:
            body.append(f"\nDaarnaast raakten **{len(intra)}** aandelen de 200-weeks MA alleen intraweek aan; "
                        "die staan in REPORT.md.")
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
    core = sum(1 for h in hits if CORE.intersection(h["signals"]))
    div = sum(1 for h in hits if h.get("divergence"))
    print(f"Klaar: {core} signalen (waarvan {div} met divergentie, "
          f"{len(hits) - core} alleen intraweek). Zie REPORT.md en data/signals.json")


if __name__ == "__main__":
    main()
