#!/usr/bin/env python3
"""
Wekelijkse cryptoscanner — top 200 munten op marktkapitalisatie
Signalen op de WEEKCHART (maandag t/m zondag, UTC):
  1) Slotkoers binnen ±2% van de 200-weeks SMA (of 100-weeks bij jonge munten)
  2) RSI(14, weekly, Wilder) < 30
  3) Bullish divergentie (regulier of verborgen)

Databronnen (allemaal gratis en zonder sleutel):
  universum  — Coinlore  (rangschikking op marktkapitalisatie)
  koersen    — Coinbase Exchange (dagcandles), Kraken als reserve

Output: identiek van vorm aan scanner.py, maar in data/crypto/ en CRYPTO_REPORT.md
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import scanner  # hergebruikt rsi_wilder, pivot_lows, detect_divergence

# ---------------------------- instellingen ----------------------------
TOP_N = 200
MA_PERIOD = 200          # weken
MA_PERIOD_SHORT = 100    # terugval voor munten met minder historie
MA_TOUCH_PCT = 0.02
RSI_PERIOD = 14
RSI_THRESHOLD = 30.0
MIN_BARS = 30            # minder weekbars dan dit -> munt overslaan
HISTORY_DAYS = 1500      # ~214 weken

DATA_DIR = os.path.join("data", "crypto")
REPORT = "CRYPTO_REPORT.md"
UNIVERSE_CACHE = os.path.join(DATA_DIR, "universe.json")

SIG_RSI = scanner.SIG_RSI
SIG_MA = scanner.SIG_MA
SIG_MA_INTRA = scanner.SIG_MA_INTRA
SIG_DIV = scanner.SIG_DIV
SIG_DIV_HID = scanner.SIG_DIV_HID
CORE = scanner.CORE

UA = {"User-Agent": "Mozilla/5.0 (compatible; crypto-scanner/1.0)"}

# Munten die geen technische analyse verdienen: stablecoins en
# wrapped/staked varianten van een andere munt (dubbeltellingen).
STABLE_EXACT = {
    "DAI", "FRAX", "GHO", "BUIDL", "BFUSD", "MIM", "EURC", "EUROC", "EURT",
    "EURS", "STEUR", "AEUR", "XSGD", "XAUT", "PAXG", "TETHER",
}
WRAPPED_EXACT = {
    "WBTC", "WETH", "WBETH", "WEETH", "STETH", "WSTETH", "CBETH", "RETH",
    "BNSOL", "BSOL", "JITOSOL", "MSOL", "WBNB", "WMATIC", "WAVAX", "WSOL",
    "SOLVBTC", "LBTC", "CBBTC", "TBTC", "BETH", "SUSDE", "SDAI", "USDC.E",
}
NAME_HINTS = ("stablecoin", "wrapped", "staked", "bridged", "tokenized", "pegged")


def is_tradeable(symbol: str, name: str) -> bool:
    """Stablecoins, valuta- en metaaltokens en wrapped/staked varianten eruit filteren:
    die volgen hun onderpand, dus technische analyse zegt er niets over."""
    s, n = symbol.upper().strip(), name.lower()
    if s in WRAPPED_EXACT or s in STABLE_EXACT:
        return False
    if "usd" in s.lower():          # USDT, USDC, PYUSD, CRVUSD, RLUSD, USD1, …
        return False
    if any(w in n for w in NAME_HINTS):
        return False
    return True


# ---------------------------- universum -------------------------------
def fetch_universe(top_n: int = TOP_N) -> tuple[list[dict], list[str]]:
    """Top-N munten op marktkapitalisatie, zonder stablecoins en wrapped varianten."""
    coins, warnings = [], []
    for start in range(0, top_n, 100):
        url = f"https://api.coinlore.net/api/tickers/?start={start}&limit=100"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=30)
                r.raise_for_status()
                data = r.json().get("data", [])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    warnings.append(f"universum vanaf rang {start}: {e}")
                    data = []
                else:
                    time.sleep(5)
        for c in data:
            sym, name = str(c.get("symbol", "")).strip(), str(c.get("name", "")).strip()
            if not sym or not is_tradeable(sym, name):
                continue
            coins.append({
                "symbol": sym.upper(),
                "name": name,
                "rank": int(c.get("rank") or 0),
                "market_cap": float(c.get("market_cap_usd") or 0),
            })
        time.sleep(1)

    if coins:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(UNIVERSE_CACHE, "w") as f:
            json.dump(coins, f, indent=1)
        return coins, warnings

    if os.path.exists(UNIVERSE_CACHE):
        with open(UNIVERSE_CACHE) as f:
            return json.load(f), warnings + ["universum uit cache geladen"]
    raise RuntimeError("Geen universum beschikbaar: " + "; ".join(warnings))


# ---------------------------- koersdata -------------------------------
def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Dagcandles -> weekcandles, maandag t/m zondag (zoals TradingView)."""
    if daily.empty:
        return daily
    weekly = daily.resample("W-SUN").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna(subset=["Close"])
    return weekly


def coinbase_daily(symbol: str) -> pd.DataFrame | None:
    """Dagcandles van Coinbase Exchange (max 300 per verzoek, dus gepagineerd)."""
    rows = []
    end = datetime.now(timezone.utc)
    for _ in range(6):                      # 6 x 300 dagen ≈ 1800 dagen
        start = end - timedelta(days=300)
        try:
            r = requests.get(
                f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles",
                params={
                    "granularity": 86400,
                    "start": start.isoformat(timespec="seconds"),
                    "end": end.isoformat(timespec="seconds"),
                },
                headers=UA, timeout=30,
            )
        except Exception:  # noqa: BLE001
            return None
        if r.status_code == 404:
            return None                     # product bestaat niet
        if r.status_code == 429:
            time.sleep(2)
            continue
        if not r.ok:
            break
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        end = start
        time.sleep(0.25)
        if len(rows) >= HISTORY_DAYS:
            break

    if len(rows) < MIN_BARS * 7:
        return None
    # Coinbase: [time, low, high, open, close, volume]
    df = pd.DataFrame(rows, columns=["time", "Low", "High", "Open", "Close", "Volume"])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["Open", "High", "Low", "Close"]].astype(float)


def kraken_weekly(symbol: str, pair_map: dict) -> pd.DataFrame | None:
    """Weekcandles van Kraken. Let op: Kraken's weken lopen donderdag t/m woensdag."""
    pair = pair_map.get(symbol.upper())
    if not pair:
        return None
    try:
        r = requests.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": pair, "interval": 10080}, headers=UA, timeout=30,
        )
        payload = r.json()
    except Exception:  # noqa: BLE001
        return None
    if payload.get("error") or not payload.get("result"):
        return None
    key = next((k for k in payload["result"] if k != "last"), None)
    rows = payload["result"].get(key) or []
    if len(rows) < MIN_BARS:
        return None
    df = pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close",
                                     "vwap", "volume", "count"])
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["Open", "High", "Low", "Close"]].astype(float)


def kraken_pairs() -> dict:
    """{basissymbool: kraken-paarnaam} voor alle USD-paren."""
    try:
        r = requests.get("https://api.kraken.com/0/public/AssetPairs",
                         headers=UA, timeout=30)
        result = r.json().get("result", {})
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for name, info in result.items():
        if info.get("quote") not in ("ZUSD", "USD"):
            continue
        ws = info.get("wsname", "")
        base = ws.split("/")[0] if "/" in ws else info.get("base", "")
        base = {"XBT": "BTC", "XDG": "DOGE"}.get(base, base)
        out.setdefault(base.upper(), name)
    return out


def load_prices(coins: list[dict]) -> tuple[dict, dict, list[str]]:
    """Weekcandles per munt + welke bron gebruikt is."""
    kp = kraken_pairs()
    if not kp:
        print("  ! Kraken-parenlijst niet beschikbaar", file=sys.stderr)

    frames, sources, missing = {}, {}, []
    for i, c in enumerate(coins, 1):
        sym = c["symbol"]
        df, src = None, None

        daily = coinbase_daily(sym)
        if daily is not None:
            weekly = _to_weekly(daily)
            if len(weekly) >= MIN_BARS:
                df, src = weekly, "coinbase"

        if df is None:
            wk = kraken_weekly(sym, kp)
            if wk is not None and len(wk) >= MIN_BARS:
                df, src = wk, "kraken"

        if df is None:
            missing.append(sym)
        else:
            # laatste (nog lopende) week weglaten als die niet compleet is
            frames[sym], sources[sym] = df, src

        if i % 25 == 0:
            print(f"  {i}/{len(coins)} munten verwerkt ({len(frames)} met data)")
    return frames, sources, missing


# ---------------------------- analyse ---------------------------------
def evaluate_coin(df: pd.DataFrame) -> dict | None:
    """Zelfde logica als de aandelenscanner, maar met een kortere MA voor jonge munten."""
    df = df.dropna(subset=["Close"])
    if len(df) < MIN_BARS:
        return None

    close = df["Close"]
    last = df.iloc[-1]
    price = float(last["Close"])
    out = {
        "close": round(price, 6 if price < 1 else 2),
        "week_end": str(df.index[-1].date()),
        "weeks_history": len(df),
        "signals": [],
    }

    rsi = scanner.rsi_wilder(close, RSI_PERIOD)
    rsi_now = float(rsi.iloc[-1])
    out["rsi"] = round(rsi_now, 1) if np.isfinite(rsi_now) else None
    if np.isfinite(rsi_now) and rsi_now < RSI_THRESHOLD:
        out["signals"].append(SIG_RSI)

    # MA: 200 weken indien mogelijk, anders 100 weken, anders geen
    ma = None
    period = None
    for p in (MA_PERIOD, MA_PERIOD_SHORT):
        if len(df) >= p:
            ma, period = float(close.rolling(p).mean().iloc[-1]), p
            break
    out["ma_period"] = period
    if ma is not None:
        dist = (price - ma) / ma
        out["ma"] = round(ma, 6 if ma < 1 else 2)
        out["dist_pct"] = round(dist * 100, 2)
        lo = float(last["Low"]) if pd.notna(last["Low"]) else price
        hi = float(last["High"]) if pd.notna(last["High"]) else price
        if abs(dist) <= MA_TOUCH_PCT:
            out["signals"].append(SIG_MA)
        elif lo <= ma <= hi:
            out["signals"].append(SIG_MA_INTRA)
    else:
        out["ma"] = None
        out["dist_pct"] = None

    div = scanner.detect_divergence(df, rsi, ma)
    out["divergence"] = div
    if div:
        out["signals"].append(SIG_DIV if div["type"] == "regulier" else SIG_DIV_HID)

    return out if out["signals"] else None


# ---------------------------- rapportage ------------------------------
def write_outputs(hits, universe_size, scanned, sources, missing, warnings):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d")
    os.makedirs(os.path.join(DATA_DIR, "history"), exist_ok=True)

    def is_core(h):
        return bool(CORE.intersection(h["signals"]))

    hits.sort(key=lambda h: (
        0 if is_core(h) else 1,
        -len(CORE.intersection(h["signals"])),
        h["rsi"] if SIG_RSI in h["signals"] and h["rsi"] is not None else 100,
        abs(h["dist_pct"] if h.get("dist_pct") is not None else 99),
    ))
    core = [h for h in hits if is_core(h)]
    intra = [h for h in hits if not is_core(h)]

    result = {
        "scan_date_utc": now.isoformat(timespec="seconds"),
        "market": "crypto",
        "criteria": {
            "ma": f"slotkoers binnen ±{MA_TOUCH_PCT:.0%} van de {MA_PERIOD}-weeks SMA "
                  f"({MA_PERIOD_SHORT}-weeks bij jonge munten)",
            "rsi": f"RSI({RSI_PERIOD}) weekly < {RSI_THRESHOLD:.0f}",
            "divergentie": "koers lagere bodem + RSI hogere bodem (regulier), of "
                           "koers hogere bodem + RSI lagere bodem in stijgende trend (verborgen)",
        },
        "universe_size": universe_size,
        "coins_scanned": scanned,
        "coins_without_data": missing,
        "sources": {"coinbase": sum(1 for s in sources.values() if s == "coinbase"),
                    "kraken": sum(1 for s in sources.values() if s == "kraken")},
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

    def div_cell(h):
        d = h.get("divergence")
        if not d:
            return "–"
        pijl = "↗" if d["type"] == "regulier" else "↘"
        return f"{d['type']} {pijl} RSI {d['rsi1']}→{d['rsi2']} ({d['date1']} → {d['date2']})"

    def table(rows):
        lines = [
            "| Munt | Naam | Rang | Signaal | Koers | RSI (w) | MA | Afstand | Divergentie |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
        for h in rows:
            ma = f"{h['ma']} ({h['ma_period']}w)" if h.get("ma") else "–"
            dist = f"{h['dist_pct']:+.1f}%" if h.get("dist_pct") is not None else "–"
            lines.append(
                f"| **{h['symbol']}** | {h['name']} | {h['rank']} | {' + '.join(h['signals'])} "
                f"| ${h['close']} | {h['rsi']} | {ma} | {dist} | {div_cell(h)} |"
            )
        return "\n".join(lines)

    md = [
        f"# Wekelijkse cryptoscan — {stamp}",
        "",
        f"*Top {TOP_N} op marktkapitalisatie → {universe_size} munten na het uitfilteren van "
        f"stablecoins en wrapped varianten, {scanned} met bruikbare koershistorie · "
        f"weekchart (ma t/m zo) · "
        f"criteria: slotkoers binnen ±{MA_TOUCH_PCT:.0%} van de {MA_PERIOD}-weeks SMA, "
        f"RSI({RSI_PERIOD}) < {RSI_THRESHOLD:.0f} en/of een bullish divergentie.*",
        "",
    ]
    if warnings:
        md += ["> ⚠️ " + " | ".join(warnings), ""]
    if core:
        n_div = sum(1 for h in core if h.get("divergence"))
        md += [f"## {len(core)} signalen", "", table(core), ""]
        md += [
            f"*Divergentie: bij {n_div} van deze {len(core)} munten. **regulier ↗** = koers "
            "lagere bodem, RSI hógere bodem (verkoopdruk neemt af). **verborgen ↘** = koers "
            "hógere bodem, RSI lagere bodem in een stijgende trend. Bij `(100w)` achter de MA "
            "is de munt te jong voor een 200-weeks gemiddelde.*",
            "",
        ]
    else:
        md += ["## Geen signalen deze week", "",
               "Geen enkele munt voldeed aan de hoofdcriteria.", ""]
    if intra:
        md += [f"## Ter info: {len(intra)} intraweek aangeraakt", "",
               "Deze munten raakten hun MA tijdens de week aan, maar sloten er verder dan "
               "2% vandaan.", "", table(intra), ""]
    if missing:
        md += [f"## {len(missing)} munt{'en' if len(missing) != 1 else ''} zonder koersdata", "",
               "Niet verhandeld op Coinbase of Kraken, dus niet te scannen: "
               + ", ".join(f"`{m}`" for m in missing) + ".", ""]
    md += ["_Databron: Coinbase Exchange en Kraken. Geen beleggingsadvies._", ""]
    with open(REPORT, "w") as f:
        f.write("\n".join(md))

    alert_md = os.path.join(DATA_DIR, "alert.md")
    alert_title = os.path.join(DATA_DIR, "alert_title.txt")
    for p in (alert_md, alert_title):
        if os.path.exists(p):
            os.remove(p)
    if core:
        both = [h for h in core if len(CORE.intersection(h["signals"])) > 1]
        n_div = sum(1 for h in core if h.get("divergence"))
        with open(alert_title, "w") as f:
            f.write(f"🪙 Cryptoscan {stamp}: {len(core)} signalen"
                    + (f" (waarvan {len(both)} dubbel)" if both else ""))
        body = [
            f"De wekelijkse cryptoscan van {stamp} vond **{len(core)} munten** die aan de "
            f"criteria voldoen" + (f", waarvan **{n_div}** met een bullish divergentie" if n_div else "") + ".",
            "", table(core[:50]),
        ]
        if len(core) > 50:
            body.append(f"\n… en nog {len(core) - 50} meer — zie {REPORT} in de repository.")
        body.append("\n_Geen beleggingsadvies._")
        with open(alert_md, "w") as f:
            f.write("\n".join(body))


# ---------------------------- main ------------------------------------
def main():
    coins, warnings = fetch_universe()
    print(f"Universum: {len(coins)} munten (na filteren van stablecoins en wrapped varianten)")

    frames, sources, missing = load_prices(coins)
    print(f"Koersdata voor {len(frames)} munten "
          f"(coinbase {sum(1 for s in sources.values() if s == 'coinbase')}, "
          f"kraken {sum(1 for s in sources.values() if s == 'kraken')})")
    if missing:
        print(f"Geen data voor {len(missing)}: {', '.join(missing[:20])}"
              + (" …" if len(missing) > 20 else ""))
    if len(frames) < len(coins) * 0.4:
        warnings.append(f"slechts {len(frames)}/{len(coins)} munten met koersdata")

    by_symbol = {c["symbol"]: c for c in coins}
    hits = []
    for sym, df in frames.items():
        try:
            res = evaluate_coin(df)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {sym}: {e}", file=sys.stderr)
            continue
        if res:
            res.update({
                "symbol": sym,
                "name": by_symbol[sym]["name"],
                "rank": by_symbol[sym]["rank"],
                "source": sources[sym],
            })
            hits.append(res)

    write_outputs(hits, len(coins), len(frames), sources, missing, warnings)
    core = sum(1 for h in hits if CORE.intersection(h["signals"]))
    div = sum(1 for h in hits if h.get("divergence"))
    print(f"Klaar: {core} signalen (waarvan {div} met divergentie, "
          f"{len(hits) - core} alleen intraweek). Zie {REPORT}")


if __name__ == "__main__":
    main()
