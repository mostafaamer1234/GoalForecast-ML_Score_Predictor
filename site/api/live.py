"""GET /api/live -> current Premier League state and model odds for the upcoming fixtures.

Starts from the Elo and last-5 form state after the bundled seasons (_model.json, written by
backtest/export_site_data.py), replays every later Premier League result from football-data.co.uk
(free for non-commercial use) with the same rules as backtest.build_features, and prices the
published upcoming fixtures with the same logistic regression. Standard library only, so it
starts fast.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import urllib.request
from collections import deque
from datetime import date, datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "_model.json")) as f:
    MODEL = json.load(f)
BASE = "https://www.football-data.co.uk"
PRIOR_GOALS, PRIOR_PTS = 1.4, 1.3  # backtest.build_features defaults for a team with no history


def get_csv(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; goalforecast-demo)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode("utf-8-sig", errors="replace")
    return [row for row in csv.DictReader(io.StringIO(text)) if any(row.values())]


def season_codes(today: date) -> list[str]:
    """Every season after the bundled one, up to the current one ('2526', '2627', ...)."""
    last = int(MODEL["state"]["last_season"][:2])
    current = today.year % 100 if today.month >= 7 else today.year % 100 - 1
    return [f"{y:02d}{y + 1:02d}" for y in range(last + 1, current + 1)]


def parse_date(s: str) -> date:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(s)


class State:
    def __init__(self) -> None:
        st, w = MODEL["state"], MODEL["window"]
        self.elo = dict(st["ratings"])
        self.gf = {t: deque(v, maxlen=w) for t, v in st["gf"].items()}
        self.ga = {t: deque(v, maxlen=w) for t, v in st["ga"].items()}
        self.pts = {t: deque(v, maxlen=w) for t, v in st["pts"].items()}
        self.w = w
        self.k, self.adv = 20.0, MODEL["home_advantage"]

    def rating(self, t: str) -> float:
        return self.elo.get(t, 1500.0)

    def q(self, store: dict, t: str) -> deque:
        return store.setdefault(t, deque(maxlen=self.w))

    def update(self, h: str, a: str, hg: int, ag: int, res: str) -> None:
        exp = 1 / (1 + 10 ** ((self.rating(a) - self.rating(h) - self.adv) / 400))
        delta = self.k * ({"H": 1.0, "D": 0.5, "A": 0.0}[res] - exp)
        self.elo[h], self.elo[a] = self.rating(h) + delta, self.rating(a) - delta
        self.q(self.gf, h).append(hg); self.q(self.ga, h).append(ag); self.q(self.gf, a).append(ag); self.q(self.ga, a).append(hg)
        self.q(self.pts, h).append({"H": 3, "D": 1, "A": 0}[res]); self.q(self.pts, a).append({"H": 0, "D": 1, "A": 3}[res])

    def form(self, t: str) -> dict:
        mean = lambda d, prior: sum(d) / len(d) if d else prior  # noqa: E731
        return {"elo": round(self.rating(t), 1), "gf5": mean(self.gf.get(t, ()), PRIOR_GOALS), "ga5": mean(self.ga.get(t, ()), PRIOR_GOALS),
                "pts5": mean(self.pts.get(t, ()), PRIOR_PTS)}


def predict(home: dict, away: dict) -> dict:
    v = {"elo_diff": home["elo"] + MODEL["home_advantage"] - away["elo"], "home_elo": home["elo"], "away_elo": away["elo"],
         "home_gf5": home["gf5"], "home_ga5": home["ga5"], "away_gf5": away["gf5"], "away_ga5": away["ga5"],
         "home_pts5": home["pts5"], "away_pts5": away["pts5"]}
    sc = MODEL["scaler"]
    x = [(v[f] - m) / s for f, m, s in zip(MODEL["features"], sc["mean"], sc["scale"])]
    z = [b + sum(w * xi for w, xi in zip(ws, x)) for ws, b in zip(MODEL["coef"], MODEL["intercept"])]
    top = max(z)
    e = [math.exp(zi - top) for zi in z]
    return {c: round(ei / sum(e), 4) for c, ei in zip(MODEL["class_order"], e)}


def implied(row: dict) -> dict | None:
    for pre in ("Avg", "B365", "Max"):
        try:
            odds = [float(row[f"{pre}{c}"]) for c in "HDA"]
        except (KeyError, ValueError, TypeError):
            continue
        inv = [1 / o for o in odds]
        return {c: round(i / sum(inv), 4) for c, i in zip("HDA", inv)}
    return None


def build() -> dict:
    today = datetime.now(timezone.utc).date()
    st = State()
    codes = season_codes(today)
    current_rows: list[dict] = []
    for code in codes:
        rows = [r for r in get_csv(f"{BASE}/mmz4281/{code}/E0.csv") if r.get("FTR") in ("H", "D", "A")]
        rows.sort(key=lambda r: (parse_date(r["Date"]), r["HomeTeam"]))
        for r in rows:
            st.update(r["HomeTeam"], r["AwayTeam"], int(r["FTHG"]), int(r["FTAG"]), r["FTR"])
        if code == codes[-1]:
            current_rows = rows
    table: dict[str, dict] = {}
    for r in current_rows:
        for team, gf, ga, res in ((r["HomeTeam"], int(r["FTHG"]), int(r["FTAG"]), {"H": "W", "D": "D", "A": "L"}[r["FTR"]]),
                                  (r["AwayTeam"], int(r["FTAG"]), int(r["FTHG"]), {"H": "L", "D": "D", "A": "W"}[r["FTR"]])):
            t = table.setdefault(team, {"team": team, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0, "last": []})
            t["p"] += 1; t["gf"] += gf; t["ga"] += ga
            t[{"W": "w", "D": "d", "L": "l"}[res]] += 1
            t["pts"] += {"W": 3, "D": 1, "L": 0}[res]
            t["last"] = (t["last"] + [res])[-5:]
    standings = sorted(table.values(), key=lambda t: (-t["pts"], -(t["gf"] - t["ga"]), -t["gf"], t["team"]))
    fixtures = []
    for r in get_csv(f"{BASE}/fixtures.csv"):
        if r.get("Div") != "E0":
            continue
        h, a = r["HomeTeam"], r["AwayTeam"]
        fixtures.append({"date": parse_date(r["Date"]).isoformat(), "time": r.get("Time", ""), "home": h, "away": a,
                         "model": predict(st.form(h), st.form(a)), "market": implied(r)})
    teams = sorted(set(t["team"] for t in standings) or st.elo)
    return {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "season": f"20{codes[-1][:2]}/{codes[-1][2:]}" if codes else None,
        "played": len(current_rows),
        "last_result": max((parse_date(r["Date"]) for r in current_rows), default=None).isoformat() if current_rows else None,
        "teams": {t: st.form(t) for t in teams},
        "table": standings,
        "fixtures": fixtures,
    }


def app(environ, start_response):
    try:
        body, status = build(), "200 OK"
        cache = "public, s-maxage=1800, stale-while-revalidate=86400"
    except Exception as e:  # upstream down or format change: the page falls back to the bundled season
        body, status, cache = {"error": f"Live data unavailable ({type(e).__name__})."}, "502 Bad Gateway", "no-store"
    raw = json.dumps(body).encode()
    start_response(status, [("Content-Type", "application/json"), ("Content-Length", str(len(raw))), ("Cache-Control", cache)])
    return [raw]
