"""Export the backtested model and end-of-season team state for the static demo site.

Writes site/data.json with:
  * logistic-regression parameters (scaler mean/scale, coefficients, intercepts) trained on all six seasons
  * each 2024/25 team's state after its last match (Elo, last-5 goals for/against, last-5 points)
  * walk-forward metrics for 2022/23-2024/25 and a calibration curve for the 2024/25 test season

The browser then computes home/draw/away probabilities for any pairing with the exact same model.

    python backtest/export_site_data.py
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest.backtest import CLASSES, FEATURES, EloState, build_features, load_matches, run

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data.json"


def final_states(matches, window: int = 5) -> dict[str, dict[str, float]]:
    """Replay every match in order and return each team's state after its last one."""
    elo = EloState()
    gf, ga, pts = (defaultdict(lambda: deque(maxlen=window)) for _ in range(3))
    for m in matches.itertuples(index=False):
        h, a = m.HomeTeam, m.AwayTeam
        elo.update(h, a, m.FTR)
        gf[h].append(m.FTHG); ga[h].append(m.FTAG); gf[a].append(m.FTAG); ga[a].append(m.FTHG)
        pts[h].append({"H": 3, "D": 1, "A": 0}[m.FTR]); pts[a].append({"H": 0, "D": 1, "A": 3}[m.FTR])
    last = matches[matches["season"] == matches["season"].max()]
    teams = sorted(set(last["HomeTeam"]) | set(last["AwayTeam"]))
    return {
        t: {"elo": round(elo.ratings[t], 2), "gf5": float(np.mean(gf[t])), "ga5": float(np.mean(ga[t])), "pts5": float(np.mean(pts[t]))}
        for t in teams
    }


def calibration(df, season: str = "2425", bins: int = 6) -> list[dict]:
    train, test = df[df["season"] < season], df[df["season"] == season]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[FEATURES], train["y"])
    p_home = model.predict_proba(test[FEATURES])[:, 0]
    hit = (test["y"].to_numpy() == 0).astype(float)
    edges = np.quantile(p_home, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_home >= lo) & (p_home <= hi)
        if mask.sum():
            out.append({"predicted": float(p_home[mask].mean()), "actual": float(hit[mask].mean()), "n": int(mask.sum())})
    return out


def main() -> None:
    matches = load_matches()
    df = build_features(matches).dropna(subset=["B365H", "B365D", "B365A"]).reset_index(drop=True)
    df["y"] = df["FTR"].map({c: i for i, c in enumerate(CLASSES)})

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(df[FEATURES], df["y"])
    scaler, lr = model[0], model[1]

    results = run(["2223", "2324", "2425"])
    metrics = {}
    for name, g in results.groupby("model", sort=False):
        w = g["matches"]
        metrics[name] = {k: float(np.average(g[k], weights=w)) for k in ("log_loss", "brier", "accuracy")}

    payload = {
        "features": FEATURES,
        "classes": CLASSES,
        "home_advantage": EloState().home_adv,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "coef": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "class_order": [CLASSES[int(c)] for c in lr.classes_],
        "teams": final_states(matches),
        "metrics": metrics,
        "test_matches": int(results.drop_duplicates("season")["matches"].sum()),
        "calibration": calibration(df),
        "seasons": "2019/20-2024/25",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} ({len(payload['teams'])} teams)")


if __name__ == "__main__":
    main()
