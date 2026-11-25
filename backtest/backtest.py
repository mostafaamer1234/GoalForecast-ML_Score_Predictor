"""Walk-forward backtest of match-outcome probabilities on real Premier League data.

For every test season, models are trained only on earlier seasons, and every
feature for a match is computed only from matches played before it (no look-ahead).
Probabilities for home win / draw / away win are scored with log loss, a
multiclass Brier score, and accuracy, against two baselines:

  * base rates   - the historical frequency of H/D/A in the training seasons
  * bookmakers   - Bet365 closing odds converted to probabilities (overround removed)

Data: football-data.co.uk (free for non-commercial use), seasons 2019/20-2024/25.

    python backtest/backtest.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA = Path(__file__).resolve().parent / "data"
CLASSES = ["H", "D", "A"]
FEATURES = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_gf5",
    "home_ga5",
    "away_gf5",
    "away_ga5",
    "home_pts5",
    "away_pts5",
]


def load_matches(data_dir: Path = DATA) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("E0_*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["season"] = path.stem.split("_")[1]
        frames.append(df[["season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["FTR"])
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True)
    return out.sort_values(["Date", "HomeTeam"], kind="stable").reset_index(drop=True)


@dataclass
class EloState:
    k: float = 20.0
    home_adv: float = 60.0
    start: float = 1500.0

    def __post_init__(self) -> None:
        self.ratings: dict[str, float] = defaultdict(lambda: self.start)

    def expected_home(self, home: str, away: str) -> float:
        return 1.0 / (1.0 + 10 ** ((self.ratings[away] - self.ratings[home] - self.home_adv) / 400))

    def update(self, home: str, away: str, result: str) -> None:
        score = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
        delta = self.k * (score - self.expected_home(home, away))
        self.ratings[home] += delta
        self.ratings[away] -= delta


def build_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute pre-match features in chronological order, then update state with the result."""
    elo = EloState()
    gf: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    ga: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def mean(q: deque, default: float) -> float:
        return float(np.mean(q)) if q else default

    rows = []
    for m in matches.itertuples(index=False):
        h, a = m.HomeTeam, m.AwayTeam
        rows.append(
            {
                "home_elo": elo.ratings[h],
                "away_elo": elo.ratings[a],
                "elo_diff": elo.ratings[h] + elo.home_adv - elo.ratings[a],
                "home_gf5": mean(gf[h], 1.4),
                "home_ga5": mean(ga[h], 1.4),
                "away_gf5": mean(gf[a], 1.4),
                "away_ga5": mean(ga[a], 1.4),
                "home_pts5": mean(pts[h], 1.3),
                "away_pts5": mean(pts[a], 1.3),
            }
        )
        # update state only after the features for this match were recorded
        elo.update(h, a, m.FTR)
        gf[h].append(m.FTHG)
        ga[h].append(m.FTAG)
        gf[a].append(m.FTAG)
        ga[a].append(m.FTHG)
        pts[h].append({"H": 3, "D": 1, "A": 0}[m.FTR])
        pts[a].append({"H": 0, "D": 1, "A": 3}[m.FTR])
    return pd.concat([matches.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def bookmaker_probs(df: pd.DataFrame) -> np.ndarray:
    implied = 1.0 / df[["B365H", "B365D", "B365A"]].to_numpy(dtype=float)
    return implied / implied.sum(axis=1, keepdims=True)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    onehot = np.eye(len(CLASSES))[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def score(name: str, y: np.ndarray, probs: np.ndarray) -> dict[str, float | str]:
    return {
        "model": name,
        "log_loss": log_loss(y, probs, labels=[0, 1, 2]),
        "brier": multiclass_brier(y, probs),
        "accuracy": accuracy_score(y, probs.argmax(axis=1)),
    }


def run(test_seasons: list[str], data_dir: Path = DATA) -> pd.DataFrame:
    df = build_features(load_matches(data_dir))
    df = df.dropna(subset=["B365H", "B365D", "B365A"]).reset_index(drop=True)
    df["y"] = df["FTR"].map({c: i for i, c in enumerate(CLASSES)})
    results = []
    for season in test_seasons:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        y_train, y_test = train["y"].to_numpy(), test["y"].to_numpy()

        base = np.bincount(y_train, minlength=3) / len(y_train)
        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        logreg.fit(train[FEATURES], y_train)
        xgb = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", eval_metric="mlogloss", random_state=42,
        )
        xgb.fit(train[FEATURES], y_train)

        for row in (
            score("Base rates (H/D/A frequency)", y_test, np.tile(base, (len(y_test), 1))),
            score("Logistic regression (Elo + form)", y_test, logreg.predict_proba(test[FEATURES])),
            score("XGBoost (Elo + form)", y_test, xgb.predict_proba(test[FEATURES])),
            score("Bookmaker odds (Bet365, de-margined)", y_test, bookmaker_probs(test)),
        ):
            results.append({"season": season, "matches": len(test), **row})
    return pd.DataFrame(results)


def to_markdown(results: pd.DataFrame) -> str:
    agg = results.groupby("model", sort=False).apply(
        lambda g: pd.Series(
            {
                "log_loss": np.average(g["log_loss"], weights=g["matches"]),
                "brier": np.average(g["brier"], weights=g["matches"]),
                "accuracy": np.average(g["accuracy"], weights=g["matches"]),
            }
        ),
        include_groups=False,
    )
    lines = ["| Model | Log loss (lower is better) | Brier (lower is better) | Accuracy |", "|---|---|---|---|"]
    for model, r in agg.iterrows():
        lines.append(f"| {model} | {r.log_loss:.4f} | {r.brier:.4f} | {r.accuracy:.1%} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", nargs="+", default=["2223", "2324", "2425"])
    args = ap.parse_args()
    results = run(args.test_seasons)
    n = int(results.drop_duplicates("season")["matches"].sum())
    print(f"Walk-forward test on {len(args.test_seasons)} Premier League seasons ({n} matches):\n")
    print(to_markdown(results))


if __name__ == "__main__":
    main()
