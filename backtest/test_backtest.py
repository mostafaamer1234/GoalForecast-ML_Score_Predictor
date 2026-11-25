from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.backtest import EloState, build_features, load_matches, multiclass_brier, run


def _toy_matches() -> pd.DataFrame:
    rows = [
        ("2425", "2024-08-10", "A", "B", 3, 0, "H"),
        ("2425", "2024-08-17", "B", "A", 0, 2, "A"),
        ("2425", "2024-08-24", "A", "B", 1, 1, "D"),
    ]
    df = pd.DataFrame(rows, columns=["season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_features_use_only_earlier_matches():
    feats = build_features(_toy_matches())
    first = feats.iloc[0]
    assert first.home_elo == first.away_elo == 1500  # nothing played yet
    assert first.home_gf5 == 1.4  # prior, not the 3 goals scored in this very match
    second = feats.iloc[1]
    assert second.away_gf5 == 3.0 and second.home_ga5 == 3.0  # only match 1 is visible
    assert second.away_elo > 1500 > second.home_elo


def test_elo_is_zero_sum():
    elo = EloState()
    elo.update("X", "Y", "H")
    assert np.isclose(elo.ratings["X"] + elo.ratings["Y"], 3000)


def test_brier_is_zero_for_perfect_predictions():
    y = np.array([0, 1, 2])
    assert multiclass_brier(y, np.eye(3)) == 0.0


def test_real_data_loads_six_seasons():
    df = load_matches()
    assert df["season"].nunique() == 6 and len(df) == 6 * 380


def test_walk_forward_never_trains_on_test_season():
    results = run(["2425"])
    assert set(results["season"]) == {"2425"} and (results["matches"] == 380).all()
    by_model = results.set_index("model")
    # a real model should beat always predicting the historical base rates
    assert by_model.loc["Logistic regression (Elo + form)", "log_loss"] < by_model.loc["Base rates (H/D/A frequency)", "log_loss"]
