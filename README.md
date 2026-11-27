# GoalForecast: ML Football/Soccer Score Predictor

![CI](https://github.com/mostafaamer1234/GoalForecast-ML_Score_Predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![React](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

GoalForecast is a full-stack machine learning football and soccer score prediction platform that combines ensemble machine learning models with a FastAPI backend and a React/Vite frontend to deliver data-driven predictions for Premier League, La Liga, and NFL games. It uses an ensemble of machine learning models to predict scorelines and win/draw/loss probabilities.

**Try it: [goalforecast.vercel.app](https://goalforecast.vercel.app)**. Model odds for this round's real Premier League fixtures next to the bookmakers', any custom matchup, and the live table with Elo and form. `site/api/live.py` replays the current season from football-data.co.uk on top of the state exported by `python -m backtest.export_site_data`. The backtest is at [/about](https://goalforecast.vercel.app/about).


## Model evaluation on real matches

`backtest/` runs a **walk-forward backtest on six real Premier League seasons** (2019/20–2024/25, data from football-data.co.uk). For each test season, models train only on earlier seasons, and every feature is computed only from matches played before kickoff:
- Elo ratings with home advantage.
- Last-5 goals for and against, and points.

Tests in `backtest/test_backtest.py` check for look-ahead leakage.

Results over the 2022/23, 2023/24 and 2024/25 seasons (1,140 matches; outcome = home win / draw / away win):

| Model | Log loss (lower is better) | Brier (lower is better) | Accuracy |
|---|---|---|---|
| Base rates (H/D/A frequency) | 1.0639 | 0.6437 | 45.1% |
| Logistic regression (Elo + form) | 0.9720 | 0.5786 | 54.4% |
| XGBoost (Elo + form) | 0.9934 | 0.5890 | 54.4% |
| Bookmaker odds (Bet365, de-margined) | 0.9487 | 0.5618 | 56.5% |

What this shows:
- The learned models clearly beat the base-rate baseline: log loss drops by about 0.09 and accuracy rises by more than 9 points.
- They land within about 2 accuracy points of bookmaker prices, which is the strongest public baseline.
- With these features the simpler, better-calibrated logistic model beats XGBoost. That makes it the right default.

```bash
pip install -r backend/requirements.txt pytest
pytest -q backtest            # leakage and metric tests
python backtest/backtest.py   # reproduce the table
```

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Machine Learning Overview](#machine-learning-overview)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [API](#api)
- [Development Notes](#development-notes)
- [Technology Stack](#technology-stack)
- [License](#license)

---

## Overview

FootyForecast follows a simple full-stack architecture:

- React/Vite frontend (port 5173) for the user interface
- FastAPI backend (port 8000) exposing prediction endpoints
- Machine learning models (Random Forest + XGBoost) serialized as pickle files

The typical workflow:

1. Frontend fetches upcoming fixtures and sends prediction requests.
2. Backend enriches teams with strength/form data and constructs feature vectors.
3. ML ensemble model made up multiple models predicts home/away scores.
4. Scores are converted into win/draw/loss probabilities.
5. Frontend displays probabilities, suggested outcome, and projected scoreline.

---

## Features

- Ensemble ML models (Random Forest + XGBoost) for robust score predictions.
- Win/draw/loss probabilities derived from predicted scorelines.
- Team strength profiles and recent form features (last 5 matches, home advantage).
- Support for multiple competitions: Premier League, La Liga, and NFL.
- REST API with single and batch prediction endpoints.
- React frontend with dedicated pages for EPL, La Liga, and NFL predictions.
- Works out of the box with synthetic data; can optionally train on real historical data via API-Football.

---


## Project Structure

```text
GoalForecast/
├── backend/                    # FastAPI backend
│   ├── main.py                # FastAPI app and routes
│   ├── prediction_agent.py    # ML prediction agent and probability shaping
│   ├── model_trainer.py       # Model training and evaluation
│   ├── feature_engineering.py # Feature extraction
│   ├── data_generator.py      # Synthetic data generation
│   ├── historical_data_fetcher.py  # Real match data from API-Football
│   ├── train_from_csv.py      # Training from CSV
│   ├── train_with_real_data.py # Training on real API data
│   ├── interactive_predictor.py # CLI predictions
│   ├── test_api.py            # API tests
│   ├── test_api_connection.py # Connection checks
│   ├── team_logos.py          # Team logo mappings
│   └── requirements.txt       # Python dependencies
│
├── src/                       # React frontend
│   ├── main.jsx               # React entry
│   ├── App.jsx                # Main app component
│   ├── lib/
│   │   └── api.js             # API client utilities
│   ├── components/
│   │   ├── Navbar.jsx
│   │   └── Footer.jsx
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── PremierLeague.jsx
│   │   ├── LaLiga.jsx
│   │   ├── NFL.jsx
│   │   └── Pages.css
│   ├── assets/                # Images, badges, etc.
│   └── index.css
│
├── *.pkl                      # Trained model files
├── package.json
├── vite.config.js
├── index.html
└── README.md
```

Key files:

* `backend/main.py`: FastAPI application and API endpoints.
* `backend/prediction_agent.py`: Prediction logic and probability shaping.
* `backend/model_trainer.py`: Model training and evaluation.
* `src/pages/PremierLeague.jsx`: Main EPL prediction UI.
* `src/lib/api.js`: Frontend API client.

---

## Machine Learning Overview

GoalForecast uses an ensemble of tree-based regressors to predict home and away goals:

* Random Forest Regressor (multi-output)
* XGBoost Regressor (multi-output)
* Final predictions are the average of both models.

Core feature groups:

* Team statistics (e.g., summary stats of player ratings).
* Recent form (last 5 matches: points, win/draw/loss counts, momentum).
* Comparative features (rating difference, form difference, strength ratio).
* Home advantage indicator.

Predicted scores are transformed into win/draw/loss probabilities using a softmax-style transformation with temperature scaling and a neutral prior. The model is typically more reliable at predicting the match outcome than the exact scoreline.

The system can train on:

* Synthetic data (default, Poisson-based goal distributions).
* Real historical data from API-Football (requires API key).
* Custom CSV datasets.

---

## Installation

### Prerequisites

* Python 3.10+
* Node.js 18+
* npm or yarn
* (Optional) API-Football key for real data

### Backend Setup

```bash
git clone https://github.com/mostafaamer1234/GoalForecast.git
cd GoalForecast

cd backend
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(Optional) Add an API-Football key:

```bash
echo "API_KEY=your_api_football_key" > .env
```

Start the backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

From the project root:

```bash
npm install
npm run dev
```

By default, the frontend expects the backend at `http://localhost:8000`. Update `src/lib/api.js` if your backend is hosted elsewhere.

---

## How to Run

### Development

1. Start the FastAPI backend:

   ```bash
   cd backend
   source .venv/bin/activate
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Start the React frontend in a new terminal:

   ```bash
   npm run dev
   ```

3. Open `http://localhost:5173` in your browser and navigate to the league pages (e.g., Premier League) to view predictions.

### Quick Start (Synthetic Data)

If no trained model is found, the backend can auto-train on synthetic data. In that case:

1. Start the backend.
2. Start the frontend.
3. Visit the Premier League page; predictions will use built-in team strength profiles and synthetic training data.

---

## API

Base URL (local development):

```text
http://localhost:8000
```

### `GET /`

Health check.

**Response:**

```json
{ "message": "FootyForecast API running (EPL, LaLiga, ML Predictions Enabled)" }
```

### `POST /predict`

Predict a single match outcome.

**Request:**

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea"
}
```

**Response (example):**

```json
{
  "probs": {
    "home_win": 0.42,
    "draw": 0.28,
    "away_win": 0.30
  },
  "suggested": "home",
  "home_score": 2,
  "away_score": 1
}
```

### `POST /batch_predict`

Predict multiple matches in one call.

**Request:**

```json
[
  { "home_team": "Arsenal", "away_team": "Chelsea" },
  { "home_team": "Liverpool", "away_team": "Manchester City" }
]
```

**Response:**

Array of prediction objects in the same format as `/predict`.

Additional endpoints (if enabled in `main.py`) include league fixtures and tables:

* `GET /fixtures`
* `GET /table`
* `GET /laliga_fixtures`
* `GET /laliga_table`

---

## Development Notes

### Testing

Backend:

```bash
cd backend
python test_api_connection.py   # Check API connectivity
python test_api.py              # Test endpoints
```

Frontend:

```bash
npm run lint
```

### Training with Real Data

1. Obtain an API-Football key.
2. Set `API_KEY` in `backend/.env`.
3. Run:

   ```bash
   cd backend
   python train_with_real_data.py --leagues 39 140 --season 2023 --max-matches 200
   ```

You can also train from CSV files with `train_from_csv.py` or generate synthetic data with `data_generator.py`.

### Deployment (High Level)

* Expose the FastAPI backend (directly or behind a reverse proxy such as nginx).
* Build the frontend (`npm run build`) and serve the static `dist/` folder.
* Keep `.env` and API keys out of version control.
* Configure CORS and HTTPS appropriately for production.

---

## Technology Stack

### Backend

- Python 3.10+
- FastAPI
- scikit-learn
- XGBoost
- pandas, numpy
- joblib
- uvicorn
- python-dotenv
- requests

### Frontend

- React 18
- Vite
- React Router
- CSS3
- (Optional) Framer Motion for animations

---


## License

MIT License

Copyright (c) 2025

