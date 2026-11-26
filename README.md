# GoalForecast - ML Football/Soccer score Predictor

**Live demo: [goalforecast.vercel.app](https://goalforecast.vercel.app)**. Pick any 2024/25 Premier League fixture and get home/draw/away probabilities from the backtested model, computed in the browser.

GoalForecast is an full-stack machine learning football and soccer score prediction platform that combines ensemble machine learning models with a FastAPI backend and a React/Vite frontend to deliver data-driven predictions for Premier League, La Liga, and NFL games. It uses an ensemble of machine learning models to predict scorelines and win/draw/loss probabilities.

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

