# Real-estate predictor (Streamlit app bundle)

This folder is a **standalone, app-only** bundle you can push to a separate repo.
It contains:
- `streamlit_app/app.py` (Streamlit UI)
- `src/` minimal helper modules used by the app
- `artifacts/` model + metadata + example rows

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Build demo model artifacts (recommended)

This repo includes metadata + example rows, but the actual `joblib` model file may be missing.
To generate a small **demo** model locally (so the app starts), run:

```bash
source .venv/bin/activate
python scripts/build_demo_artifacts.py
```

## Run

```bash
source .venv/bin/activate
streamlit run streamlit_app/app.py
```

## Use examples in the app

1. In the sidebar, keep:
   - `Months window = 6`
   - `Model kind = combined`
2. In `Input`, select **Example mode**.
3. Pick `Example #...` from **Load example (from temporal test set)**.
4. Click **Predict for selected example**.
5. Review:
   - predicted `price_per_m`
   - lower/upper interval
   - comparison table in **Example predictions**

Tip: if you switch to **URL mode**, fetched ad fields can be edited before prediction.

## Artifacts

The app expects these files in `artifacts/` (already included here):
- `model_recent_6m_combined.joblib`
- `model_recent_6m_combined.meta.json`
- `interval_recent_6m_combined.json`
- `examples_recent_6m_combined.test.json`

Optional diagnostics:
- `backtest_trend_vs_baseline.csv`
- `trend_diagnostic_recent6m.png`

