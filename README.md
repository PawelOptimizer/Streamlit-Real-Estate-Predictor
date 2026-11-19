# Real-estate predictor (Streamlit app bundle)

This folder is a **standalone, app-only** bundle you can push to a separate repo.
It contains:
- `streamlit_app/app.py` (Streamlit UI)
- `src/` minimal helper modules used by the app (incl. `llm_explain.py` for the LLM explanation)
- `artifacts/` models + metadata + example rows

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Optional (for the LLM explanation): add your OpenAI key to `.env` in the repo root:

```bash
echo 'OPENAI_API_KEY="your_api_key_here"' > .env
```

You can also paste the key in the app, or set it as an environment variable / Streamlit secret.

## Run

```bash
source .venv/bin/activate
streamlit run streamlit_app/app.py
```

## Use examples in the app

1. In the sidebar, keep:
   - `Months window = 6`
   - `Model kind = combined` (tabular + ad text) or `tabular` (no text)
2. In `Input`, select **Example mode**.
3. Pick `Example #...` from **Load example (from temporal test set)**.
4. Click **Predict for selected example**.
5. Review:
   - predicted `price_per_m`
   - lower/upper interval
   - comparison table in **Example predictions**
6. In **Explain prediction (XAI)**, review the what-if table and click **Explain with LLM**.

Tip: if you switch to **URL mode**, fetched ad fields can be edited before prediction.

## Model

- `HistGradientBoostingRegressor` trained on the last 6 months of the dataset (Aug 2020 - Jan 2021, one city).
- `combined` adds the ad title + description (TF-IDF reduced with SVD).
- Temporal split: train on older ads, test on the most recent 20%. Test MAPE is about 4.1%
  (naive district-median baseline: about 8.0%); exact numbers are in the `*.meta.json` files.
- The data ends in January 2021, so predictions for current ads reflect prices from that period.

## Artifacts

The app expects these files in `artifacts/` (already included here, for `combined` and `tabular`):
- `model_recent_6m_<kind>.joblib`
- `model_recent_6m_<kind>.meta.json`
- `interval_recent_6m_<kind>.json`
- `examples_recent_6m_<kind>.test.json`

The models were saved with `scikit-learn==1.9.1` (pinned in `requirements.txt`).
