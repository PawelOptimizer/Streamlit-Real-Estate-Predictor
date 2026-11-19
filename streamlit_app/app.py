from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# project root = parent of streamlit_app/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import load_json, load_model  # noqa: E402
from src.llm_explain import DEFAULT_LLM_MODEL, explain_prediction, resolve_api_key  # noqa: E402
from src.otodom_extract import extract_otodom_features  # noqa: E402
from src.text_utils import strip_html_to_text  # noqa: E402


DISTRICT_LABELS = {
    (52.384, 16.941): "Rataje (Os. Piastowskie)",
    (52.378, 16.987): "Franowo / Chartowo",
    (52.399, 16.964): "Lacina (Rataje)",
    (52.380, 16.944): "Rataje",
    (52.388, 16.976): "Chartowo (Os. Lecha)",
    (52.380, 16.965): "Zegrze",
    (52.412, 16.922): "Stare Miasto",
    (52.408, 16.931): "Stare Miasto / Chwaliszewo",
}


def _default_artifact_paths(*, months: int, model_kind: str) -> dict[str, Path]:
    base = ROOT / "artifacts"
    stem = f"recent_{months}m_{model_kind}"
    return {
        "model": base / f"model_{stem}.joblib",
        "meta": base / f"model_{stem}.meta.json",
        "interval": base / f"interval_{stem}.json",
        "examples": base / f"examples_{stem}.test.json",
    }


def _ensure_all_columns(row: dict, columns: list[str]) -> pd.DataFrame:
    full = {c: row.get(c, None) for c in columns}
    return pd.DataFrame([full])


def _prepare_row_for_prediction(row_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    # Keep training/inference dtypes aligned (column groups come from the model metadata).
    prepared = row_df.copy()
    numeric_cols = meta.get("numeric_columns", [])
    string_cols = list(meta.get("multilabel_columns", [])) + ["__text__"]
    for c in prepared.columns:
        if c in numeric_cols:
            prepared[c] = pd.to_numeric(prepared[c], errors="coerce").astype(float)
        elif c in string_cols:
            prepared[c] = prepared[c].fillna("").astype(str)
        else:
            prepared[c] = prepared[c].where(prepared[c].notna(), np.nan)
    return prepared


def _predict(model, row_df: pd.DataFrame, meta: dict) -> float:
    return float(model.predict(_prepare_row_for_prediction(row_df, meta))[0])


def _what_if_effects(model, row_df: pd.DataFrame, meta: dict, coverage: pd.DataFrame) -> pd.DataFrame:
    """Simple local sensitivity analysis: change one input at a time and measure the prediction change."""
    base = _predict(model, row_df, meta)
    row = row_df.iloc[0]
    scenarios: list[tuple[str, dict]] = []
    m_val = pd.to_numeric(row.get("m"), errors="coerce")
    if pd.notna(m_val):
        scenarios += [("area -10 m²", {"m": max(10.0, m_val - 10)}), ("area +10 m²", {"m": m_val + 10})]
    rooms = pd.to_numeric(row.get("no_rooms"), errors="coerce")
    if pd.notna(rooms):
        if rooms > 1:
            scenarios.append(("1 room fewer", {"no_rooms": rooms - 1}))
        scenarios.append(("1 room more", {"no_rooms": rooms + 1}))
    if row.get("market") in ("primary", "secondary"):
        other = "secondary" if row.get("market") == "primary" else "primary"
        scenarios.append((f"market = {other}", {"market": other}))
    floor = pd.to_numeric(row.get("param_floor"), errors="coerce")
    if "param_floor" in row_df.columns and pd.notna(floor):
        scenarios.append(("ground floor", {"param_floor": 0.0}))
    if "param_build_year" in row_df.columns and pd.notna(pd.to_numeric(row.get("param_build_year"), errors="coerce")):
        scenarios.append(("build year unknown", {"param_build_year": np.nan}))
    if not coverage.empty:
        top = coverage.iloc[0]
        scenarios.append(
            (
                f"location = most common district ({top['district_hint']})",
                {"district_lat": top["lat_r"], "district_lon": top["lon_r"], "map_lat": top["lat_r"], "map_lon": top["lon_r"]},
            )
        )
    if "__text__" in row_df.columns and str(row.get("__text__") or "").strip():
        scenarios.append(("no ad text", {"__text__": ""}))

    out = []
    for name, changes in scenarios:
        alt_row = row_df.copy()
        for k, v in changes.items():
            alt_row.at[0, k] = v
        y = _predict(model, alt_row, meta)
        out.append({"change": name, "price_per_m": round(y), "delta": round(y - base), "delta_pct": round((y - base) / base * 100, 1)})
    return pd.DataFrame(out).sort_values("delta", key=np.abs, ascending=False, ignore_index=True)


def _compute_days_since_train_min(created_at_raw: str | None, train_min_raw: str | None) -> float | None:
    if not created_at_raw or not train_min_raw:
        return None
    # Parse both as UTC to avoid tz-aware vs tz-naive subtraction errors.
    created_at = pd.to_datetime(created_at_raw, errors="coerce", utc=True)
    train_min = pd.to_datetime(train_min_raw, errors="coerce", utc=True)
    if pd.isna(created_at) or pd.isna(train_min):
        return None
    return float((created_at - train_min).total_seconds() / 86400.0)


def _aggregate_coverage(df: pd.DataFrame) -> pd.DataFrame:
    x = df[["district_lat", "district_lon"]].dropna().copy()
    if x.empty:
        return pd.DataFrame()
    x["lat_r"] = x["district_lat"].round(3)
    x["lon_r"] = x["district_lon"].round(3)
    grp = (
        x.groupby(["lat_r", "lon_r"])
        .size()
        .sort_values(ascending=False)
        .rename("count")
        .reset_index()
    )
    total = int(grp["count"].sum())
    grp["share_pct"] = grp["count"] / total * 100.0
    grp["district_hint"] = grp.apply(
        lambda r: DISTRICT_LABELS.get((float(r["lat_r"]), float(r["lon_r"])), "Unknown cluster"),
        axis=1,
    )
    return grp


@st.cache_data(show_spinner=False)
def _district_coverage(months: int) -> pd.DataFrame:
    csv_path = ROOT / "real_estate_ads.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(
        csv_path,
        parse_dates=["created_at", "created_at_first"],
        low_memory=False,
    )
    base = df.dropna(subset=["price_per_m", "m", "created_at", "market"]).copy()
    base = base[(base["m"] > 10) & (base["m"] < 500)]
    base = base[(base["price_per_m"] > 500) & (base["price_per_m"] < 50_000)]
    cut = base["created_at"].max() - pd.DateOffset(months=months)
    recent = base.loc[base["created_at"] >= cut].copy()
    if recent.empty:
        return pd.DataFrame()
    return _aggregate_coverage(recent)


def _district_coverage_from_examples(examples: list[dict]) -> pd.DataFrame:
    rows = [e.get("x", {}) for e in examples if isinstance(e, dict)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "district_lat" not in df.columns or "district_lon" not in df.columns:
        return pd.DataFrame()
    return _aggregate_coverage(df)


def _district_coverage_from_meta(meta: dict) -> pd.DataFrame:
    grp = pd.DataFrame(meta.get("district_coverage") or [])
    if grp.empty:
        return grp
    grp["district_hint"] = grp.apply(
        lambda r: DISTRICT_LABELS.get((float(r["lat_r"]), float(r["lon_r"])), "Unknown cluster"),
        axis=1,
    )
    return grp


def _prediction_bounds(meta: dict, examples: list[dict]) -> tuple[float, float] | None:
    # Prefer target quantiles of the full training window (saved by the training script).
    if meta.get("target_q01") is not None and meta.get("target_q99") is not None:
        return float(meta["target_q01"]), float(meta["target_q99"])
    ys = [float(e.get("y_true")) for e in examples if isinstance(e, dict) and e.get("y_true") is not None]
    if len(ys) < 3:
        return None
    s = pd.Series(ys)
    lo = float(s.quantile(0.05))
    hi = float(s.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    return lo, hi


st.set_page_config(page_title="Real-estate price_per_m predictor", layout="wide")

st.title("Real-estate `price_per_m` predictor (demo)")
st.caption(
    "Loads a saved sklearn pipeline trained on the last N months slice and runs interactive predictions."
)

with st.sidebar:
    st.header("Artifacts")
    months = st.number_input("Months window", min_value=1, max_value=36, value=6, step=1)
    model_kind = st.selectbox("Model kind", options=["combined", "tabular"], index=0)

    paths = _default_artifact_paths(months=int(months), model_kind=str(model_kind))
    model_path = Path(st.text_input("Model path", value=str(paths["model"])))
    meta_path = Path(st.text_input("Metadata path", value=str(paths["meta"])))
    interval_path = Path(st.text_input("Interval path", value=str(paths["interval"])))
    examples_path = Path(st.text_input("Examples path", value=str(paths["examples"])))

    st.divider()
    st.caption("If these paths don’t exist yet, run the training script to generate them.")


@st.cache_resource(show_spinner=True)
def _load_all(model_path: str, meta_path: str, interval_path: str, examples_path: str):
    model = load_model(model_path)
    meta = load_json(meta_path)
    interval = load_json(interval_path)
    examples = load_json(examples_path)
    return model, meta, interval, examples


try:
    model, meta, interval, examples = _load_all(
        str(model_path), str(meta_path), str(interval_path), str(examples_path)
    )
except Exception as e:
    st.error(
        "Could not load artifacts. Make sure you trained and saved the model first."
    )
    st.exception(e)
    st.stop()

raw_cols: list[str] = list(meta.get("raw_columns", []))
if not raw_cols:
    st.error("Metadata is missing `raw_columns`.")
    st.stop()

st.subheader("Model info")
left, right = st.columns(2)
with left:
    st.write(
        {
            "slice": meta.get("slice"),
            "model_kind": meta.get("model_kind"),
            "window_cut": meta.get("window_cut"),
            "window_max": meta.get("window_max"),
            "n_train": meta.get("n_train"),
            "n_test": meta.get("n_test"),
            "split_ts": meta.get("split_ts"),
        }
    )
with right:
    st.write({"interval_residual_q05": interval.get("residual_q05"), "interval_residual_q95": interval.get("residual_q95")})
    test_metrics = meta.get(f"metrics_{meta.get('model_kind')}")
    if test_metrics:
        st.write({"test_mape_%": round(test_metrics["mape_pct"], 2), "test_mae": round(test_metrics["mae"])})

st.divider()

st.subheader("District guide (from training coverage)")
coverage = _district_coverage_from_meta(meta)
if coverage.empty:
    coverage = _district_coverage(int(months))
if coverage.empty:
    coverage = _district_coverage_from_examples(examples)
if not coverage.empty:
    shown = coverage.head(8).copy()
    st.dataframe(
        shown[["district_hint", "lat_r", "lon_r", "count", "share_pct"]].rename(
            columns={
                "district_hint": "district",
                "lat_r": "district_lat~",
                "lon_r": "district_lon~",
                "count": "train_rows",
                "share_pct": "coverage_%",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Pick examples/ads close to high-coverage districts for more reliable predictions."
    )
else:
    st.info("Could not compute district coverage from dataset or examples.")

st.divider()

st.subheader("Input")

input_mode = st.radio(
    "Input source",
    options=["Example mode", "URL mode"],
    horizontal=True,
    help="Example mode uses saved test examples. URL mode focuses only on the fetched ad.",
)

ad_url = ""
if input_mode == "URL mode":
    st.caption("URL mode: only URL-based prediction is shown.")
    with st.expander("Prefill from Otodom URL (experimental)", expanded=True):
        ad_url = st.text_input(
            "Otodom ad URL",
            value="",
            placeholder="https://www.otodom.pl/...",
        )
        fetch_from_url = st.button("Fetch parameters from URL")
        if fetch_from_url:
            if not ad_url.strip():
                st.warning("Paste a URL first.")
            else:
                try:
                    extracted = extract_otodom_features(ad_url.strip())
                    # Compute _days from fetched publish date vs train reference when possible.
                    created_at_raw = extracted.get("__created_at__")
                    train_min_raw = meta.get("date_min")
                    days = _compute_days_since_train_min(created_at_raw, train_min_raw)
                    if days is not None:
                        extracted["_days"] = days
                    else:
                        extracted.setdefault("_days", 0.0)
                    st.session_state["url_prefill_row"] = extracted
                    st.success("Fetched data from URL. Form has been prefilled.")
                except Exception as e:
                    st.error("Could not fetch/parse this URL.")
                    st.exception(e)
else:
    st.caption("Example mode: select from saved temporal test examples.")

ex_options = [f"Example #{i+1}" for i in range(len(examples))]
ex_choice = "(none)"
if input_mode == "Example mode":
    ex_choice = st.selectbox(
        "Load example (from temporal test set)",
        options=["(none)"] + ex_options,
        index=1 if ex_options else 0,
    )

base_row: dict = {}
if input_mode == "Example mode" and ex_choice != "(none)":
    idx = int(ex_choice.split("#")[1]) - 1
    base_row = dict(examples[idx]["x"])

url_prefill_row = st.session_state.get("url_prefill_row", {})
if input_mode == "URL mode" and isinstance(url_prefill_row, dict):
    base_row.update(url_prefill_row)

if input_mode == "URL mode" and isinstance(url_prefill_row, dict) and url_prefill_row:
    with st.expander("Fetched values preview (from URL)"):
        shown = {
            "market": url_prefill_row.get("market"),
            "m": url_prefill_row.get("m"),
            "no_rooms": url_prefill_row.get("no_rooms"),
            "district_lat": url_prefill_row.get("district_lat"),
            "district_lon": url_prefill_row.get("district_lon"),
            "map_lat": url_prefill_row.get("map_lat"),
            "map_lon": url_prefill_row.get("map_lon"),
            "_days": url_prefill_row.get("_days"),
            "__created_at__": url_prefill_row.get("__created_at__"),
            "__title__": url_prefill_row.get("__title__"),
            "__description__": url_prefill_row.get("__description__"),
        }
        st.write(shown)

row_df = _ensure_all_columns(base_row, raw_cols)
# Keep editable row flexible in Streamlit widgets:
# some example columns can come in as pandas Arrow string dtype, which rejects
# assigning numeric widget values (e.g., no_rooms=2.0). Object dtype avoids this.
row_df = row_df.astype(object)

main_left, main_right = st.columns([2, 1])

with main_left:
    c1, c2, c3 = st.columns(3)
    with c1:
        if "market" in row_df.columns:
            market = st.selectbox(
                "market",
                options=["secondary", "primary"],
                index=0
                if str(row_df.at[0, "market"] or "secondary") == "secondary"
                else 1,
            )
            row_df.at[0, "market"] = market
    with c2:
        if "m" in row_df.columns:
            m = st.number_input(
                "m (area)",
                min_value=5.0,
                max_value=500.0,
                value=float(row_df.at[0, "m"] or 50.0),
                step=1.0,
            )
            row_df.at[0, "m"] = m
    with c3:
        if "no_rooms" in row_df.columns:
            rooms = st.number_input(
                "no_rooms",
                min_value=0.0,
                max_value=20.0,
                value=float(row_df.at[0, "no_rooms"] or 2.0),
                step=1.0,
            )
            row_df.at[0, "no_rooms"] = rooms

    geo1, geo2 = st.columns(2)
    with geo1:
        if "district_lat" in row_df.columns:
            row_df.at[0, "district_lat"] = st.number_input(
                "district_lat",
                value=float(row_df.at[0, "district_lat"] or 0.0),
                step=0.0001,
                format="%.6f",
            )
        if "district_lon" in row_df.columns:
            row_df.at[0, "district_lon"] = st.number_input(
                "district_lon",
                value=float(row_df.at[0, "district_lon"] or 0.0),
                step=0.0001,
                format="%.6f",
            )
    with geo2:
        if "map_lat" in row_df.columns:
            row_df.at[0, "map_lat"] = st.number_input(
                "map_lat",
                value=float(row_df.at[0, "map_lat"] or 0.0),
                step=0.0001,
                format="%.6f",
            )
        if "map_lon" in row_df.columns:
            row_df.at[0, "map_lon"] = st.number_input(
                "map_lon",
                value=float(row_df.at[0, "map_lon"] or 0.0),
                step=0.0001,
                format="%.6f",
            )

    if "__text__" in row_df.columns:
        with st.expander("Text (optional, used by combined model)"):
            default_title = str(url_prefill_row.get("__title__", "")) if isinstance(url_prefill_row, dict) else ""
            default_description = str(url_prefill_row.get("__description__", "")) if isinstance(url_prefill_row, dict) else ""
            title = st.text_input("title (optional)", value=default_title)
            description = st.text_area("description (optional)", value=default_description, height=120)
            combined = (strip_html_to_text(title) + " " + strip_html_to_text(description)).strip()
            if combined:
                row_df.at[0, "__text__"] = combined
            elif base_row.get("__text__"):
                # Preserve text fetched from URL when user does not type title/description.
                row_df.at[0, "__text__"] = str(base_row.get("__text__"))

    if "_days" in row_df.columns:
        row_df.at[0, "_days"] = st.number_input(
            "_days (days since train-min created_at)",
            value=float(row_df.at[0, "_days"] or 0.0),
            step=1.0,
        )

with main_right:
    st.write("Raw row sent to model")
    st.dataframe(row_df, width="stretch", hide_index=True)

st.divider()

st.subheader("Prediction")
col_pred, col_int = st.columns(2)
with col_pred:
    btn_label = "Predict for selected example" if input_mode == "Example mode" else "Predict for URL ad"
    do_pred = st.button(btn_label)
with col_int:
    st.caption("Interval is approximate: computed from residual quantiles on the temporal test set.")

if do_pred:
    raw_y_hat = _predict(model, row_df, meta)
    y_hat = raw_y_hat
    clip_warning = None
    bounds = _prediction_bounds(meta, examples)
    if bounds is not None:
        b_lo, b_hi = bounds
        y_hat = float(np.clip(raw_y_hat, b_lo, b_hi))
        if not np.isclose(y_hat, raw_y_hat):
            clip_warning = (
                f"Raw prediction {raw_y_hat:,.0f} is outside the typical training range "
                f"[{b_lo:,.0f}, {b_hi:,.0f}]. Showing clipped value for stability."
            )
    # Keep the result across reruns (e.g. when the "Explain" button is clicked).
    st.session_state["last_prediction"] = {"row_json": row_df.to_json(), "y_hat": y_hat, "clip_warning": clip_warning}
    st.session_state.pop("llm_explanation", None)

pred_state = st.session_state.get("last_prediction")
if pred_state and pred_state["row_json"] != row_df.to_json():
    st.info("Inputs changed since the last prediction - click the predict button again.")
    pred_state = None

if pred_state:
    y_hat = pred_state["y_hat"]
    if pred_state["clip_warning"]:
        st.warning(pred_state["clip_warning"])
    days_max = meta.get("days_max")
    row_days = pd.to_numeric(row_df.at[0, "_days"], errors="coerce") if "_days" in row_df.columns else np.nan
    if days_max is not None and pd.notna(row_days) and row_days > days_max:
        st.info(
            f"This ad is newer than the training data (which ends {str(meta.get('date_max', ''))[:10]}). "
            "The prediction reflects market prices from that period, not today's."
        )
    q05 = float(interval["residual_q05"])
    q95 = float(interval["residual_q95"])
    lo = y_hat + q05
    hi = y_hat + q95

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted price_per_m", f"{y_hat:,.0f}")
    p2.metric("Lower (q05)", f"{lo:,.0f}")
    p3.metric("Upper (q95)", f"{hi:,.0f}")

    st.write("Prediction interval view")
    viz_df = pd.DataFrame(
        [
            {"kind": "lower", "value": lo},
            {"kind": "prediction", "value": y_hat},
            {"kind": "upper", "value": hi},
        ]
    )
    rule_df = pd.DataFrame([{"start": lo, "end": hi, "y": "price_per_m"}])
    span = max(1.0, hi - lo)
    x_min = lo - 0.15 * span
    x_max = hi + 0.15 * span
    x_scale = alt.Scale(domain=[x_min, x_max], nice=False, zero=False)

    interval_rule = (
        alt.Chart(rule_df)
        .mark_rule(strokeWidth=8)
        .encode(
            x=alt.X("start:Q", title="price_per_m", scale=x_scale),
            x2="end:Q",
            y=alt.Y("y:N", title=""),
            color=alt.value("#4E79A7"),
        )
    )
    pred_point = (
        alt.Chart(viz_df[viz_df["kind"] == "prediction"])
        .mark_point(size=140, filled=True)
        .encode(
            x=alt.X("value:Q", title="price_per_m", scale=x_scale),
            y=alt.value(30),
            color=alt.value("#E15759"),
            tooltip=[alt.Tooltip("value:Q", format=",.0f", title="prediction")],
        )
    )
    bounds_points = (
        alt.Chart(viz_df[viz_df["kind"] != "prediction"])
        .mark_tick(thickness=3, size=28)
        .encode(
            x=alt.X("value:Q", title="price_per_m", scale=x_scale),
            y=alt.value(30),
            color=alt.value("#4E79A7"),
            tooltip=[
                alt.Tooltip("kind:N", title="bound"),
                alt.Tooltip("value:Q", format=",.0f", title="value"),
            ],
        )
    )
    chart = (interval_rule + bounds_points + pred_point).properties(height=90)
    st.altair_chart(chart, width="stretch")
    st.caption("Zoomed x-axis around interval. Blue segment = [q05, q95], red point = prediction.")

    st.subheader("Explain prediction (XAI)")
    st.caption(
        "What-if analysis: one input is changed at a time while all others stay fixed. "
        "An LLM then turns these numbers into a short explanation."
    )
    effects = _what_if_effects(model, row_df, meta, coverage)
    st.dataframe(effects, width="stretch", hide_index=True)

    api_key = resolve_api_key()
    if not api_key:
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Or set OPENAI_API_KEY in .env, environment variables or Streamlit secrets.",
        )
    llm_model = st.text_input("LLM model", value=DEFAULT_LLM_MODEL)
    if st.button("Explain with LLM", disabled=not api_key):
        with st.spinner("Calling LLM..."):
            try:
                st.session_state["llm_explanation"] = explain_prediction(
                    api_key=api_key,
                    model=llm_model.strip() or DEFAULT_LLM_MODEL,
                    row=row_df.iloc[0].to_dict(),
                    prediction={"price_per_m": round(y_hat), "lower_q05": round(lo), "upper_q95": round(hi)},
                    effects=effects.to_dict(orient="records"),
                    meta=meta,
                )
            except Exception as e:
                st.error("LLM call failed.")
                st.exception(e)
    if st.session_state.get("llm_explanation"):
        st.markdown(st.session_state["llm_explanation"])

st.divider()

if input_mode == "Example mode":
    st.subheader("Example predictions (from saved temporal test rows)")
if input_mode == "Example mode" and examples:
    q05 = float(interval["residual_q05"])
    q95 = float(interval["residual_q95"])
    ex_rows = []
    for i, e in enumerate(examples, start=1):
        y_pred = float(e["y_pred"])
        ex_rows.append(
            {
                "example_id": i,
                "y_true": float(e["y_true"]),
                "y_pred": y_pred,
                "lower_q05": y_pred + q05,
                "upper_q95": y_pred + q95,
            }
        )
    ex_df = pd.DataFrame(ex_rows)

    if ex_choice != "(none)":
        selected_idx = int(ex_choice.split("#")[1])
        selected_row = ex_df.loc[ex_df["example_id"] == selected_idx].iloc[0]
        s_lat = float(examples[selected_idx - 1]["x"].get("district_lat") or np.nan)
        s_lon = float(examples[selected_idx - 1]["x"].get("district_lon") or np.nan)
        district_name = "Unknown cluster"
        if np.isfinite(s_lat) and np.isfinite(s_lon):
            key = (round(s_lat, 3), round(s_lon, 3))
            district_name = DISTRICT_LABELS.get(key, district_name)
        st.write("Selected example prediction")
        st.caption(f"District hint: **{district_name}** ({s_lat:.3f}, {s_lon:.3f})")
        s1, s2, s3 = st.columns(3)
        s1.metric("Selected y_pred", f"{selected_row['y_pred']:,.0f}")
        s2.metric("Selected lower (q05)", f"{selected_row['lower_q05']:,.0f}")
        s3.metric("Selected upper (q95)", f"{selected_row['upper_q95']:,.0f}")
        st.dataframe(
            pd.DataFrame([selected_row]).drop(columns=["example_id"]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.dataframe(ex_df, width="stretch", hide_index=True)

    with st.expander("Show all example predictions"):
        st.dataframe(ex_df, width="stretch", hide_index=True)
elif input_mode == "Example mode":
    st.info("No examples file found or it was empty.")

