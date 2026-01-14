from flask import Flask, render_template, request, redirect, url_for, jsonify
from sqlalchemy import create_engine
import pandas as pd
import plotly.express as px
import os, time, hashlib
import numpy as np

app = Flask(__name__)

# --- DB config (USE env vars; don't hardcode passwords) ---
SQL_SERVER   = os.getenv("AZURE_SQL_SERVER",   "auseast.database.windows.net")
SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE", "JobMarketDb")
SQL_USERNAME = os.getenv("AZURE_SQL_USERNAME", "CloudSAfc8188ca")
SQL_PASSWORD = os.getenv("AZURE_SQL_PASSWORD", "Aryakrish24")

ENGINE = create_engine(
    f"mssql+pyodbc://{SQL_USERNAME}:{SQL_PASSWORD}@{SQL_SERVER}:1433/{SQL_DATABASE}"
    "?driver=ODBC+Driver+18+for+SQL+Server"
    "&Encrypt=yes"
    "&TrustServerCertificate=no"
)

# --- Base cache (raw DB df) ---
_cache = {"ts": 0, "df": None}
CACHE_TTL_SECONDS = 60

def load_data_cached():
    now = time.time()
    if _cache["df"] is not None and (now - _cache["ts"]) < CACHE_TTL_SECONDS:
        return _cache["df"]

    df = pd.read_sql(
        """
        SELECT *
        FROM JobPosts
        WHERE created >= DATEADD(day, -90, GETUTCDATE())
        ORDER BY created DESC
        """,
        ENGINE
    )
    _cache["df"] = df
    _cache["ts"] = now
    return df


# --- “Processed df” cache (parsed timestamps + job_key) ---
_cache_keys = {"ts": 0, "df": None}
CACHE_KEYS_TTL_SECONDS = 180  # 2–5 mins is fine

def _make_job_key(df: pd.DataFrame) -> pd.Series:
    """
    Faster than df.apply(axis=1): build one string series then hash.
    """
    s = (
        df["title"].fillna("").astype(str) + "|" +
        df["company"].fillna("").astype(str) + "|" +
        df["location"].fillna("").astype(str)
    )
    # sha256 per row
    return s.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())

def load_data_with_keys_cached():
    now = time.time()
    if _cache_keys["df"] is not None and (now - _cache_keys["ts"]) < CACHE_KEYS_TTL_SECONDS:
        return _cache_keys["df"]

    df = load_data_cached().copy()

    # Parse timestamps ONCE
    df["created"] = pd.to_datetime(df["created"], utc=True, errors="coerce")
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")

    # Compute job_key ONCE
    df["job_key"] = _make_job_key(df)

    _cache_keys["df"] = df
    _cache_keys["ts"] = now
    return df


def _bucketize(ts: pd.Series, bucket: str) -> pd.Series:
    """
    Avoid timezone warning from .to_period() by converting to tz-naive first.
    bucket = "D" or "M"
    """
    ts_naive = ts.dt.tz_convert(None)
    return ts_naive.dt.to_period(bucket).dt.start_time


@app.route("/", methods=["GET"])
def home():
    df = load_data_cached()
    roles = sorted(df["role"].dropna().unique().tolist()) if not df.empty else []
    return render_template("home.html", roles=roles)

@app.route("/insights")
def insights():
    q = (request.args.get("q") or "").strip()
    return render_template("insights.html", query=q)


@app.route("/api/market_summary")
def api_market_summary():
    # Small timing helper
    t0 = time.time()
    def log(msg):
        nonlocal t0
        print(msg, round(time.time() - t0, 3))
        t0 = time.time()

    q = (request.args.get("q") or "").strip().lower()
    loc = (request.args.get("loc") or "").strip().lower()
    window = (request.args.get("window") or "month").strip().lower()

    df = load_data_with_keys_cached().copy()
    log("load_data_with_keys_cached:")

    # Filters
    if q:
        df = df[df["role"].astype(str).str.lower().str.contains(q, na=False)]
    if loc:
        df = df[df["location"].astype(str).str.lower().str.contains(loc, na=False)]
    log("filters:")

    now = pd.Timestamp.now(tz="UTC")
    if window == "year":
        start = now - pd.Timedelta(days=365)
        bucket = "M"
    else:
        start = now - pd.Timedelta(days=30)
        bucket = "D"

    # Created counts
    created_df = df[df["created"] >= start].copy()
    created_df["bucket"] = _bucketize(created_df["created"], bucket)
    created_series = created_df.groupby("bucket")["job_key"].nunique().sort_index()
    log("created_series:")

    # Closed counts (simple approximation using latest snapshot)
    latest_snap = df["fetched_at"].max()
    latest_keys = set(df.loc[df["fetched_at"] == latest_snap, "job_key"].unique())

    df["is_closed"] = ~df["job_key"].isin(latest_keys)

    closed_df = df[(df["is_closed"]) & (df["created"] >= start)].copy()
    closed_df["bucket"] = _bucketize(closed_df["created"], bucket)
    closed_series = closed_df.groupby("bucket")["job_key"].nunique().sort_index()
    log("closed_series:")

    # Align dates
    idx = created_series.index.union(closed_series.index)
    created_series = created_series.reindex(idx, fill_value=0)
    closed_series = closed_series.reindex(idx, fill_value=0)

    k_created = int(created_series.sum())
    k_closed = int(closed_series.sum())
    k_active = int(len(latest_keys))
    k_net = int(k_created - k_closed)

    chart_df = pd.DataFrame({
        "date": idx,
        "created": created_series.values,
        "closed": closed_series.values,
    })

    fig = px.line(chart_df, x="date", y=["created", "closed"], title=None)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Jobs",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Raleway"),
    )
    log("plotly:")

    return jsonify({
        "kpis": {"created": k_created, "closed": k_closed, "net": k_net, "active_now": k_active},
        "chart": fig.to_plotly_json()
    })


@app.route("/api/forecast")
def api_forecast():
    q = (request.args.get("q") or "").strip().lower()
    loc = (request.args.get("loc") or "").strip().lower()
    years = int(request.args.get("years") or 5)

    df = load_data_with_keys_cached().copy()

    if q:
        df = df[df["role"].astype(str).str.lower().str.contains(q, na=False)]
    if loc:
        df = df[df["location"].astype(str).str.lower().str.contains(loc, na=False)]

    # Monthly job count
    m = df.dropna(subset=["created"]).copy()
    m["month"] = _bucketize(m["created"], "M")
    series = m.groupby("month").size().sort_index()

    if len(series) < 6:
        fig = px.line(title=None)
        fig.update_layout(
            margin=dict(l=20,r=20,t=20,b=20),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Raleway")
        )
        return jsonify({"chart": fig.to_plotly_json(), "note": "Not enough data to forecast."})

    # Simple linear trend fit (fast)
    y = series.values.astype(float)
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)  # y ≈ a*x + b

    horizon_months = years * 12
    x_future = np.arange(len(y) + horizon_months)
    y_hat = a * x_future + b
    y_hat = np.clip(y_hat, 0, None)

    hist_x = series.index
    hist_fit = y_hat[:len(y)]

    future_months = pd.date_range(hist_x[-1], periods=horizon_months + 1, freq="MS")[1:]
    future_y = y_hat[len(y):]

    fig = px.line(title=None)
    fig.add_scatter(x=hist_x, y=series.values, mode="lines+markers", name="Historical")
    fig.add_scatter(x=hist_x, y=hist_fit, mode="lines", name="Trend fit")
    fig.add_scatter(x=future_months, y=future_y, mode="lines", name="Forecast")

    fig.update_layout(
        margin=dict(l=20,r=20,t=20,b=20),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Jobs per month",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Raleway")
    )

    return jsonify({"chart": fig.to_plotly_json()})


@app.route("/api/insights")
def api_insights():
    t0 = time.time()

    q = (request.args.get("q") or "").strip().lower()
    loc = (request.args.get("loc") or "").strip().lower()

    df = load_data_with_keys_cached().copy()
    print("load_data_with_keys_cached:", round(time.time() - t0, 3)); t0 = time.time()

    # Filters
    if q:
        df = df[df["role"].astype(str).str.lower().str.contains(q, na=False)]
    if loc:
        df = df[df["location"].astype(str).str.lower().str.contains(loc, na=False)]
    print("filters:", round(time.time() - t0, 3)); t0 = time.time()

    # Latest snapshot
    latest_snapshot_time = df["fetched_at"].max()
    latest_df = df[df["fetched_at"] == latest_snapshot_time].copy()

    # Dedupe current listings
    latest_df = latest_df.drop_duplicates(subset=["job_key"])

    # Nearby
    nearby_df = latest_df.sort_values("created", ascending=False).head(50)

    # Top salary
    top_salary_df = (
        latest_df.dropna(subset=["salary_avg"])
                 .sort_values("salary_avg", ascending=False)
                 .head(50)
    )

    # Opening soon
    opening_df = (
        latest_df[latest_df["created"] >= (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7))]
        .sort_values("created", ascending=False)
        .head(50)
    )

    # Closing soon approximation:
    latest_keys = set(latest_df["job_key"].unique())
    recently_seen = df[df["created"] >= (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30))].copy()
    recently_seen = recently_seen.sort_values("fetched_at").drop_duplicates("job_key", keep="last")

    closing_df = (
        recently_seen[~recently_seen["job_key"].isin(latest_keys)]
        .sort_values("fetched_at", ascending=False)
        .head(50)
    )

    # Related (FIXED: mask is built on latest_df so no reindex warning)
    related_df = pd.DataFrame(columns=latest_df.columns)
    if q:
        q_words = [w for w in q.split() if len(w) >= 3]
        if q_words:
            mask = latest_df["role"].astype(str).str.lower().apply(lambda r: any(w in r for w in q_words))
            related_df = latest_df.loc[mask].drop_duplicates("job_key").head(50)

    def fmt_out(dfx: pd.DataFrame):
        cols = ["title", "company", "location", "salary_avg", "created"]
        out = dfx[cols].copy()
        out["created"] = pd.to_datetime(out["created"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")
        return out.to_dict("records")

    print("total endpoint:", round(time.time() - t0, 3))

    return jsonify({
        "nearby": fmt_out(nearby_df),
        "top_salary": fmt_out(top_salary_df),
        "opening_soon": fmt_out(opening_df),
        "closing_soon": fmt_out(closing_df),
        "related": fmt_out(related_df),
    })


@app.route("/search", methods=["GET"])
def search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return redirect(url_for("home"))
    return redirect(url_for("dashboard", role=q))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        df = load_data_with_keys_cached().copy()
    except Exception as e:
        return render_template("dashboard.html", error=str(e), roles=[], selected_role=None)

    if df.empty:
        return render_template("dashboard.html", error=None, roles=[], selected_role=None, empty=True)

    roles = sorted([r for r in df["role"].dropna().unique()])
    selected_role = request.args.get("role") or (roles[0] if roles else None)

    filtered = df[df["role"] == selected_role] if selected_role else df
    total_jobs = len(filtered)

    fig1 = px.histogram(filtered, x="salary_avg", title="Salary Distribution")
    fig1_html = fig1.to_html(full_html=False, include_plotlyjs="cdn")

    filtered_sorted = filtered.sort_values("created")
    age_days = (pd.Timestamp.now(tz="UTC") - filtered_sorted["created"]).dt.days
    fig2 = px.line(filtered_sorted, x="created", y=age_days, title="Posting Age Over Time")
    fig2.update_layout(yaxis_title="Age (days)")
    fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

    role_df = filtered.copy()
    role_df["week"] = _bucketize(role_df["created"], "W")  # simple weekly bucket

    weekly_posted = role_df.groupby("week")["job_key"].nunique().reset_index(name="jobs_posted")

    latest_snapshot_time = role_df["fetched_at"].max()
    latest_snapshot = set(role_df[role_df["fetched_at"] == latest_snapshot_time]["job_key"].unique())
    role_df["is_closed"] = ~role_df["job_key"].isin(latest_snapshot)

    weekly_closed = role_df[role_df["is_closed"]].groupby("week")["job_key"].nunique().reset_index(name="jobs_closed")

    weekly_summary = pd.merge(weekly_posted, weekly_closed, on="week", how="left").fillna(0)
    weekly_summary = weekly_summary[(weekly_summary["jobs_posted"] > 0) | (weekly_summary["jobs_closed"] > 0)]

    fig3 = px.bar(
        weekly_summary.melt(id_vars="week", value_vars=["jobs_posted", "jobs_closed"]),
        x="week", y="value", color="variable",
        barmode="group",
        title=f"Weekly Jobs Posted vs Closed — {selected_role}"
    )
    fig3_html = fig3.to_html(full_html=False, include_plotlyjs=False)

    table_cols = ["title", "company", "location", "salary_avg", "created"]
    table_rows = filtered[table_cols].head(200).copy()
    table_rows["created"] = table_rows["created"].dt.strftime("%Y-%m-%d %H:%M UTC")

    return render_template(
        "dashboard.html",
        error=None,
        empty=False,
        roles=roles,
        selected_role=selected_role,
        total_jobs=total_jobs,
        fig1_html=fig1_html,
        fig2_html=fig2_html,
        fig3_html=fig3_html,
        table_rows=table_rows.to_dict(orient="records")
    )


if __name__ == "__main__":
    app.run(debug=True)
