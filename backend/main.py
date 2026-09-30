"""FinancePredict backend: trains LinearRegression models from data/finance_data.csv."""
import os, sqlite3, json, datetime as dt
import numpy as np, pandas as pd, joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "data", "finance_data.csv")
DB = os.path.join(BASE, "database", "finance.db")
MODELS = os.path.join(BASE, "models_store")
VERSION = "v1"
FEATURES = ["monthly_income", "rent", "food", "transportation", "utilities",
            "healthcare", "entertainment", "education", "other_expenses"]
EXP_COLS = FEATURES[1:]
CATEGORIES = ["Rent", "Food", "Transportation", "Utilities", "Healthcare", "Entertainment", "Education", "Other"]
for d in (os.path.dirname(CSV), os.path.dirname(DB), MODELS):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------- dataset
def generate_dataset(n=800, seed=42):
    """Synthetic but realistic INR dataset: expenses scale with income (plus noise);
    savings = income - expenses - irregular unrecorded spending."""
    r = np.random.default_rng(seed)
    inc = np.clip(r.lognormal(10.6, 0.5, n), 12000, 300000).round(-2)
    g = lambda lo, hi: r.uniform(lo, hi, n)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1), "age": r.integers(21, 60, n), "monthly_income": inc,
        "rent": inc * g(.15, .32) + r.normal(0, 800, n),
        "food": 2500 + inc * g(.06, .13) + r.normal(0, 500, n),
        "transportation": 800 + inc * g(.02, .07) + r.normal(0, 300, n),
        "utilities": 900 + inc * g(.02, .05) + r.normal(0, 200, n),
        "healthcare": inc * g(.01, .05) + r.normal(0, 300, n),
        "entertainment": inc * g(.02, .08) + r.normal(0, 400, n),
        "education": inc * g(0, .06) + r.normal(0, 300, n),
        "other_expenses": inc * g(.01, .06) + r.normal(0, 300, n)})
    df[EXP_COLS] = df[EXP_COLS].clip(lower=0).round(0)
    df["total_expenses"] = df[EXP_COLS].sum(axis=1)
    df["savings"] = (df.monthly_income - df.total_expenses - r.normal(0, 0.03, n) * inc).round(0)
    return df


def load_clean():
    if not os.path.exists(CSV):
        generate_dataset().to_csv(CSV, index=False)
    df = pd.read_csv(CSV)
    if df.empty:
        raise RuntimeError("Dataset is empty")
    # preprocessing: numeric coercion, missing values, validation, outlier removal
    num = FEATURES + ["total_expenses", "savings"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df[num] = df[num].fillna(df[num].median())
    df = df[(df[FEATURES] >= 0).all(axis=1)]
    q1, q3 = df.monthly_income.quantile([.25, .75])
    df = df[df.monthly_income.between(q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- training
def fit(df, cols, target, name, target_label):
    X, y = df[cols], df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)  # 80/20
    m = LinearRegression().fit(Xtr, ytr)  # OLS needs no scaling; coefficients stay in rupee units
    p = m.predict(Xte)
    mse = mean_squared_error(yte, p)
    joblib.dump(m, os.path.join(MODELS, f"{target}_{VERSION}.joblib"))
    terms = " ".join(f"{'+' if c >= 0 else '-'} {abs(c):.4f}×{f}" for c, f in zip(m.coef_, cols))
    return {"model": m, "info": {
        "name": name, "target": target, "target_label": target_label, "features": cols,
        "n_features": len(cols), "n_train": len(Xtr), "n_test": len(Xte), "version": VERSION,
        "intercept": float(m.intercept_), "coefficients": dict(zip(cols, map(float, m.coef_))),
        "equation": f"{target} = {m.intercept_:.2f} {terms}",
        "r2": float(r2_score(yte, p)), "mae": float(mean_absolute_error(yte, p)),
        "mse": float(mse), "rmse": float(np.sqrt(mse)),
        "test_actual": yte.round(0).tolist(), "test_pred": np.round(p, 0).tolist()}}


STATE = {}


def train_all():
    df = load_clean()
    STATE["df"] = df
    STATE["simple"] = fit(df, ["monthly_income"], "total_expenses", "Simple Linear Regression", "Monthly Expense")
    STATE["multi"] = fit(df, FEATURES, "savings", "Multiple Linear Regression", "Monthly Savings")


def insights():
    m = STATE["multi"]["info"]; s = STATE["simple"]["info"]
    co = m["coefficients"]
    strongest = max(EXP_COLS, key=lambda k: abs(co[k]))
    pos = [k for k, v in co.items() if v > 0]; neg = [k for k, v in co.items() if v < 0]
    return [
        f"Strongest expense coefficient (by magnitude): {strongest} ({co[strongest]:+.3f} savings per ₹1 increase, other features held constant).",
        f"Positive coefficients: {', '.join(pos) or 'none'}. Negative coefficients: {', '.join(neg) or 'none'}.",
        f"In the simple model, +₹1,000 income is associated with ≈ ₹{s['coefficients']['monthly_income'] * 1000:,.0f} more monthly expense.",
        f"In the multiple model, +₹1,000 income (all else equal) changes predicted savings by ≈ ₹{co['monthly_income'] * 1000:,.0f}.",
        f"The simple model explains {s['r2'] * 100:.1f}% and the multiple model {m['r2'] * 100:.1f}% of test-set variance in their targets.",
        "These are data-driven model insights about a sample dataset, not personal financial advice. Coefficients show association, not causation."]


# ---------------------------------------------------------------- database
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS expenses(id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL,
            category TEXT NOT NULL, amount REAL NOT NULL, description TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
            model_type TEXT, model_version TEXT, income REAL, inputs TEXT, predicted_value REAL, predicted_extra TEXT);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);""")


# ---------------------------------------------------------------- API
app = FastAPI(title="FinancePredict API")
MAXV = 10_000_000


class SimpleIn(BaseModel):
    income: float = Field(..., ge=0, le=MAXV)


class MultiIn(BaseModel):
    income: float = Field(..., ge=0, le=MAXV)
    rent: float = Field(..., ge=0, le=MAXV)
    food: float = Field(..., ge=0, le=MAXV)
    transportation: float = Field(..., ge=0, le=MAXV)
    utilities: float = Field(..., ge=0, le=MAXV)
    healthcare: float = Field(..., ge=0, le=MAXV)
    entertainment: float = Field(..., ge=0, le=MAXV)
    education: float = Field(..., ge=0, le=MAXV)
    other_expenses: float = Field(..., ge=0, le=MAXV)


class ExpenseIn(BaseModel):
    date: str
    category: str
    amount: float = Field(..., gt=0, le=MAXV)
    description: str = ""


class SettingsIn(BaseModel):
    monthly_income: float = Field(..., ge=0, le=MAXV)


@app.on_event("startup")
def startup():
    try:
        train_all()
    except Exception as e:  # surfaced via /health
        STATE["error"] = str(e)
    init_db() if "df" in STATE else None


@app.exception_handler(sqlite3.Error)
async def dberr(_, e):
    return JSONResponse(status_code=500, content={"detail": "Database error. Please try again."})


def ready():
    if "df" not in STATE:
        raise HTTPException(503, "Models unavailable: " + STATE.get("error", "training failed"))


def public(info, extra=False):
    keys = {"test_actual", "test_pred"}
    return {k: v for k, v in info.items() if extra or k not in keys}


def log_pred(model_type, income, inputs, val, extra):
    with db() as c:
        c.execute("INSERT INTO predictions(timestamp,model_type,model_version,income,inputs,predicted_value,predicted_extra) VALUES(?,?,?,?,?,?,?)",
                  (dt.datetime.now().isoformat(timespec="seconds"), model_type, VERSION, income, json.dumps(inputs), val, json.dumps(extra)))


@app.get("/health")
def health():
    return {"status": "ok" if "df" in STATE else "error", "rows": len(STATE.get("df", [])), "error": STATE.get("error")}


@app.get("/dataset")
def dataset(limit: int = 100):
    ready(); df = STATE["df"]
    return {"path": "backend/data/finance_data.csv", "rows": len(df), "columns": list(df.columns),
            "describe": json.loads(df[FEATURES + ["total_expenses", "savings"]].describe().round(1).to_json()),
            "sample": json.loads(df.head(limit).to_json(orient="records"))}


@app.get("/model/metrics")
def metrics():
    ready(); df = STATE["df"]; s = STATE["simple"]["info"]; m = STATE["multi"]["info"]
    pts = df.sample(min(300, len(df)), random_state=1)
    lo, hi = float(df.monthly_income.min()), float(df.monthly_income.max())
    line = [{"x": x, "y": float(STATE["simple"]["model"].predict(pd.DataFrame({"monthly_income": [x]}))[0])} for x in (lo, hi)]
    return {"simple": public(s), "multiple": public(m), "insights": insights(),
            "scatter": [{"x": a, "y": b} for a, b in zip(pts.monthly_income, pts.total_expenses)], "line": line,
            "actual_vs_pred": [{"x": a, "y": b} for a, b in zip(m["test_actual"], m["test_pred"])],
            "categories": CATEGORIES}


@app.post("/predict/simple")
def predict_simple(b: SimpleIn, save: bool = True):
    ready(); s = STATE["simple"]
    val = float(s["model"].predict(pd.DataFrame({"monthly_income": [b.income]}))[0])
    warn = ["Income is outside the range the model was trained on; extrapolation is unreliable."] \
        if not (STATE["df"].monthly_income.min() <= b.income <= STATE["df"].monthly_income.max()) else []
    if save:
        log_pred("Simple Linear Regression", b.income, {"income": b.income}, val, {})
    i = s["info"]
    return {"predicted_expense": round(val, 2), "model": i["name"], "r2": i["r2"], "mae": i["mae"], "rmse": i["rmse"],
            "equation": i["equation"], "warnings": warn, "note": "Model estimate, not a guarantee."}


@app.post("/predict/multiple")
def predict_multiple(b: MultiIn, save: bool = True):
    ready(); mm = STATE["multi"]
    d = b.model_dump()
    row = pd.DataFrame([[d["income"]] + [d[k] for k in EXP_COLS]], columns=FEATURES)
    val = float(mm["model"].predict(row)[0])
    total = float(sum(d[k] for k in EXP_COLS))
    warn = []
    if total > d["income"]: warn.append("Total expenses exceed income.")
    if d["income"] > 1_000_000: warn.append("Unusually high income; prediction may be unreliable.")
    rate = (val / d["income"] * 100) if d["income"] > 0 else 0.0
    extra = {"total_expenses": total, "savings_rate": rate, "disposable_income": d["income"] - total}
    if save:
        log_pred("Multiple Linear Regression", d["income"], d, val, extra)
    i = mm["info"]
    return {"predicted_savings": round(val, 2), "predicted_total_expenses": total, "savings_rate": round(rate, 2),
            "disposable_income": d["income"] - total, "model": i["name"], "r2": i["r2"], "mae": i["mae"],
            "mse": i["mse"], "rmse": i["rmse"], "coefficients": i["coefficients"], "intercept": i["intercept"],
            "equation": i["equation"], "warnings": warn, "note": "Model estimate, not a guarantee."}


# expenses
@app.get("/expenses")
def list_expenses(q: str = "", category: str = "", date_from: str = "", date_to: str = ""):
    sql, a = "SELECT * FROM expenses WHERE 1=1", []
    if q: sql += " AND description LIKE ?"; a.append(f"%{q}%")
    if category: sql += " AND category=?"; a.append(category)
    if date_from: sql += " AND date>=?"; a.append(date_from)
    if date_to: sql += " AND date<=?"; a.append(date_to)
    with db() as c:
        return [dict(r) for r in c.execute(sql + " ORDER BY date DESC, id DESC", a)]


def check_exp(e: ExpenseIn):
    if e.category not in CATEGORIES: raise HTTPException(422, "Unknown category.")
    try: dt.date.fromisoformat(e.date)
    except ValueError: raise HTTPException(422, "Date must be YYYY-MM-DD.")


@app.post("/expenses", status_code=201)
def add_expense(e: ExpenseIn):
    check_exp(e)
    with db() as c:
        cur = c.execute("INSERT INTO expenses(date,category,amount,description) VALUES(?,?,?,?)", (e.date, e.category, e.amount, e.description))
    return {"id": cur.lastrowid}


@app.put("/expenses/{eid}")
def edit_expense(eid: int, e: ExpenseIn):
    check_exp(e)
    with db() as c:
        if c.execute("UPDATE expenses SET date=?,category=?,amount=?,description=? WHERE id=?", (e.date, e.category, e.amount, e.description, eid)).rowcount == 0:
            raise HTTPException(404, "Expense not found.")
    return {"ok": True}


@app.delete("/expenses/{eid}")
def del_expense(eid: int):
    with db() as c:
        if c.execute("DELETE FROM expenses WHERE id=?", (eid,)).rowcount == 0:
            raise HTTPException(404, "Expense not found.")
    return {"ok": True}


# predictions history + settings
@app.get("/predictions")
def list_predictions(model_type: str = ""):
    with db() as c:
        rows = c.execute("SELECT * FROM predictions" + (" WHERE model_type=?" if model_type else "") + " ORDER BY id DESC",
                         [model_type] if model_type else []).fetchall()
    return [{**dict(r), "inputs": json.loads(r["inputs"]), "predicted_extra": json.loads(r["predicted_extra"] or "{}")} for r in rows]


@app.delete("/predictions/{pid}")
def del_prediction(pid: int):
    with db() as c:
        if c.execute("DELETE FROM predictions WHERE id=?", (pid,)).rowcount == 0:
            raise HTTPException(404, "Prediction not found.")
    return {"ok": True}


@app.get("/settings")
def get_settings():
    with db() as c:
        r = c.execute("SELECT value FROM settings WHERE key='monthly_income'").fetchone()
    return {"monthly_income": float(r[0]) if r else 0.0}


@app.put("/settings")
def put_settings(s: SettingsIn):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO settings VALUES('monthly_income',?)", (str(s.monthly_income),))
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")
