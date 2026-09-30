# FinancePredict
Personal Expense & Savings Prediction Using Linear and Multiple Linear Regression.

## Run
```bash
cd backend
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000
```
Open http://localhost:8000 (the API also serves the dashboard; docs at /docs).
On first start the app generates `backend/data/finance_data.csv` (800 rows), trains both models, saves them with joblib to `backend/models_store/`, and seeds demo expenses in SQLite.
The dashboard loads Chart.js from a CDN, so an internet connection is needed the first time.

## ML methodology
- Preprocessing: numeric coercion, median imputation, non-negative validation, IQR outlier removal on income.
- 80/20 train/test split (`random_state=42`); user input is never used for training.
- **Simple LR**: `monthly_income -> total_expenses`. **Multiple LR**: 9 features -> `savings`. Both use scikit-learn `LinearRegression`.
- Metrics (R², MAE, MSE, RMSE) are computed on the test set at training time.
- Scaling is skipped on purpose: OLS is scale-invariant and unscaled coefficients stay interpretable in rupees.

## Dataset note
The CSV is synthetic (expenses scale with income plus noise; savings = income − expenses − irregular spending). A high multiple-regression R² reflects that structure. Replace `finance_data.csv` with real data using the same columns to retrain.

## API
`GET /health`, `GET /dataset`, `GET /model/metrics`, `POST /predict/simple`, `POST /predict/multiple`, CRUD on `/expenses`, `GET/DELETE /predictions`, `GET/PUT /settings`.

## Future improvements
React/TypeScript/Tailwind/Recharts frontend, authentication, model retraining endpoint, regularised models (Ridge/Lasso), real datasets.
