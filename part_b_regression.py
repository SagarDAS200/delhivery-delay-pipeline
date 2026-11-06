import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.insert(0, "src")
from shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

scaler = StandardScaler().fit(X_tr)
pd.DataFrame({"metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
              "value": [len(X_tr), len(X_val), len(X_te),
                        round(float(np.abs(scaler.transform(X_tr).mean(axis=0)).max()), 4),
                        round(float(np.abs(scaler.transform(X_te).mean(axis=0)).max()), 4)]}
             ).to_csv("outputs/part_b_checks.csv", index=False)


def metrics(name, split, y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"model": name, "split": split, "MAE": round(mean_absolute_error(y_true, pred), 4),
            "MSE": round(mse, 4), "RMSE": round(float(np.sqrt(mse)), 4), "R2": round(r2_score(y_true, pred), 4)}


rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
base = np.full(len(y_val), y_tr.mean())

depths = [1, 2, 3, 4, 6, 8, 12, None]
rows = []
for d in depths:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_tr, y_tr)
    rows.append({"max_depth": "unlimited" if d is None else d,
                 "train_rmse": round(rmse(y_tr, t.predict(X_tr)), 4),
                 "val_rmse": round(rmse(y_val, t.predict(X_val)), 4)})
depth_df = pd.DataFrame(rows)
depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)
# lowest val_rmse; ties -> smaller depth (unlimited is the largest)
best_i = min(range(len(depths)), key=lambda i: (depth_df.loc[i, "val_rmse"], float("inf") if depths[i] is None else depths[i]))
best_depth = depths[best_i]

models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
res = [metrics("mean_baseline", "val", y_val, base)]
for n, m in models.items():
    m.fit(X_tr, y_tr)
    res.append(metrics(n, "val", y_val, m.predict(X_val)))
met = pd.DataFrame(res)

lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_tr, y_tr)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": round(rmse(y_val, p), 4), "val_r2": round(r2_score(y_val, p), 4)})
pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

cand = met[met["model"] != "mean_baseline"].sort_values("RMSE", kind="stable")
top2 = list(cand["model"][:2])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_trval, y_trval = pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val])
cv = []
for n in top2:
    s = cross_val_score(models[n], X_trval, y_trval, cv=kf)
    cv.append({"model": n, "mean_r2": round(s.mean(), 4), "std_r2": round(s.std(), 4)})
pd.DataFrame(cv).to_csv("outputs/part_b_cv.csv", index=False)

final = top2[0]
test_row = metrics(final, "test", y_te, models[final].predict(X_te))
pd.concat([met, pd.DataFrame([test_row])], ignore_index=True).to_csv("outputs/part_b_metrics.csv", index=False)
print("best depth", best_depth, "top2", top2, "final", final)
