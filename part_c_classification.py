import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, f1_score
sys.path.insert(0, "src")
from shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_tr), len(X_val), len(X_te), "delayed", y_tr.sum(), y_val.sum(), y_te.sum())


def cm_row(label_key, label, y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {label_key: label, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# 2. baseline + classifiers
rows = [cm_row("model", "blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int))]
lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_tr, y_tr)
rows.append(cm_row("model", "logistic_regression", y_val, lr.predict(X_val)))
best_d, best_f = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_tr, y_tr)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f:
        best_d, best_f = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_tr, y_tr)
rows.append(cm_row("model", f"decision_tree_depth_{best_d}", y_val, tree.predict(X_val)))
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
rows.append(cm_row("model", "random_forest_default", y_val, rf.predict(X_val)))
pd.DataFrame(rows).to_csv("outputs/part_c_models.csv", index=False)

# 3. feature scores
sel_all = SelectKBest(f_classif, k="all").fit(X_tr, y_tr)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sel_all.scores_, "rf_importance": rf.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs[["f_score", "rf_importance"]] = fs[["f_score", "rf_importance"]].round(4)
fs.to_csv("outputs/part_c_feature_scores.csv", index=False)

ks = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)),
                  ("rf", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_tr, y_tr)
    ks.append({"k": k, "val_f1": round(f1_score(y_val, p.predict(X_val)), 4)})
pd.DataFrame(ks).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning (selector fitted on training rows only)
sel7 = SelectKBest(f_classif, k=7).fit(X_tr, y_tr)
Z_tr, Z_val = sel7.transform(X_tr), sel7.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Z_tr, y_tr)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Z_tr, y_tr)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Z_tr, y_tr)
tun = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
     "best_params": str(gs.best_params_), "best_cv_accuracy": round(gs.best_score_, 4),
     "val_f1": round(f1_score(y_val, gs.best_estimator_.predict(Z_val)), 4)},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
     "best_params": str(rs.best_params_), "best_cv_accuracy": round(rs.best_score_, 4),
     "val_f1": round(f1_score(y_val, rs.best_estimator_.predict(Z_val)), 4)},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "{'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1}", "best_cv_accuracy": np.nan,
     "val_f1": round(f1_score(y_val, default7.predict(Z_val)), 4)},
]
tun = pd.DataFrame(tun)
tun.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. final configuration: highest val_f1; ties -> default, grid, randomized
prio = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
win = min(tun.itertuples(), key=lambda r: (-r.val_f1, prio[r.search]))
if win.search == "default_forest_same_7_features":
    params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
else:
    params = (gs if win.search == "grid_search" else rs).best_params_
final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_tr, y_tr)

# 6. threshold
thr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pv = final.predict_proba(X_val)[:, 1]
trows = [cm_row("threshold", t, y_val, (pv >= t).astype(int)) for t in thr]
tdf = pd.DataFrame(trows)
tdf.to_csv("outputs/part_c_thresholds.csv", index=False)
ok = tdf[tdf["recall"] >= 0.80]
chosen = float(ok["threshold"].max()) if len(ok) else 0.1

# 7. one test evaluation
pt = final.predict_proba(X_te)[:, 1]
pd.DataFrame([cm_row("threshold", chosen, y_te, (pt >= chosen).astype(int))]).to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": 7, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": chosen,
               "chosen_from": win.search}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen", win.search, params, chosen)
