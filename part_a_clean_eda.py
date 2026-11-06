import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from shipment_prep import clean_shipments, build_features, FEATURES

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# "before" counts: after duplicates removed, before any filling
dedup = raw.drop_duplicates()
dv_before = pd.to_numeric(dedup["declared_value"].astype("string").str.replace("₹", "", regex=False)
                          .str.replace(",", "", regex=False), errors="coerce")
sentinel = int((dedup["parcel_weight_kg"] > 100).sum())
spell_before = int(dedup["service_type"].nunique())

clean, medians = clean_shipments(raw)
df = build_features(clean, hubs)
df["is_delayed"] = (df["delivery_hours"] > df["promised_hours"]).astype(int)
df.to_csv("data/modelling_table.csv", index=False)

q1, q3 = df["delivery_hours"].quantile(0.25), df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(dedup["parcel_weight_kg"].isna().sum()),
    "sentinel_weight_rows": sentinel,
    "missing_declared_value_before": int(dv_before.isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": spell_before,
    "service_type_spellings_after": int(df["service_type"].nunique()),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(df),
    "nulls_in_features_and_target": int(df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(df["is_delayed"].sum()),
    "delay_rate": df["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": int(((df["delivery_hours"] < lo) | (df["delivery_hours"] > hi)).sum()),
    "declared_value_mean_inr": df["declared_value_inr"].mean(),
    "declared_value_median_inr": df["declared_value_inr"].median(),
    "log_declared_value_mean": df["log_declared_value"].mean(),
    "log_declared_value_median": df["log_declared_value"].median(),
}
out = pd.DataFrame({"metric": list(checks), "value": [round(float(v), 4) for v in checks.values()]})
out.to_csv("outputs/part_a_checks.csv", index=False)

df["distance_band"] = pd.cut(df["distance_km"], bins=[0, 200, 400, 600, 2000],
                             labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = df.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                           delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    for grp, r in g.iterrows():
        rows.append({"dimension": dim, "group": str(grp), "shipments": int(r.shipments), "delayed": int(r.delayed),
                     "delay_rate": round(r.delay_rate, 4), "mean_hours": round(r.mean_hours, 4)})
pd.DataFrame(rows).to_csv("outputs/part_a_delay_by_group.csv", index=False)

num = df.drop(columns=["distance_band"]).select_dtypes(include=[np.number])
num.corr().round(4).to_csv("outputs/part_a_correlations.csv")
print(out.to_string(index=False))
