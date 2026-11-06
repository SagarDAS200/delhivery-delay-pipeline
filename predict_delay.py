import sys
import joblib
import pandas as pd
sys.path.insert(0, "src")
from shipment_prep import clean_shipments, build_features

raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = {r.column: float(r.median) for r in imp.itertuples()}

clean, _ = clean_shipments(raw, medians)
feats = build_features(clean, hubs)
out = feats[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = model.predict_proba(feats[order])[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out = out.merge(segments, on="hub_id", how="left")
out["action"] = "No action"
out.loc[out["flagged"] == 1, "action"] = "Proactive customer call"
out.loc[(out["flagged"] == 1) & (out["segment"] == "Escalate"), "action"] = "Escalate to hub manager"
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(len(out), "parcels;", int(out["flagged"].sum()), "flagged;", int((out["action"] == "Escalate to hub manager").sum()), "escalations")
