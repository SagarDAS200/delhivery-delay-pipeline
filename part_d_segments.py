import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

prof = (df.groupby("hub_id").agg(shipments=("is_delayed", "size"), mean_delivery_hours=("delivery_hours", "mean"),
                                 delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                                 mean_n_stops=("n_stops", "mean")).sort_index().reset_index())
prof.round(4).to_csv("outputs/hub_profile.csv", index=False)

cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(prof[cols])
D = squareform(pdist(Z))
iu = np.triu_indices(len(Z), k=1)

sep = []
labels_by_k = {}
for k in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
    labels_by_k[k] = lab
    same = (lab[:, None] == lab[None, :])[iu]
    d = D[iu]
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": k, "across_over_within": round(d[~same].mean() / d[same].mean(), 4),
                "cluster_sizes": "-".join(str(s) for s in sizes)})
pd.DataFrame(sep).to_csv("outputs/part_d_separation.csv", index=False)

lab = labels_by_k[3]
prof["cluster"] = lab
order = prof.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
prof["segment"] = prof["cluster"].map(names)

seg = prof[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

sp = (prof.groupby("segment").agg(hubs=("hub_id", "size"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                                  delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                                  mean_n_stops=("mean_n_stops", "mean")).reindex(["Standard", "Watch", "Escalate"]).reset_index())
sp.round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
seg[seg["segment"] != seg["hub_tier"].map(implied)].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
