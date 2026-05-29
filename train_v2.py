"""Train V2 RandomForestClassifier and save model_v2.pkl

This script reproduces the essential steps from `V2_Model copy.ipynb`.
It reads `from_V1_supply_chain_data.csv`, performs feature construction, trains
a RandomForestClassifier, and saves a joblib file with the model and metadata.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib


def main():
    df = pd.read_csv("from_V1_supply_chain_data.csv")

    # Compute exports totals if necessary
    if "total_exports" not in df.columns:
        total_exports = (
            df[df["flow"].eq("Export")]
            .groupby(["year", "reporter", "hs_code"])["trade_value_usd"]
            .sum()
            .reset_index()
            .rename(columns={"trade_value_usd": "total_exports"})
        )
        df = df.merge(total_exports, on=["year", "reporter", "hs_code"], how="left")

    df["export_concentration_ratio"] = np.where(
        df["total_exports"].gt(0), df["trade_value_usd"] / df["total_exports"], np.nan
    )

    # Alternative supplier count
    supplier_counts = (
        df.groupby(["year", "reporter", "hs_code"])["partner"].nunique().reset_index()
        .rename(columns={"partner": "alternative_supplier_count"})
    )
    df = df.merge(supplier_counts, on=["year", "reporter", "hs_code"], how="left")

    # Network metrics (simple degree centrality on directed graph)
    try:
        import networkx as nx
        G = nx.from_pandas_edgelist(df, source="reporter", target="partner", edge_attr="trade_value_usd", create_using=nx.DiGraph())
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        df["network_centrality"] = df["reporter"].map(centrality)
        df["bottleneck_score"] = df["reporter"].map(betweenness)
    except Exception:
        df["network_centrality"] = np.nan
        df["bottleneck_score"] = np.nan

    # Impute numeric missing values
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

    features = [
        "trade_value_usd",
        "net_weight_kg",
        "lag_trade_1y",
        "rolling_avg_3y",
        "volatility_index",
        "supplier_dependency_ratio",
        "export_concentration_ratio",
        "alternative_supplier_count",
        "network_centrality",
        "bottleneck_score",
    ]

    target = "disruption_label"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    export_obj = {"model": model, "features": features, "imputer": imputer}
    joblib.dump(export_obj, "model_v2.pkl")
    print("Saved model_v2.pkl")


if __name__ == "__main__":
    main()
