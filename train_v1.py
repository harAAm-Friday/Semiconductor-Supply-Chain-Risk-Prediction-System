"""Train V1 RandomForestRegressor and save model_v1.pkl

This script reproduces the essential steps from `V1_Phase copy.ipynb`.
It reads `2ndTRY_semiconductor_supply_chain.csv`, performs feature engineering,
trains a RandomForestRegressor, and saves a joblib file with the model and metadata.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib


def main():
    df = pd.read_csv("2ndTRY_semiconductor_supply_chain.csv")

    # Basic cleaning
    df = df.drop_duplicates()

    # Ensure numeric columns
    numeric_cols = ["trade_value_usd", "net_weight_kg"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort for group operations
    df = df.sort_values(by=["reporter", "partner", "hs_code", "flow", "year"]) 

    # Lag feature (previous year's trade value)
    df["lag_trade_1y"] = (
        df.groupby(["reporter", "partner", "hs_code", "flow"])["trade_value_usd"].shift(1)
    )

    # YoY growth percentage
    df["yoy_growth_pct"] = ((df["trade_value_usd"] - df["lag_trade_1y"]) / df["lag_trade_1y"]) * 100

    # Rolling average and volatility (3-year window)
    df["rolling_avg_3y"] = (
        df.groupby(["reporter", "partner", "hs_code", "flow"])["trade_value_usd"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    df["volatility_index"] = (
        df.groupby(["reporter", "partner", "hs_code", "flow"])["trade_value_usd"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).std())
    )

    # Supplier dependency ratio
    total_imports = (
        df[df["flow"].eq("Import")]
        .groupby(["year", "reporter", "hs_code"])["trade_value_usd"]
        .sum()
        .reset_index()
        .rename(columns={"trade_value_usd": "total_imports"})
    )
    df = df.merge(total_imports, on=["year", "reporter", "hs_code"], how="left")
    df["supplier_dependency_ratio"] = np.where(
        df["total_imports"].gt(0), df["trade_value_usd"] / df["total_imports"], np.nan
    )

    # Sudden drop flag
    df["sudden_drop_flag"] = (df["yoy_growth_pct"] < -20).astype(int)

    # Future trade value (target)
    df["future_trade_value_usd"] = (
        df.groupby(["reporter", "partner", "hs_code", "flow"])["trade_value_usd"].shift(-1)
    )

    # Drop rows missing required model cols
    required_model_cols = [
        "trade_value_usd",
        "net_weight_kg",
        "lag_trade_1y",
        "yoy_growth_pct",
        "rolling_avg_3y",
        "volatility_index",
        "supplier_dependency_ratio",
        "sudden_drop_flag",
        "future_trade_value_usd",
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_model_cols)

    features = [
        "trade_value_usd",
        "net_weight_kg",
        "lag_trade_1y",
        "yoy_growth_pct",
        "rolling_avg_3y",
        "volatility_index",
        "supplier_dependency_ratio",
        "sudden_drop_flag",
    ]
    target = "future_trade_value_usd"

    X = df[features]
    y = df[target]

    # Simple imputation for numeric columns
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)

    # Train/test split and training
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    export_obj = {
        "model": model,
        "features": features,
        "imputer": imputer,
    }

    joblib.dump(export_obj, "model_v1.pkl")
    print("Saved model_v1.pkl")


if __name__ == "__main__":
    main()
