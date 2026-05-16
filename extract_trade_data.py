import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# UN Comtrade v1 endpoint. Use /data with an API key for descriptions and
# higher limits, or /public/preview without a key for a small public preview.
API_KEY = os.getenv("COMTRADE_API_KEY", "").strip()
BASE_URL = (
    "https://comtradeapi.un.org/data/v1/get/C/A/HS"
    if API_KEY
    else "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
)

YEARS = list(range(2018, 2025))

COUNTRIES = {
    "China": "156",
    "USA": "842",
    "Japan": "392",
    "Korea": "410",
}

HS_CODES = [
    "854231",
    "854232",
    "854239",
]

FLOWS = {
    "Imports": "M",
    "Exports": "X",
}

OUTPUT_FILE = "semiconductor_trade_data.csv"


def make_session():
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_trade_data():
    session = make_session()
    all_data = []
    commodity_codes = ",".join(HS_CODES)

    total_requests = len(COUNTRIES) * len(YEARS) * len(FLOWS)
    progress = tqdm(total=total_requests, desc="Fetching UN Comtrade data")

    for country_name, reporter_code in COUNTRIES.items():
        for year in YEARS:
            for flow_name, flow_code in FLOWS.items():
                params = {
                    "reporterCode": reporter_code,
                    "partnerCode": "0",
                    "flowCode": flow_code,
                    "cmdCode": commodity_codes,
                    "period": str(year),
                    "format": "JSON",
                    "maxRecords": 50000,
                    "includeDesc": "true",
                    "breakdownMode": "classic",
                }

                headers = {}
                if API_KEY:
                    headers["Ocp-Apim-Subscription-Key"] = API_KEY

                try:
                    response = session.get(
                        BASE_URL,
                        params=params,
                        headers=headers,
                        timeout=30,
                    )
                    response.raise_for_status()
                    payload = response.json()
                except requests.RequestException as exc:
                    progress.write(
                        f"Failed {country_name} | {year} | {flow_name}: {exc}"
                    )
                    progress.update(1)
                    continue

                records = payload.get("data", [])
                if not records:
                    progress.write(f"No data for {country_name} | {year} | {flow_name}")

                for row in records:
                    all_data.append(
                        {
                            "year": row.get("period") or row.get("refYear"),
                            "reporter": row.get("reporterDesc") or country_name,
                            "reporter_code": row.get("reporterCode"),
                            "partner": row.get("partnerDesc") or "World",
                            "partner_code": row.get("partnerCode"),
                            "flow": row.get("flowDesc") or flow_name,
                            "flow_code": row.get("flowCode"),
                            "hs_code": row.get("cmdCode"),
                            "commodity": row.get("cmdDesc"),
                            "trade_value_usd": row.get("primaryValue"),
                            "net_weight_kg": row.get("netWgt"),
                        }
                    )

                progress.update(1)
                time.sleep(1)

    progress.close()
    return pd.DataFrame(all_data)


if __name__ == "__main__":
    df = fetch_trade_data()
    print("\nDataset Shape:")
    print(df.shape)
    print("\nSample Data:")
    print(df.head())
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
