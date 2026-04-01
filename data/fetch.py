import requests, time, logging

log = logging.getLogger(__name__)

def fetch_data(max_records):
    records, skip = [], 0

    while len(records) < max_records:
        try:
            url = f"https://api.fda.gov/drug/event.json?limit=100&skip={skip}"
            res = requests.get(url, timeout=10).json()

            if "results" not in res:
                break

            records.extend(res["results"])
            skip += 100
            time.sleep(0.3)#to bypass api security mechanisms 

        except:
            break

    if len(records) < 200:
        raise RuntimeError("Insufficient data")

    return records[:max_records]