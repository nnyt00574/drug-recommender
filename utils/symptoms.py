from collections import defaultdict

def build_symptoms(records):
    m = defaultdict(list)

    for r in records:
        try:
            drugs=[d.get("medicinalproduct","").upper()
                   for d in r.get("patient",{}).get("drug",[])]

            reactions=[rx.get("reactionmeddrapt","").lower()
                       for rx in r.get("patient",{}).get("reaction",[])]

            for s in reactions:
                m[s].extend(drugs)
        except:
            continue

    out={}
    for s,d in m.items():
        freq={}
        for x in d:
            freq[x]=freq.get(x,0)+1
        out[s]=sorted(freq,key=freq.get,reverse=True)[:5]

    return out