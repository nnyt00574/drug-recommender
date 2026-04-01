import numpy as np

def build_features(df, emb, gnn_emb, gae_emb):
    X = []
    y = np.stack(df["reaction_vec"].values)

    for _, r in df.iterrows():
        ea = emb.get(r["drug_a"], np.zeros(32))
        eb = emb.get(r["drug_b"], np.zeros(32))

        ga = gnn_emb.get(r["drug_a"], np.zeros(32))
        gb = gnn_emb.get(r["drug_b"], np.zeros(32))

        za = gae_emb.get(r["drug_a"], np.zeros(32))
        zb = gae_emb.get(r["drug_b"], np.zeros(32))

        ea_full = np.concatenate([ea, ga, za])
        eb_full = np.concatenate([eb, gb, zb])

        X.append(np.concatenate([
            [r["severity"]],
            ea_full * eb_full,
            np.abs(ea_full - eb_full)
        ]))

    return np.array(X), y