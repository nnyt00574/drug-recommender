import networkx as nx

def build_graph(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(row["drug_a"], row["drug_b"])#construct edges between drugs

    return G