from node2vec import Node2Vec

def compute_embeddings(G, dim):
    
    n2v = Node2Vec(G, dimensions=dim, walk_length=15, num_walks=50, workers=1)
    model = n2v.fit(window=5)#if they appear in a 5 step window they are in the same neighbourhood 
    return {n: model.wv[n] for n in G.nodes()}