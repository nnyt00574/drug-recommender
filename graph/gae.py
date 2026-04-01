import torch
import torch.nn as nn
import numpy as np

class GraphAutoEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),#input dimernsions
            nn.ReLU(), #removes negative values 
            nn.Linear(64, 32)#reducing dimernsions to 32
        )

    def forward(self, X):
        Z = self.encoder(X)#passes x through input to get z output
        A_pred = torch.sigmoid(Z @ Z.T)#decoder stage performs self multiplication with its own transpose, returns probability that edge exists
        return A_pred, Z


def train_gae(G, embeddings, epochs=50):
    nodes = list(G.nodes())#gets a list of all nodes in the grapf
    idx = {n: i for i, n in enumerate(nodes)}#creates mapping dictionary

    A = torch.zeros(len(nodes), len(nodes))#Initializes a square matrix of zeros (the Adjacency Matrix).

    for u, v in G.edges():
        A[idx[u], idx[v]] = 1
        A[idx[v], idx[u]] = 1

    X = torch.FloatTensor(np.array([embeddings[n] for n in nodes]))

    model = GraphAutoEncoder(X.shape[1])#converts node2Vec to tensor 
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        A_pred, Z = model(X)
        loss = ((A_pred - A) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return {nodes[i]: Z[i].detach().numpy() for i in range(len(nodes))}