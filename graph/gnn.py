import torch
import torch.nn as nn
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=32):#2 layers have 64 output 32
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, A, X):
        H = torch.matmul(A, X)#convolution step
        H = torch.relu(self.fc1(H))#weighted data of 1st to average ReLu
        H = torch.matmul(A, H)#second roud of message passing to learn from second order connections
        return self.fc2(H)

def train_gnn(G, embeddings, epochs=50):
    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}

    A = np.zeros((len(nodes),len(nodes)))
    for u,v in G.edges():
        A[idx[u],idx[v]] = 1
        A[idx[v],idx[u]] = 1

    A = A + np.eye(len(A))
    D = np.diag(1/np.sqrt(A.sum(axis=1)))
    A = torch.FloatTensor(D @ A @ D)

    X = torch.FloatTensor(np.array([embeddings[n] for n in nodes]))

    model = GCN(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        H = model(A,X)
        loss = ((H - X)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    return {nodes[i]: H[i].detach().numpy() for i in range(len(nodes))}