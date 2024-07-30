import torch

def edge_index_to_adjacency(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj

# Verilen edge_index tensoru
edge_index = torch.tensor([
    [0,  1,  1,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  6,  6,  6,  7,  7,
     7,  8,  9,  9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 15, 15, 16,
     17, 17, 17, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21, 22, 22, 23, 24, 24,
     24, 25, 26, 26, 26, 27, 28, 28, 28, 29, 30, 30, 30, 31, 32, 32, 32, 33,
     34, 34, 34, 35, 36, 36, 36, 37],
    [1,  0,  2,  1,  3, 36,  2,  4,  3,  5, 32,  4,  6,  5,  7, 30,  6,  8,
     9,  7,  7, 10,  9, 11, 17, 10, 12, 11, 13, 15, 12, 14, 13, 12, 16, 15,
     10, 18, 30, 17, 19, 18, 20, 28, 19, 21, 20, 22, 24, 21, 23, 22, 21, 25,
     26, 24, 24, 27, 28, 26, 19, 26, 29, 28,  6, 17, 31, 30,  4, 33, 34, 32,
     32, 35, 36, 34,  2, 34, 37, 36]
])

# Grafın toplam düğüm sayısını belirleme
num_nodes = torch.max(edge_index) + 1

# Adjacency matrixı oluşturma
adj_matrix = edge_index_to_adjacency(edge_index, num_nodes)
print(adj_matrix)

print(adj_matrix.shape)