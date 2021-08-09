import numpy as np


def get_hop_distance(num_node, edge, max_hop=1):
    # link matrix
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


class GraphWithPartition:  # Unidirected, connections with hop limit
    """The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints.
        - ntu-rgb+d: Is consists of 25 joints.
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(
        self,
        num_nodes,
        center,
        inward_edges,
        strategy="distance",
        max_hop=1,
        dilation=1,
    ):
        self.num_nodes = num_nodes
        self.center = center
        self.self_edges = [[i, i] for i in range(self.num_nodes)]
        self.inward_edges = inward_edges
        self.edges = self.self_edges + self.inward_edges

        self.max_hop = max_hop
        self.dilation = dilation

        self.hop_dis = get_hop_distance(self.num_nodes, self.edges, max_hop=max_hop)
        self.get_adjacency(strategy)

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_nodes, self.num_nodes))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_nodes, self.num_nodes))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_nodes, self.num_nodes))
                a_close = np.zeros((self.num_nodes, self.num_nodes))
                a_further = np.zeros((self.num_nodes, self.num_nodes))
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError()


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class SpatialGraph:
    def __init__(self, num_nodes, inward_edges, strategy="spatial"):

        self.num_nodes = num_nodes
        self.strategy = strategy
        self.self_edges = [(i, i) for i in range(num_nodes)]
        self.inward_edges = inward_edges
        self.outward_edges = [(j, i) for (i, j) in self.inward_edges]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        if self.strategy == "spatial":
            return get_spatial_graph(
                self.num_nodes, self.self_edges, self.inward_edges, self.outward_edges
            )
        else:
            raise ValueError()
