import numpy as np
import networkx as nx
def refill_e(edges, n, amount):
    if amount == 0:
        return edges
    # print(edges)
    # ee = np.sort(edges).tolist()
    ee = {tuple(row) for row in np.sort(edges).tolist()}
    new_e = []
    check = 0
    while len(new_e) < amount:
        _e = np.random.randint(n, size=2)
        # _ee = np.sort(_e).tolist()
        _ee = tuple(np.sort(_e).tolist())
        check += 1
        if not(_ee in ee) and _e[0] != _e[1]:
            # ee.append(_ee)
            ee.add(_ee)
            new_e.append(_e)
            check = 0
            # print(f"refill - {len(new_e)}/{amount}")
        if check % 1000 == 999:
            print(f"refill - {check + 1} times in a row fail")
    # print(new_e)
    return np.append(edges, new_e, axis=0)


def remove_e(edges, noise, no_disc=True, until_connected=False):
    ii = 0
    while True:
        ii += 1
        print(f"##<{ii}>##")

        if no_disc:
            bin_count = np.bincount(edges.flatten())
            rows_to_delete = []
            for i, edge in enumerate(edges):
                if np.random.sample(1)[0] < noise:
                    e, f = edge
                    if bin_count[e] > 1 and bin_count[f] > 1:
                        bin_count[e] -= 1
                        bin_count[f] -= 1
                        rows_to_delete.append(i)
            new_edges = np.delete(edges, rows_to_delete, axis=0)
        else:
            new_edges = edges[np.random.sample(edges.shape[0]) >= noise]

        graph = nx.Graph(new_edges.tolist())
        graph_cc = len(max(nx.connected_components(graph), key=len))
        print(graph_cc, np.amax(edges)+1)
        graph_connected = graph_cc == np.amax(edges) + 1
        # if not graph_connected:
        #     break
        if graph_connected or not until_connected:
            break
    return new_edges


def load_as_nx(path):
    G_e = np.loadtxt(path, int)
    G = nx.Graph(G_e.tolist())
    print("Just checking",nx.is_directed(G))
    return np.array(G.edges)


def loadnx(path):
    G_e = np.loadtxt(path, int)
    return nx.Graph(G_e.tolist())



def generate_graphs(G, source_noise=0.00, target_noise=0.00, refill=False):

    Src_e=G
    n = np.amax(Src_e) + 1
    nedges = Src_e.shape[0]

    gt_e = np.array((
        np.arange(n),
        np.random.permutation(n)
    ))

    Gt = (
        gt_e[:, gt_e[1].argsort()][0],
        gt_e[:, gt_e[0].argsort()][1]
    )

    Tar_e = Gt[0][Src_e]

    Src_e = remove_e(Src_e, source_noise)
    Tar_e = remove_e(Tar_e, target_noise)

    if refill:
        Src_e = refill_e(Src_e, n, nedges - Src_e.shape[0])
        Tar_e = refill_e(Tar_e, n, nedges - Tar_e.shape[0])

    return Src_e, Tar_e,  Gt

def edges_to_adj(edges, n=None):
    if n is None:
        n = int(edges.max()) + 1
    G = nx.from_edgelist(edges)
    return nx.to_numpy_array(G, nodelist=range(n), dtype=int)
def load_as_nx(path):
    G_e = np.loadtxt(path, int)
    G = nx.Graph(G_e.tolist())
    #print("Just checking",nx.is_directed(G))
    return np.array(G.edges)
def read_real_graph(n, name_):
    print(f'Making {name_} graph...')
    filename = open(f'{name_}', 'r')
    lines = filename.readlines()
    G = nx.Graph()
    for i in range(n): G.add_node(i)
    for line in lines:
        u_v = (line[:-1].split(' '))
        u = int(u_v[0])
        v = int(u_v[1])
        G.add_edge(u, v)
    print(f'Done {name_} ...')
    return G       


def eval_align(ma, mb, gmb):

    try:
        gmab = np.arange(gmb.size)
        gmab[ma] = mb
        gacc = np.mean(gmb == gmab)

        mab = gmb[ma]
        acc = np.mean(mb == mab)

    except Exception:
        mab = np.zeros(mb.size, int) - 1
        gacc = acc = -1.0
    alignment = np.array([ma, mb, mab]).T
    alignment = alignment[alignment[:, 0].argsort()]
    return gacc
    #return gacc, acc, alignment