
import numpy as np
import networkx as nx
import ot
import scipy
import torch
from sklearn.metrics.pairwise import euclidean_distances
# from sinkhorn import sinkhorn,sinkhorn_epsilon_scaling,sinkhorn_knopp,sinkhorn_stabilized
#you can also use ot.sinhorn
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment
from scipy.optimize import quadratic_assignment

def perm2mat(p):
    n = np.max(p.shape)
    P = np.zeros((n,n))
    for i in range(n):
        P[i, p[i]] = 1
    return P
def convertToPermHungarian(M, n1, n2):

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)
    P = np.zeros((n2, n1))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, row_ind,col_ind
def eucledian_dist(F1, F2, n):
    D = euclidean_distances(F1, F2)
    return D
def convertToPermHungarian2new(row_ind, col_ind, n, m):
    P = torch.zeros((n,m), dtype = torch.float64)
    #A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    ans = []
    n = max(len(row_ind), len(col_ind))
    for i in range(n):
        #P[row_ind[i]][col_ind[i]] = 1
        #if (row_ind[i] >= n) or (col_ind[i] >= m):
        #    continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans


def feature_extraction1(G,simple = True):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""
    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 2))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]
    node_features[:, 0] = degs
    node_features = np.nan_to_num(node_features)
    egonets = {n: nx.ego_graph(G, n) for n in node_list}
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]
    node_features[:, 1] = neighbor_degs
    return np.nan_to_num(node_features)



def feature_extraction(G,simple):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # number of edges in the neighborhood

    if simple==False:
        neighbor_edges = [
            egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
            for n in node_list
        ]

    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    # node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    if simple==False:
        neighbor_outgoing_edges = [
            len(
                [
                    edge
                    for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                    if not egonets[i].has_edge(*edge)
                ]
            )
            for i in node_list
        ]   

    # number of neighbors of neighbors (not in neighborhood)
    if simple==False:
        neighbors_of_neighbors = [
            len(
                set([p for m in G.neighbors(n) for p in G.neighbors(m)])
                - set(G.neighbors(n))
                - set([n])
            )
            if node_degree_dict[n] > 0
            else 0
            for n in node_list
        ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts
    if (simple==False):
        node_features[:, 4] = neighbor_edges #create if statement
        node_features[:, 5] = neighbor_outgoing_edges#
        node_features[:, 6] = neighbors_of_neighbors#

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)
def relaxed_normAPPB_FW_seeds(A, B, max_iter=1000, seeds=0, verbose=False):
    AtA = np.dot(A.T, A)
    BBt = np.dot(B, B.T)
    p = A.shape[0]
    
    def f1(P):
        return np.linalg.norm(np.dot(A, P) - np.dot(P, B), ord='fro') ** 2
    
    tol = 5e-2
    tol2 = 1e-4
    
    P = np.ones((p, p)) / (p - seeds)
    P[:seeds, :seeds] = np.eye(seeds)
    
    f = f1(P)
    var = 1
    s = 0

    while not (np.abs(f) < tol) and (var > tol2) and (s<max_iter):
        fold = f
        
        grad = 2*(np.dot(AtA, P) - np.dot(np.dot(A.T, P), B) - np.dot(np.dot(A, P), B.T) + np.dot(P, BBt))
        
        grad[:seeds, :] = 0
        grad[:, :seeds] = 0
        
        #G = np.round(grad)
        
        row_ind, col_ind = linear_sum_assignment(grad[seeds:, seeds:])
        
        Ps = perm2mat(col_ind)
        Ps[:seeds, :seeds] = np.eye(seeds) 
        
        C = np.dot(A, P - Ps) + np.dot(Ps - P, B)
        D = np.dot(A, Ps) - np.dot(Ps, B)
        
        aq = np.trace(np.dot(C, C.T))
        bq = np.trace(np.dot(C, D.T) + np.dot(D, C.T))
        aopt = -bq / (2 * aq)
        
        Ps4 = aopt * P + (1 - aopt) * Ps
        
        f = f1(Ps4)
        P = Ps4
        
        var = np.abs(f - fold)
        s += 1
    
    #_, col_ind = linear_sum_assignment(-P.T)
    
    return P

def convex_init(A, B, D, mu, niter):
    #np.set_printoptions(suppress=True)
    n = len(A)
    P = torch.ones((n,n), dtype = torch.float64)
    P=P/n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D

    avg_degree_A = (A.sum(dim=1)).mean()
    avg_degree_B = (B.sum(dim=1)).mean()
    if (min(avg_degree_A, avg_degree_B) < 3):
        qap_weightage = 2
    else:
        qap_weightage = 1

    for i in range(niter):
        for it in range(1, 11):
            G=((-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T))*qap_weightage) + K+ i*(mat_ones - 2*P)
            q= ot.sinkhorn(ones, ones, G, reg,numItermax=1500)
            #q = sinkhorn(ones, ones, G, reg,method='sinkhorn', maxIter = 1500, stopThr = 1e-5)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P

def convex_init1(A, B, D, mu, niter, P0=None):
    #np.set_printoptions(suppress=True)
    n = len(A)
    if P0 is None:
        P0 = relaxed_normAPPB_FW_seeds(A, B)
    P = torch.from_numpy(P0).double()
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D

    avg_degree_A = (A.sum(dim=1)).mean()
    avg_degree_B = (B.sum(dim=1)).mean()
    if (min(avg_degree_A, avg_degree_B) < 3):
        qap_weightage = 2
    else:
        qap_weightage = 1


    for i in range(niter):
        for it in range(1, 11):
            G=((-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)) * qap_weightage) + K+ i*(mat_ones - 2*P)
            #q = sinkhorn(ones, ones, G, reg,method='sinkhorn', maxIter = 1500, stopThr = 1e-5)
            q=ot.sinkhorn(ones, ones, G, reg,method='sinkhorn',numItermax=1500)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P


def Alpine_pp_new(A,B, K, niter,A1,weight=1):
    m = len(A)
    n = len(B)
    I_p = torch.zeros((m,m+1),dtype = torch.float64)
    for i in range(m):
        I_p[i,i] = 1
    Pi=torch.ones((m+1,n),dtype = torch.float64)
    Pi[:-1,:] *= 1/n
    Pi[-1,:] *= (n-m)/n
    reg = 1.0
    mat_ones = torch.ones((m+1, n), dtype = torch.float64)
    ones_ = torch.ones(n, dtype = torch.float64)
    ones_augm_ = torch.ones(m+1, dtype = torch.float64)
    ones_augm_[-1] = n-m

    for i in range(10):
        for it in range(1, 11):
            deriv=(-4*I_p.T@(A-I_p@Pi@B@Pi.T@I_p.T)@I_p@Pi@B)+i*(mat_ones - 2*Pi)+K
            #q=sinkhorn(ones_augm_, ones_, deriv, reg,method="sinkhorn",maxIter = 500, stopThr = 1e-5) 
            q=ot.sinkhorn(ones_augm_, ones_, deriv, reg,method='sinkhorn',numItermax=1500)

            alpha = (2 / float(2 + it) )    
            Pi[:m,:n] = Pi[:m,:n] + alpha * (q[:m,:n] - Pi[:m,:n])
    Pi=Pi[:-1]
    P2,row_ind,col_ind = convertToPermHungarian(Pi, n, m)
    forbnorm = LA.norm(A - I_p[:,:m].T@P2@B@P2.T@I_p[:,:m], 'fro')**2
    return Pi, forbnorm,row_ind,col_ind


def Fugal(Src,Tar ,iter,simple,mu,EFN=5):
    print("Fugal")
    torch.set_num_threads(40)
    dtype = np.float64
    for i in range(Src.shape[0]):
        row_sum1 = np.sum(Src[i, :])
    # If the sum of the row is zero, add a self-loop
        if row_sum1 == 0:
            Src[i, i] = 1
    for i in range(Tar.shape[0]):
        row_sum = np.sum(Tar[i, :])
    # If the sum of the row is zero, add a self-loop
        if row_sum == 0:
            Tar[i, i] = 1
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)
    A = torch.tensor((Src), dtype = torch.float64)
    B = torch.tensor((Tar), dtype = torch.float64)
    simple=True
    F1 = feature_extraction(Src1,simple)
    F2 = feature_extraction(Tar1,simple)
    D = eucledian_dist(F1, F2, n)
    D = torch.tensor(D, dtype = torch.float64)
    #just see Fugal initialization
    if (n< 370):
        mu=0.5
    elif (n<400):
        mu=1
    elif (n<700):
        mu=0.1
    elif (n<1165):
        mu=0.5
    elif (n<1700):
        mu=2
    else:
        mu=1
    P = convex_init(A, B, D, mu, iter)
    return P

def Fugal_init(Src,Tar, iter,simple,mu,EFN=5,P0=None):
    print("FugalGrad")
    torch.set_num_threads(40)
    dtype = np.float64
    for i in range(Src.shape[0]):
        row_sum1 = np.sum(Src[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum1 == 0:
            Src[i, i] = 1
    for i in range(Tar.shape[0]):
        row_sum = np.sum(Tar[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum == 0:
            Tar[i, i] = 1
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)
    A = torch.tensor((Src), dtype = torch.float64)
    B = torch.tensor((Tar), dtype = torch.float64)
    simple=True
    #
    #
    
    #EFN 5 equals fugal
    if (EFN==5):
        F1 = feature_extraction(Src1,simple)
        F2 = feature_extraction(Tar1,simple)
    D = eucledian_dist(F1, F2, n)
    D = torch.tensor(D, dtype = torch.float64)
    if (n< 370):
        mu=0.5
    elif (n<400):
        mu=1
    elif (n<700):
        mu=0.1
    elif (n<1165):
        mu=0.5
    elif (n<1700):
        mu=2
    else:
        mu=1
    P=convex_init1(A, B, D, mu, iter, P0=P0)
    return P


def Alpine(Gq, Gt, mu=1, niter=10, weight=2):
    n1 = Gq.number_of_nodes()
    n2 = Gt.number_of_nodes()
    n = max(n1, n2)
    for node in nx.isolates(Gq):
        Gq.add_edge(node, node)
    for node in nx.isolates(Gt):
        Gt.add_edge(node, node)
        
    Gq.add_node(n1)
    Gq.add_edge(n1,n1)
    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    #weight=1
    if (weight==2):
        F1 = feature_extraction1(Gq)
        F2 = feature_extraction1(Gt) 
    else:
        F1 = feature_extraction(Gq)
        F2 = feature_extraction(Gt)
    D = eucledian_dist(F1,F2,n)
    D = torch.tensor(D, dtype = torch.float64)
    P, forbnorm,row_ind,col_ind = Alpine_pp_new(A[:n1,:n1], B, mu*D, niter,A)
    #_, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    #list_of_nodes = []
    #for el in ans: list_of_nodes.append(el[1])
    return P#,ans, list_of_nodes, forbnorm    


def QAP_init(Src,Tar,P0=None):
    print("QAP")
    torch.set_num_threads(40)
    dtype = np.float64
    dtype = np.float64
    for i in range(Src.shape[0]):
        row_sum1 = np.sum(Src[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum1 == 0:
            Src[i, i] = 1
    for i in range(Tar.shape[0]):
        row_sum = np.sum(Tar[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum == 0:
            Tar[i, i] = 1
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)
    A = torch.tensor((Src), dtype = torch.float64)
    B = torch.tensor((Tar), dtype = torch.float64)

    if P0 is None:
        P0 = relaxed_normAPPB_FW_seeds(Src,Tar,max_iter=1000)
    res_qap = quadratic_assignment(-Tar,Src,method='faq',options={"P0": P0.T, "maxiter": 30})
    #res_qap = quadratic_assignment(-Tar,Src,method='faq',options={"maxiter": 30})

    perm = res_qap.col_ind
    P_perm = np.zeros((len(perm), len(perm)))
    P_perm[perm, np.arange(len(perm))] = 1
    return P_perm
def QAP(Src,Tar):
    print("QAP")
    n = len(Src)
    P = torch.ones((n,n), dtype = torch.float64)
    P=P/n
    torch.set_num_threads(40)
    dtype = np.float64
    dtype = np.float64
    for i in range(Src.shape[0]):
        row_sum1 = np.sum(Src[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum1 == 0:
            Src[i, i] = 1
    for i in range(Tar.shape[0]):
        row_sum = np.sum(Tar[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum == 0:
            Tar[i, i] = 1
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)
    A = torch.tensor((Src), dtype = torch.float64)
    B = torch.tensor((Tar), dtype = torch.float64)
    res_qap = quadratic_assignment(-Tar,Src,method='faq',options={"P0": P, "maxiter": 30})
    #res_qap = quadratic_assignment(-Tar,Src,method='faq',options={"maxiter": 30})

    perm = res_qap.col_ind
    P_perm = np.zeros((len(perm), len(perm)))
    P_perm[perm, np.arange(len(perm))] = 1
    return P_perm