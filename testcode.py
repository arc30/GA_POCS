
#recomendations
# please create a proper test scenarion-> 
#generate the datasets prior with the noise so you dont need to rerun the same experimetns
#Nice idea is to precompute the convex matrix P and then pass it to algorithms,
#the reason is that is high complexity.


import networkx as nx
from GA_Archana import *
from noise import *
#netscience 379
#inf-euroroad 1174
#highschool 327
#voles 712
#multimanga 1004
with open("results.txt", "w") as f:
    "hi everybody, we have Fugal, Alpine and QAP"
#hints-> for sparce datasets we have to increase the QAP value (ex -torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T) *2)                    
iter=10
simple=True
mu=1
EFN=5
G1 = read_real_graph(n = 379, name_ = f'datasets/netscience.txt')
A1 = nx.to_numpy_array(G1, dtype=int)
A1N,A2N,GT=generate_graphs(np.array(np.transpose(np.nonzero(np.triu(A1, 1)))),0,0.2)
A1=edges_to_adj(A1N)
A2=edges_to_adj(A2N)
print("Fugal")
P=Fugal(A1,A2,iter,simple,mu,EFN)
row_ind, col_ind = scipy.optimize.linear_sum_assignment(P, maximize=True)
X1=max(eval_align(row_ind,col_ind,GT[0]),eval_align(row_ind,col_ind,GT[1]),
eval_align(col_ind,row_ind,GT[0]),eval_align(col_ind,row_ind,GT[1]))
print(X1)

G1 = nx.Graph(A1)
G2 = nx.Graph(A2)
#P1=Alpine(G1,G2,mu,iter,2)
#print("h2")
#row_ind, col_ind = scipy.optimize.linear_sum_assignment(P1, maximize=True)
#X2=max(eval_align(row_ind,col_ind,GT[0]),eval_align(row_ind,col_ind,GT[1]),
#eval_align(col_ind,row_ind,GT[0]),eval_align(col_ind,row_ind,GT[1]))
#print(X2)

P2=Fugal_init(A1,A2,iter,simple,mu,EFN)
row_ind, col_ind = scipy.optimize.linear_sum_assignment(P2, maximize=True)
X3=max(eval_align(row_ind,col_ind,GT[0]),eval_align(row_ind,col_ind,GT[1]),
eval_align(col_ind,row_ind,GT[0]),eval_align(col_ind,row_ind,GT[1]))
print(X3)

print("QAP")
P3=QAP(A1,A2)
row_ind, col_ind = scipy.optimize.linear_sum_assignment(P3, maximize=True)
X4=max(eval_align(row_ind,col_ind,GT[0]),eval_align(row_ind,col_ind,GT[1]),
eval_align(col_ind,row_ind,GT[0]),eval_align(col_ind,row_ind,GT[1]))
print(X4)

print("QAP_init")

P4=QAP_init(A1,A2)
row_ind, col_ind = scipy.optimize.linear_sum_assignment(P4, maximize=True)
X5=max(eval_align(row_ind,col_ind,GT[0]),eval_align(row_ind,col_ind,GT[1]),
eval_align(col_ind,row_ind,GT[0]),eval_align(col_ind,row_ind,GT[1]))
print(X5)
#with open("results.txt", "w") as f:
#    f.write(f'{X1} {X2} {X3} {X4} {X5}\n')
with open("results.txt", "w") as f:
    f.write(f' {X5} \n')
