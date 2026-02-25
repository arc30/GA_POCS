import torch
import math
from scipy.optimize import linear_sum_assignment
import numpy as np


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def FNorm2(X):
    return torch.trace(X.T @ X).item()
def edge_cost():
    return 2 

def true_dist(A, B, D, P):
    k = edge_cost()
    return (
        k * FNorm2(A @ P - P @ B) / 2
        + torch.trace(P.T @ D).item()
    )


def dist(A, B, D, P, mu):
    k = edge_cost()
    return (
        k * FNorm2(A @ P - P @ B) / 2
        + mu * torch.trace(P.T @ D).item()
    )


# -------------------------------------------------
# Penalty
# -------------------------------------------------
def penalty1(P):
    """
    ||P1-1||^2 + ||P^T1-1||^2 + ||max(0,-P)||^2 + ||max(0,P-1)||^2
    """
    n = P.shape[0]
    one = torch.ones(n, device=P.device)
    term1 = torch.norm(P @ one - one) ** 2
    term2 = torch.norm(P.T @ one - one) ** 2
    term3 = torch.norm(torch.clamp(-P, min=0)) ** 2
    term4 = torch.norm(torch.clamp(P - 1, min=0)) ** 2
    return term1 + term2 + term3 + term4
def penalty(P):
    rows, cols = P.shape
    I = torch.ones((cols, 1), dtype=torch.float32)

    v1 = P @ I - I
    v2 = P.T @ I - I

    p1 = (v1.T @ v1).item() + (v2.T @ v2).item()

    zero = torch.zeros((rows, cols), dtype=torch.float32)
    ones = torch.ones((rows, cols), dtype=torch.float32)

    temp1 = torch.maximum(zero, -P)
    p2 = torch.sum(temp1 * temp1).item()

    temp2 = torch.maximum(zero, P - ones)
    p3 = torch.sum(temp2 * temp2).item()

    return p1 + p2 + p3


# -------------------------------------------------
# Gradients (manual, identical to C++)
# -------------------------------------------------
def dist_gradient1(A, B, D, P, mu):
    """
    Gradient of ||A P - P B||_F^2 + mu * trace(P^T D)
    """
    grad_frob = 2 * (A.T @ (A @ P - P @ B) - (A @ P - P @ B) @ B.T)
    grad_linear = mu * D
    return grad_frob + grad_linear
def dist_gradient(A, B, D, P, mu):
    k = edge_cost()
    return (
         2* k*  (
            A.T @ A @ P
            - A.T @ P @ B
            - A @ P @ B.T
            + P @ B @ B.T
        )
        + mu * D
    )


def penalty_gradient(P):
    rows, cols = P.shape

    I = torch.ones((cols, 1), dtype=torch.float32)

    P1 = (
        2 * (P @ I) @ I.T
        - 4 * (I @ I.T)
        + 2 * (I @ I.T) @ P
    )

    zero = torch.zeros_like(P)
    ones = torch.ones_like(P)

    temp1 = torch.maximum(zero, -P)
    P2 = -2 * temp1

    temp2 = torch.maximum(zero, P - ones)
    P3 = 2 * temp2

    return P1 + P2 + P3


# -------------------------------------------------
# Hungarian projection
# -------------------------------------------------

def convertToPermHung(M, maximize=True):
    rows, cols = M.shape
    P = torch.zeros((rows, cols), dtype=torch.float32)

    M_np = M.detach().cpu().numpy()

    if maximize:
        cost = -M_np
    else:
        cost = M_np

    row_ind, col_ind = linear_sum_assignment(cost)

    for r, c in zip(row_ind, col_ind):
        P[r, c] = 1

    return P


# -------------------------------------------------
# ADAM optimization (manual, matches C++)
# -------------------------------------------------

def adam(A, B, D=None):
    rows = A.shape[0]
    step=0
    alpha = 0.001
    sigma = 5
    lam = 0
    device="cpu"
    eig_A = np.linalg.eigvalsh(A)
    eig_B = np.linalg.eigvalsh(B)
    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)
    ones = torch.ones((rows, rows), dtype=torch.float32)
    if D is None:
        D = torch.zeros((rows, rows), dtype=torch.float32, device=device)
    else:
        D = torch.tensor(D, dtype=torch.float32, device=device)
    P = torch.eye(rows, dtype=torch.float32)
    S = torch.eye(rows, dtype=torch.float32)
    opt_P = P.clone()

    lambda_convex = np.min([(eig_A[i] - eig_B[j])**2 / 2 
                            for i in range(len(eig_A)) for j in range(len(eig_B))])
    mu = 1

    prev_dist = 0
    cur_dist = dist(A, B, D, P, mu) + sigma * penalty(P) + lam * torch.trace(P.T @ (ones - P)).item()

    opt_perm_dist = cur_dist

    b1 = 0.9
    b2 = 0.999

    m = torch.zeros_like(P)
    v = torch.zeros_like(P)

    while True:
        print(step)
        step = 0

        while abs(cur_dist - prev_dist) > 1e-6:
            step += 1

            g1 = dist_gradient(A, B, D, P, mu)
            g2 = penalty_gradient(P)
            g3 = ones - 2 * P

            gradient = g1 + sigma * g2 + lam * g3

            m = b1 * m + (1 - b1) * gradient
            v = b2 * v + (1 - b2) * (gradient * gradient)

            m_hat = m / (1 - b1 ** step)
            v_hat = v / (1 - b2 ** step)

            P = P - (alpha * m_hat) / (torch.sqrt(v_hat) + 1e-8)

            prev_dist = cur_dist
            cur_dist = (
                dist(A, B, D, P, mu)
                + sigma * penalty(P)
                + lam * torch.trace(P.T @ (ones - P)).item()
            )
            cur_dist = (torch.norm(A @ P - P @ B)**2 + mu * torch.trace(P.T @ D) 
                        + sigma * penalty(P) + lam * torch.trace(P.T @ (ones - P)))
            if math.isinf(cur_dist):
                print("inf")
                break

        if math.isinf(cur_dist):
            print("inf")
            break

        hung_perm = convertToPermHung(P, True)
        hung_dist = true_dist(A, B, D, hung_perm)

        if hung_dist < opt_perm_dist:
            opt_perm_dist = hung_dist
            opt_P = S @ hung_perm

        A = hung_perm @ A @ hung_perm.T
        D = hung_perm @ D
        S = S @ hung_perm.T

        sigma *= 2
        prev_dist = 0

        if sigma > 1000:
            break

        lam += 0.5
        # lambda_convex is seen as 0 for more of the datasets. So, what value of lam to be used?
        #
        if lam >= lambda_convex:
           break 
    return opt_P.cpu().numpy().T



# -------------------------------------------------
# Main interface (stochastic_dist)
# -------------------------------------------------

def stochastic_dist(v1_1, v2_1, labels_1,
                    v1_2, v2_2, labels_2):

    n1 = len(labels_1)
    n2 = len(labels_2)
    num_nodes = max(n1, n2)

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    B = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for i, j in zip(v1_1, v2_1):
        A[i, j] = 1

    for i, j in zip(v1_2, v2_2):
        B[i, j] = 1

    D = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i >= n1:
                D[i, j] = node_ins(labels_2[j])
            elif j >= n2:
                D[i, j] = node_del(labels_1[i])
            else:
                D[i, j] = node_sub(labels_1[i], labels_2[j])

    hung_perm = adam(A, B, D)
    hung_dist = true_dist(A, B, D, hung_perm)

    mapping = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if hung_perm[i, j] == 1:
                if i >= n1:
                    mapping.append((None, j))
                elif j >= n2:
                    mapping.append((i, None))
                else:
                    mapping.append((i, j))

    return mapping, hung_dist
