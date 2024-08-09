# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:07:07 2024

@author: liitl
"""

import numpy as np 
import cvxpy as cp
from time import time

# pseudo
np.random.seed(0)

m, n = 80, 40
k = 4
rate = -1
L = 2 # lambda in Algorithm
iter = 2000

m1 = np.random.rand(m, k)
m2 = np.random.rand(k, n)
origin = np.matmul(m1, m2)
mask = np.random.rand(m, n) > rate
observed = mask*origin # 点积
 
guess = 40
m3 = np.random.rand(m, guess)
m4 = np.random.rand(guess, n)
X = np.matmul(m3, m4)
print(f'问题规模{m}×{n},初始loss值{np.sum(np.square(mask*X - mask*origin))}')



def f_grad(X):
    return mask*X - observed

def reduce_SVD(U, sigma, VT, L=0):
    if L != 0:
        sigma = sigma - L
    else:
        sigma = np.around(sigma, 10)
    sigma = np.array([x for x in sigma if x>0])
    U = U[:, 0:len(sigma)]
    VT = VT[0:len(sigma), :]
    return U, sigma, VT

start = time()
U, sigma, VT = np.linalg.svd(X, full_matrices=False)
U, sigma, VT = reduce_SVD(U, sigma, VT)
for i in range(iter):    
    U_h, sigma_h, VT_h = np.linalg.svd(X - f_grad(X), full_matrices=False)
    U_G, sigma_G, VT_G = reduce_SVD(U, sigma, VT, L)
    
    U_qr = np.concatenate((U_G, U),axis=1)
    Q_U, R_U= np.linalg.qr(U_qr)
    
    VT_qr = np.concatenate((VT_G, VT),axis=0)
    Q_V, R_V = np.linalg.qr(VT_qr.T) # 注意将VT转置为V
    
    U_A = Q_U
    VT_A = Q_V.T
    
    S = cp.Variable((U_A.shape[1], VT_A.shape[0]))
    USA = Q_U@S@(Q_V.T)
    objective = cp.Minimize(0.5*cp.sum_squares(cp.multiply(mask, USA)-observed)
                            + L*cp.normNuc(X))
    problem = cp.Problem(objective)
    problem.solve()
    result_S = S.value
    
    U_S, sigma_S, VT_S = np.linalg.svd(result_S, full_matrices=False)
    U_S, sigma_S, VT_S = reduce_SVD(U_S, sigma_S, VT_S)
    U = U_A@U_S
    V = (VT_S@VT_A).T # 矩阵乘法转置公式
    VT = V.T
    Sigma = np.zeros((U.shape[1], VT.shape[0]))
    Sigma[0:len(sigma_S), 0:len(sigma_S)] = np.diag(sigma_S)
    X = U@Sigma@(VT)
    loss = np.sum(np.square(X - origin))
    print(f'第{i+1}次迭代，loss值{loss}')

end = time()
print('time:',end-start)


