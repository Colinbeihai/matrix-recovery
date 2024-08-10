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

m, n = 80, 80
k = 4
rate = 0.2
L = 0.3 # lambda in Algorithm
iter = 10

m1 = np.random.rand(m, k)
m2 = np.random.rand(k, n)
origin = np.matmul(m1, m2)
mask = np.random.rand(m, n) > rate
observed = mask*origin # 点积
 
# 猜测的问题秩上界，k不大于guess,否则最开始迭代会按照总规模算，第一次迭代特别慢
guess = 10
m3 = np.random.rand(m, guess)
m4 = np.random.rand(guess, n)
X = np.matmul(m3, m4)

# 随机生成X
# X = np.random.rand(m, n)
print(f'问题规模{m}×{n},初始loss值{np.sum(np.square(mask*X - mask*origin))}')

def f_grad(X):
    return mask*X - observed

def reduce_SVD(U, sigma, VT, L=0):
    if L != 0:
        sigma = sigma - L
    else:
        sigma = sigma - 1e-3
    sigma = sigma[sigma>0]
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
    VT_qr = np.concatenate((VT_G, VT),axis=0)
    
    Q_U, R_U= np.linalg.qr(U_qr, 'complete')
    Q_V, R_V = np.linalg.qr(VT_qr.T, 'complete') # 注意将VT转置为V
    
    # 从分解出的Q中找到U_A, V_A
    rule1 = np.sum( np.absolute( np.around(R_U, 6) ), axis=1 ) # 找全0行
    rule1 = rule1[rule1>0]
    k1 = len(rule1)
    U_A = Q_U[:, 0:k1]
    
    rule2 = np.sum( np.absolute( np.around(R_V, 6) ), axis=1 )
    rule2 = rule1[rule1>0]
    k2 = len(rule2)
    V_A = Q_V[:, 0:k2]
    VT_A = V_A.T
    
    # 对S的shape做删减,开始用cvx包做凸优化
    S = cp.Variable((k1, k2))
    USA = U_A@S@(VT_A)
    objective = cp.Minimize(0.5*cp.sum_squares(cp.multiply(mask, USA)-observed)
                            + L*cp.normNuc(S))
    problem = cp.Problem(objective)
    problem.solve()
    result_S = S.value
    # 凸优化结束，结果是result_S
    
    U_S, sigma_S, VT_S = np.linalg.svd(result_S, full_matrices=False)
    U_S, sigma_S, VT_S = reduce_SVD(U_S, sigma_S, VT_S)
    U = U_A@U_S
    V = (VT_S@VT_A).T # 矩阵乘法转置公式
    
    VT = V.T
    Sigma = np.zeros((U.shape[1], VT.shape[0]))
    Sigma[0:len(sigma_S), 0:len(sigma_S)] = np.diag(sigma_S)
    X = U@Sigma@(VT)
    loss = 0.5*np.sum(np.square(X - origin))
    print(f'第{i+1}次迭代，loss值{loss},k1={k1},k2={k2}')
    
    # if loss < 1:
    #     break

end = time()
print('time:',end-start)
  


