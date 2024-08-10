clear all;

rng(0); % pseudo

n = 10; % num of matrix
k = 5; % dim of matrix
lambda = 0.4;
Precision = 1e-6;
iter = 10;

A = randn(n*k,k);
B = randn(k,n*k);
AB = A*B; % rank k matrix

rate = 0.2; % choose mask threshold, influence observed
mask = rand(n*k, n*k) > rate;
observed = mask .* AB;

guess = 10; % k << min(m,n), guess one number >k, but also <<min(m,n)
X = randn(n*k, guess)*randn(guess, n*k);
% 如果不guess直接生成，最开始迭代会按照总规模算，第一次迭代特别慢

[U, S, V] = svd(X); % we have U*S*V'=X

for i = 1:iter
    f_grad = mask.*X - observed; % derivarive fuction of recovery problem
    [U_h, S_h, V_h] = svd(X - f_grad);
    sigma = diag(S_h) - lambda;
    sigma = sigma(sigma>0);
    U_G = U_h(:, 1:size(sigma, 1)); % 取出sigma的长度，对U进行部分选择
    V_G = V_h(:, 1:size(sigma, 1)); 

    [Q_U, R_U] = qr([U_G U]);
    [Q_V, R_V] = qr([V_G V]);
    
    % 从Q_U, R_U中筛选出U_A, R_U
    % U_A = 
    
    cvx_begin quiet
        variable S(n*k, n*k)
        minimize (0.5*sum(sum_square(mask.*(U_A*S*V_A')-observed)) + lambda*norm_nuc(S))
    cvx_end

    [U_S, S_S, V_S] = svd(S);
    % 对特别小的奇异值直接舍弃 Precision = 1e-6
    sigma = diag(S_S);
    sigma = sigma(sigma>Precision);
    U_S = U_S(:, 1:size(sigma, 1));
    V_S = V_S(:, 1:size(sigma, 1)); 
    
    U = U_A * U_S;
    V = V_A * V_S;
    X = U*diag(sigma)*V';
    
    loss = sum((X - AB).^2, "all");
    fprintf('loss: %f\n', loss);
    
end

    



