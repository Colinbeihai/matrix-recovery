clear all;

% pseudo
% rng(0); 

n = 5; % num of matrix
k = 5; % dim of matrix
lambda = 0.4;
Precision = 1e-6;
iter = 20;

A = randn(n*k,k);
B = randn(k,n*k);
AB = A*B; % rank k matrix

rate = 0.3; % choose mask threshold, influence observed
mask = rand(n*k, n*k) > rate;
observed = mask .* AB;

guess = 10; % k << min(m,n), guess one number >k, but also <<min(m,n)
X = randn(n*k, guess)*randn(guess, n*k);

[U, S, V] = svd(X); % we have U*S*V'=X

for i = 1:iter
    f_grad = mask.*X - observed; % derivarive fuction of recovery problem
    [U_h, S_h, V_h] = svd(X - f_grad);
    sigma = diag(S_h) - lambda;
    sigma = sigma(sigma>0);
    U_G = U_h(:, 1:size(sigma, 1)); % ȡ��sigma�ĳ��ȣ���U���в���ѡ��
    V_G = V_h(:, 1:size(sigma, 1)); 

    [Q_U, R_U] = qr([U_G U]);
    [Q_V, R_V] = qr([V_G V]);
    
    % ��Q_U, Q_V��ɸѡ��U_A, V_A
    judge1 = sum( abs( round(R_U, 4) ), 2);
    judge1 = judge1(judge1>0);
    k1 = size(judge1, 1);
    U_A = Q_U(:, 1:k1);
    
    judge2 = sum( abs( round(R_V, 4 ) ), 2);
    judge2 = judge2(judge2>0);
    k2 = size(judge2, 1);
    V_A = Q_V(:, 1:k2);
    
    cvx_begin quiet
        variable S(k1, k2)
        minimize (0.5*sum(sum_square(mask.*(U_A*S*V_A')-observed)) + lambda*norm_nuc(S))
    cvx_end

    [U_S, S_S, V_S] = svd(S);
    % ���ر�С������ֱֵ������ Precision = 1e-6
    sigma = diag(S_S);
    sigma = sigma(sigma>Precision);
    U_S = U_S(:, 1:size(sigma, 1));
    V_S = V_S(:, 1:size(sigma, 1)); 
    
    U = U_A * U_S;
    V = V_A * V_S;
    X = U*diag(sigma)*V';
    
    loss = sum((X - AB).^2, "all");
    fprintf('No.%d iteration,S:%d��%d, loss: %f\n', i, k1, k2,loss);
end

    



