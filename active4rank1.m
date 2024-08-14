clear all;

rng(0);

n = 4; % num of matrix
k = 5; % dim of matrix

m = round(2*n*log10(2*n)); % number of servers

p = 71;
q = 73;

iter = 60; % number of max alternating optimization iterations

lambda = 0.5;
Precision = 1e-3;


A = randn(n*k, k);
B = randn(k, n*k);

AB = A*B; % rank k matrix

Apower = randsample(p, n);
Bpower = randsample(q, n);

nodes = randsample(p*q, m);

Anodes = mod(nodes, p);
Bnodes = mod(nodes, q);

X = exp(-1i*2*pi*Apower*Anodes'/p)/sqrt(n);
Y = exp(-1i*2*pi*Bpower*Bnodes'/q)/sqrt(n);

Ei = repelem(eye(k),1,k);
Ej = repmat(eye(k),1,k);

Ei = sparse(Ei);
Ej = sparse(Ej);


Xexpand = kron(X,Ei);
Yexpand = kron(Y,Ej);


eps = 1e-5;
% meas = sum(Xexpand.'*AB.*Yexpand.',2);
% meas = round((1/eps)*10*meas)*eps/10;

mask = (Xexpand * Yexpand');
meas = mask .* AB;

guess = 2*k; % k << min(m,n), guess one number >k, but also <<min(m,n)
X = randn(n*k, guess)*randn(guess, n*k);
err = mean(abs(AB-X),"all");
    fprintf('初始loss: %f\n', err);
[U, S, V] = svd(X);
truthK = n*k;

for i = 1:iter
    % f_grad is the derivarive fuction of the problem
    f_grad = (mask.*X - meas) .* mask;
    [U_h, S_h, V_h] = svd(X - f_grad);
    sigma = diag(S_h) - lambda;
    sigma = sigma(sigma>0);
    U_G = U_h(:, 1:size(sigma, 1)); % 取出sigma的长度，对U进行部分选择
    V_G = V_h(:, 1:size(sigma, 1)); 

    [Q_U, R_U] = qr([U_G U]);
    [Q_V, R_V] = qr([V_G V]);
    
    % 从Q_U, Q_V中筛选出U_A, V_A
    judge1 = abs(sum( round(R_U, 3) , 2));
    judge2 = abs(sum( round(R_V, 3) , 2));

    judge1 = judge1(judge1>0.1);
    judge2 = judge2(judge2>0.1);
    k1 = size(judge1, 1);
    k2 = size(judge2, 1);

    ks = min( truthK , guess);
%     disp(ks);
        
    U_A = Q_U(:, 1:k1);
    V_A = Q_V(:, 1:k2);
    

    cvx_begin quiet
        variable S(size(U_A, 2), size(V_A, 2)) complex
        ER = mask.*(U_A*S*V_A')-meas;
        minimize(0.5*sum(dot(ER, ER))+lambda*norm_nuc(S))
    cvx_end

    [U_S, S_S, V_S] = svd(full(S));
    % 对特别小的奇异值直接舍弃 Precision << 1
    sigma2 = diag(S_S);
    sigma2 = sigma2(sigma2>Precision);
    truthK = size(sigma2, 1);
    disp(size(sigma2, 1));
    U_S = U_S(:, 1:size(sigma2, 1));
    V_S = V_S(:, 1:size(sigma2, 1)); 
    
    U = U_A * U_S;
    V = V_A * V_S;
    X = U*diag(sigma2)*V';
    
    loss = sum((X - AB).^2, "all");
    err = mean(abs(AB-X),"all");
    fprintf('No.%d iteration,S:%d×%d, loss: %f\n', i, ks, ks, err);
end

