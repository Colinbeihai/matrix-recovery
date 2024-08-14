clear all;

n = 10; % num of matrix
k = 20; % dim of matrix

m = round(2*n*log10(2*n)); % number of servers

p = 57;
q = 61;

iter = 50; % number of max alternating optimization iterations
iiter = 100; % number of max lsqr iterations

A = rand(n*k,k);
B = rand(k,n*k);

AB = A*B; % rank k matrix

Apower = randsample(p, n);
Bpower = randsample(q, n);

nodes = randsample(p*q, m);

Anodes = mod(nodes, p);
Bnodes = mod(nodes, q);

X = exp(-j*2*pi*Apower*Anodes'/p)/sqrt(n);
Y = exp(-j*2*pi*Bpower*Bnodes'/q)/sqrt(n);

Ei = repelem(eye(k),1,k);
Ej = repmat(eye(k),1,k);

Ei = sparse(Ei);
Ej = sparse(Ej);


Xexpand = kron(X,Ei);
Yexpand = kron(Y,Ej);


eps = 1e-5;

meas = sum(Xexpand.'*AB.*Yexpand.',2);
meas = round((1/eps)*10*meas)*eps/10;

% cvx_begin
% cvx_solver mosek
%     variable Z(n*k, n*k);
%     minimize(norm_nuc(Z));
%     subject to
%         norm(sum((Xexpand.'*Z).*Yexpand.',2)-meas) <= eps
% cvx_end


U = zeros(n*k, k);
Uh = zeros(n*k, k);
V = zeros(k, n*k);
Vh = zeros(k, n*k);


[U,~,~] = svd(full(Xexpand * diag(meas) * Yexpand.'));
U = real(U(:,[1:k]));

% U= randn(n*k,k);

B = cell(m*k*k, 1);
for i = 1:m*k*k
    B{i} = Yexpand(:,i) * (Xexpand(:,i).');
end

A = zeros(m*k*k, n*k*k);
vreltol = Inf;
ureltol = Inf;


for i=1:iter
    parfor j=1:m*k*k
        A(j,:) = reshape((B{j} * U).', 1, []);
    end
    [v,~,newvreltol] = lsqr(A,meas,0.1*eps,iiter);
    Vh = reshape(real(v), k, n*k);
    V = orth(Vh.');
    V = V.';
    parfor j=1:m*k*k
        A(j,:) = reshape((V * B{j}).', 1, []);
    end
    [u,~,newureltol] = lsqr(A,meas,0.1*eps,iiter);
    Uh = reshape(real(u), n*k, k);
    U = orth(Uh);
    fprintf('Ref tols: %f, %f\n', newvreltol, newureltol);
    if(abs(vreltol/newvreltol-1) < 0.01 && abs(ureltol/newureltol-1) < 0.01) 
        break;
    end
    vreltol = newvreltol;
    ureltol = newureltol;
end

Z = Uh*V;

err = mean(abs(AB-Z),"all");

fprintf('Mean Abs err: %f\n', err);