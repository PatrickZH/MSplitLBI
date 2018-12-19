h_l2_m = zeros(500,1);
lambda_vec = [0:5/499:5];
D = eye(p);
for i = 1:s
    D(i,i) = 0;
end
for i=1:500
    lambda = lambda_vec(i);
    XtopX_lambda = [[A,B];[B',C + lambda * eyet]];
    XtopX = [[A,B];[B',C]];
%     XtopX = X' * X;
%     XtopX_lambda = XtopX + lambda * D;
    beta_est_l2 = XtopX_lambda ^(-1) * X' * y / n;
    h_l2_m(i) = norm(beta_est_l2 - beta) / norm(beta);
end


nu = 10;
D = eye(p);
theta = pinv(nu * (X' * X) / n + D' * D) * (nu * X' * y / n);
d_phi = - D * theta / nu;
opt.t0 = 1 / max(max(abs(d_phi)));
data.X = X;
data.y = y;
data.D = D;
opt.intercept=false;
opt.fast_init=false;
opt.t_ratio = 100;
opt.nu = nu;
opt.kappa = 100;
obj = linear_split(data, opt);
theta = obj.theta;
gamma = obj.phi;
theta_tilde = theta;
for i=1:size(theta,2)
    theta_tmp = theta(:,i);
    gamma_tmp = gamma(:,i);
    theta_tilde_tmp = zeros(length(theta_tmp),1);
    theta_tilde_tmp(gamma_tmp~=0) = theta_tmp(gamma_tmp~=0);
    theta_tilde(:,i) = theta_tilde_tmp;
end

h_theta = zeros(500,1);
y_theta = zeros(500,1);
for i=1:500
    h_theta(i) = norm(theta(:,i)-beta)/norm(beta);
    y_theta(i) = norm(X * theta(:,i) - X * beta)/norm(X * beta);
end


h_theta_tilde = zeros(500,1);
y_theta_tilde = zeros(500,1);
for i=1:500
    h_theta_tilde(i) = norm(theta_tilde(:,i)-beta)/norm(beta);
    y_theta_tilde(i) = norm(X * theta_tilde(:,i) - X * beta)/norm(X * beta);
end





Xs = X(:,1:s);
Xt = X(:,s+1:end);
A = Xs'*Xs / n;
B = Xs'*Xt / n;
C = Xt'*Xt / n;
eyet = eye(p-s);
betat = beta(s+1:end);

rng(2*seed);
a = 0.5 * randn(n,1);
h_s_hat = zeros(499,1);
h_abs_s_hat = zeros(499,1);
h_t_hat = zeros(499,1);
h_abs_t_hat = zeros(499,1);
h_hat = zeros(499,1);
h_abs_hat = zeros(499,1);
for i=1:499
    lambda = lambda_vec(i+1);
    s_hat_dif1 = lambda * A^(-1) * B * (C + lambda * eyet - B'*A^(-1)*B)^(-1) * betat;
    s_hat_dif2 = (A^(-1) + A^(-1) * B * (C + lambda * eyet - B'*A^(-1)*B)^(-1) * B' * A^(-1)) * Xs' / n * a;
    s_hat_dif3 = -A^(-1) * B * (C + lambda * eyet - B'*A^(-1)*B)^(-1) * Xt' / n * a;
    s_hat = s_hat_dif1 + s_hat_dif2 + s_hat_dif3;
    abs_s_hat = abs(s_hat_dif1) + abs(s_hat_dif2) + abs(s_hat_dif3);
    h_s_hat(i) = norm(s_hat);
    h_abs_s_hat(i) = norm(abs_s_hat);
    
    t_hat_dif1 = -lambda * (C + lambda * eyet - B'*A^(-1)*B)^(-1) * betat;
    t_hat_dif2 = -(C + lambda * eyet - B'*A^(-1)*B)^(-1) * B' * A^(-1) * Xs' / n * a;
    t_hat_dif3 = (C + lambda * eyet - B'*A^(-1)*B)^(-1) * Xt' / n * a;
    t_hat = t_hat_dif1 + t_hat_dif2 + t_hat_dif3;
    abs_t_hat = abs(t_hat_dif1) + abs(t_hat_dif2) + abs(t_hat_dif3);
    h_t_hat(i) = norm(t_hat);
    h_abs_t_hat(i) = norm(abs_t_hat);
    
    h_hat(i) = norm([s_hat;t_hat]);
    h_abs_hat(i) = norm([abs_s_hat;abs_t_hat]);
    
end


lambda = 1;
XtopX_lambda = [[A,B];[B',C + lambda * eyet]];
XtopX = [[A,B];[B',C]];
beta_est = XtopX_lambda ^(-1) * XtopX * beta;
dif_ground_s = beta_est(1:s) - beta(1:s);
dif_ground_t = beta_est(s+1:p) - beta(s+1:end);
beta_est_m = XtopX_lambda ^ (-1) * X' * y / n;
dif_ground_m_s = beta_est_m(1:s) - beta(1:s);
dif_ground_m_t = beta_est_m(s+1:end) - beta(s+1:end);

s_hat_dif1 = lambda * A^(-1) * B * (C + lambda * eyet - B'*A^(-1)*B)^(-1) * betat;
s_hat_dif2 = (A^(-1) + A^(-1) * B * (C + lambda * eyet - B'*A^(-1)*B)^(-1) * B' * A^(-1)) * Xs' / n  * a;
s_hat_dif3 = -A^(-1) * B * (C + lambda * eyet - B'*A^(-1)*B)^(-1) * Xt' / n * a;
s_hat = s_hat_dif1 + s_hat_dif2 + s_hat_dif3;
t_hat_dif1 = -lambda * (C + lambda * eyet - B'*A^(-1)*B)^(-1) * betat;
t_hat_dif2 = -(C + lambda * eyet - B'*A^(-1)*B)^(-1) * B' * A^(-1) * Xs' / n * a;
t_hat_dif3 = (C + lambda * eyet - B'*A^(-1)*B)^(-1) * Xt' / n * a;
t_hat = t_hat_dif1 + t_hat_dif2 + t_hat_dif3;