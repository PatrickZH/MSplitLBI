%% Generate X
n = 100;p = 80; s=5;
corr_tt = corr_tt;
% corr_st = 0.2;
% cov_ss = eye(s);
% cov_st = corr_st * ones(s,p-s);
% cov_ts = corr_st * ones(p-s,s);
% cov_tt = corr_tt * ones(p-s,p-s); 
% cov_tt(logical(speye(size(cov_tt)))) = 1;
% cov = [[cov_ss,cov_st];[cov_ts,cov_tt]];
cov = corr_tt * ones(p,p);
cov(logical(speye(size(cov)))) = 1;
rng(seed);
X = mvnrnd(zeros(p,1),cov,n);

%% Generate y
beta = zeros(p,1);
beta(1:s) = 2;
beta(s+1:8*s) = 0.2;
rng(2*seed);
y = X * beta + 0.5 * randn(n,1);

%% Using MLE
beta_est_mle = X \ y;
h_mle = norm(beta_est_mle - beta)/norm(beta);
y_mle = norm(X * beta_est_mle - X*beta) / norm(X * beta);
%% Using Ridge
h_l2 = zeros(500,1);
lambda_vec = [0:5/499:5];
for i=1:500
    lambda = lambda_vec(i);
    XtopX = X' * X / n;
    Xtopy = X' * y / n;
    beta_est_l2 = (XtopX + lambda * eye(p))^(-1) * Xtopy;
    h_l2(i) = norm(beta_est_l2 - beta)/norm(beta);
end

%% Using Lasso
h_l1 = zeros(500,1);
lambda_vec = [0:5/499:5];
for i=1:500
    lambda = lambda_vec(i);
    beta_est_l1 = linear_lasso_glmnet(X, y, lambda);
    h_l1(i) = norm(beta_est_l1 - beta)/norm(beta);
end

%% Using elastic
h_elastic = zeros(500,19);
alpha_vec = [0.05:0.05:0.95];
for i=1:500
    for j=1:19
        lambda = lambda_vec(i);
        alpha = alpha_vec(j);
        beta_est_elastic = linear_elastic_glmnet(X, y, lambda,alpha);
        h_elastic(i,j) = norm(beta_est_elastic - beta)/norm(beta);
    end
end

%% Using Split LBI

nu = nu;
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
opt.kappa = 5;
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
