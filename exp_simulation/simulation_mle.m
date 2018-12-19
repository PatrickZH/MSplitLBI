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
