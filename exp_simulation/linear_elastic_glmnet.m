function s = linear_elastic_glmnet(X,y,lambda,alpha)
[N,n] = size(X);
options.lambda = lambda;
options.alpha = alpha;
options.intr = 0;
options.standardize = false;
options.thresh = 1e-6;
fit = glmnet(X,y,'gaussian',options);
options.standardize = false;
s = fit.beta;
end
