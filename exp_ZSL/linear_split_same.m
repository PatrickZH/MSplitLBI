function obj = linear_split_same(data, opt, verbose)

% Input
%   data.y: Response.
%   data.X: Predictors.
%   opt: Algorithm options.
%   verbose: Show running details if verbose = true.
% Output
%   obj.alpha: Intercept, if any.
%   obj.theta: Main parameter.
%   obj.z: Auxilary parameter z = rho + phi / kappa.
%   obj.phi: Augmented parameter.
%   obj.nu
%   obj.delta: Step size;
%   obj.t_seq: opt.t_seq;
%   obj.K: Length of opt.t_seq;
%   obj.class

% Initialization

obj.class = 'linear split';

y = data.y;
X = data.X;
[n, p] = size(X);
if ~isfield(data, 'D') || isempty(data.D)
    D = eye(p);
else
    D = data.D;
end
% y = full(y);
% X = full(X);
% D = full(D);
m = size(D, 1);

opt = initial(opt);
if isempty(opt.nu)
    nu = n * norm(full(D), 2)^2 / norm(X, 2)^2 / 2; %!?
else
    nu = opt.nu;
end
kappa = opt.kappa;
if isempty(opt.delta)
    delta = opt.c * nu / kappa / (1 + nu * norm(X, 2)^2 / n + norm(full(D), 2)^2); %!?
else
    delta = opt.delta;
end
if isempty(opt.t_seq)
    if isempty(opt.t_ratio)
        if n < p, opt.t_ratio = 10; else opt.t_ratio = 100; end
    elseif opt.t_ratio <= 1, error('t_max/t_min should be larger than 1.');
    end
else
    opt.t_seq = sort(opt.t_seq);
    if opt.t_seq(1) < 0, error('Time should be non-negative.'); end
    opt.t_num = length(opt.t_seq);
end

if nargin < 3, verbose = true; end




if opt.intercept
    X_tilde = [ones(n, 1), X]; D_tilde = [zeros(m, 1), D];
    theta_tilde = pinv(nu * (X_tilde' * X_tilde) / n + D_tilde' * D_tilde) * (nu * X_tilde' * y / n);
    alpha = theta_tilde(1);
    theta = theta_tilde(2:end);
%     options = optimset('Display', 'notify', 'TolFun', 10 * eps, 'GradObj', 'on');
%     x = fminunc(@(x) func_linear_split_intercept(x, y, X, D, nu), zeros(p + 1, 1), options);
%     alpha = x(1);
%     theta = x(2:end);
else
    alpha = 0;
    theta = (nu * (X' * X) / n + D' * D) \ (nu * X' * y / n);
%     options = optimset('Display', 'notify', 'TolFun', 10 * eps, 'GradObj', 'on');
%     theta = fminunc(@(x) func_linear_split(x, y, X, D, nu), zeros(p, 1), options);
end
z = zeros(m, 1);
phi = zeros(m, 1);
obj.alpha = repmat(alpha, 1, opt.t_num);
obj.theta = repmat(theta, 1, opt.t_num);
obj.z = zeros(m, opt.t_num);
obj.phi = zeros(m, opt.t_num);
obj.cost = zeros(1, opt.t_num);



% From 0 to t0
t0 = opt.t0;
opt.t_seq = logspace(log10(t0), log10(t0 * opt.t_ratio), opt.t_num);  

if opt.fast_init
    d_phi = - D * theta / nu;
    z = z - t0 * d_phi;
else
    t0 = 0;
    theta = zeros(p,1);
end

rec_cur = sum(opt.t_seq <= t0) + 1;
steps_remain = ceil((opt.t_seq(end) - t0) / delta);
steps_remain
% Iterations from t0
D = sparse(D);
if verbose, fprintf(['Linearized Bregman Iteration (', obj.class, '):\n']); end
tic
for step_cur = 1:steps_remain
    if rec_cur > opt.t_num, break; end
    if opt.intercept
        d_alpha = alpha + mean(X * theta - y);
    end
    d_theta = X' * (X * theta + alpha - y) / n + D' * (D * theta - phi) / nu;
    d_phi = (phi - D * theta) / nu;
    if opt.intercept
        alpha = alpha - kappa * delta * d_alpha;
    end
    theta = theta - kappa * delta * d_theta;
    z = z - delta * d_phi;
    phi = kappa * sign(z) .* max(abs(z) - 1, 0);
    %phi(phi<=0) = 0;
    while true
        dt = step_cur * delta + t0 - opt.t_seq(rec_cur);
        if dt < 0, break; end
        if opt.intercept
            obj.alpha(:, rec_cur) = alpha + kappa * dt * d_alpha;
        end
        obj.theta(:, rec_cur) = theta + kappa * dt * d_theta;
        obj.z(:, rec_cur) = z + dt * d_phi;
        obj.phi(:, rec_cur) = ...
            kappa * sign(z + dt * d_phi) .* max(abs(z + dt * d_phi) - 1, 0);
%         obj.phi(:, rec_cur) = ...
%             kappa * max(z + dt * d_phi - 1, 0);
        obj.cost(rec_cur) = toc * 1;
        rec_cur = rec_cur + 1;
        if rec_cur > opt.t_num, break; end
    end
    if verbose && ismember(step_cur, round(steps_remain ./ [100 50 20 10 5 2 1]))
        fprintf('Process: %0.2f%%. Time: %f\n', step_cur / steps_remain * 100, toc);
    end
end
fprintf('\n');

obj.nu = nu;
obj.delta = delta;
obj.t_seq = opt.t_seq;
obj.K = length(opt.t_seq);

end
