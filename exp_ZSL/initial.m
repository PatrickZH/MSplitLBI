function opt_out = initial(opt_in)

% Damping factor (kappa)
opt_out.kappa = 10;
% Step size (delta)
opt_out.delta = [];
% nu. Only available for split problems!
opt_out.nu = [];
% Factor when calculating step size (delta) from data
opt_out.c = 1;
% Sequence of recorded times
opt_out.t_seq = [];
% Number of recorded times
opt_out.t_num = 500;
% t_max/t_min
opt_out.t_ratio = 100;
% Normalizing the predictors?
opt_out.normalize = false;
% Having intercept?
if ~isempty(opt_in.intercept)
    opt_out.intercept = opt_in.intercept;
end

if isfield(opt_in,'t0')
    opt_out.t0 = opt_in.t0;
end

if isfield(opt_in,'index_true')
    opt_out.index_true = opt_in.index_true;
end

if isfield(opt_in,'ratio')
    opt_out.ratio = opt_in.ratio;
end
% Directly setting the intercept to be the MLE before t0?
if ~isempty(opt_in.fast_init)
    opt_out.fast_init = opt_in.fast_init;
else
    opt_out.fast_init = false;
end
% The way to group the predictors. Only available for grouped problems!
opt_out.grouptype = 'ungrouped';

if nargin == 0
    if nargout == 0
        disp('Default options:');
        disp(opt_out);
    end
    return;
end

% Substitute opt_in fields for the default ones

fields = fieldnames(opt_in);
for i = 1:length(fields)
    field = fields{i};
    if isfield(opt_out, field);
        opt_out.(field) = opt_in.(field);
    end
end

end