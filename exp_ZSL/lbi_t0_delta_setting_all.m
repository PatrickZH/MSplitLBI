function [t0,delta] = lbi_t0_delta_setting_all(X_data,Y_data,D,opt,loss_name)
[n,~] = size(X_data{1});
t0 = inf;
delta = inf;
nu = opt.nu;
kappa = opt.kappa;
split_num = length(X_data);
tic
for s = 1:split_num
    X = X_data{s};
    Y = Y_data{s};
    for i = 1:size(Y,2)
        y = Y(:,i);
        if strcmp(loss_name,'logistic')
            delta_temp = nu / kappa / (1 + nu * norm(full(bsxfun(@minus, X, mean(X))), 2)^2 / n + norm(full(D), 2)^2);
            theta_temp = logisitic_mosek(X,y,D,nu);
        end
        if strcmp(loss_name,'linear')
            y(y==-1) = 0;
            delta_temp = nu / kappa / max(nu * norm(X, 2)^2 / n,norm(D,2)^2);
            theta_temp = linear_mosek_zst(X,y,D,nu);
        end
        t0_temp = 1 / max(abs(D* theta_temp / nu));
        if delta_temp < delta
            delta = delta_temp;
        end
        if t0_temp < t0
            t0 = t0_temp;
        end
         
%         delta
    end
    if mod(s,10) == 0
        formatSpec = 'Now we have finished %0.2f%% preprocessing for split lbi.\n';
        fprintf(formatSpec,s / split_num * 100);
        toc;
    end
end
end