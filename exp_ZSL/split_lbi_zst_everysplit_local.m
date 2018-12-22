function [pred,tilde_beta_nonzeros,result_each_sim] = split_lbi_zst_everysplit_local(X,Y,save_lbi_num,opt,loss_name,local_k,local_coef)
[~,p] = size(X);
unseen_num = size(Y,2);
Theta_each_sim = zeros(p,save_lbi_num,unseen_num);
Theta_revised_each_sim = zeros(p,save_lbi_num,unseen_num);
% pred_tilde_beta = zeros(save_lbi_num,1);
pred = zeros(save_lbi_num,1);
tilde_beta_nonzeros = zeros(save_lbi_num,1);
data.X = X;
%data.D = D;
for u = 1:unseen_num
    y = Y(:,u);
    data.y = y;
    if strcmp(loss_name,'logistic')
        lbi = logistic_split_same(data,opt,true);
        y(y==-1) = 0;
        phi = lbi.phi;
        phi(abs(phi) < 1e-12) = 0;
        theta = lbi.theta;
        theta_revised = zeros(p,size(phi,2));
        pred_tilde_beta_temp = zeros(size(phi,2),1);
        pred_beta_temp = zeros(size(phi,2),1);
        tilde_beta_nonzeros_temp = zeros(size(phi,2),1);
        for kk = 1:size(phi, 2)
            
            fprintf('%d\n', kk);
            %tmp = D(~phi(:, kk), :); tmp = tmp' * tmp;
            %theta_revised(:, kk) = (eye(p) - pinv(full(tmp)) * tmp) * theta(:, kk);
            %theta_revised_temp = theta_revised(:,kk);
            %theta_revised_temp(abs(theta_revised_temp)<=1e-6) = 0;
            theta_revised(phi(:, kk)~=0,kk) = theta(phi(:, kk)~=0,kk);
            %tilde_beta_nonzeros_temp(kk) = sum(theta_revised_temp~=0);
            y_tilde = exp(X*theta_revised(:,kk)) ./ (exp(X*theta_revised(:,kk)) + 1);
            y_tilde(y_tilde > 0.5) = 1;
            y_tilde(y_tilde <= 0.5) = 0;
            
            y_beta = exp(X*theta(:,kk)) ./ (1 + exp(X*theta(:,kk)));
            y_beta(y_beta > 0.5) = 1;
            y_beta(y_beta <= 0.5) = 0;
%             pred_tilde_beta_temp(kk) = sum(y == y_tilde);
%             pred_beta_temp(kk) = sum(y == y_beta);
        end
    end
    if strcmp(loss_name,'linear')
        y(y==-1) = 0;
        data.y = y;
        d = norms(repmat(y,1,size(X,2)) - X,[],1);
        [~,ind_sort] = sort(d,'ascend');
        d(ind_sort(1:local_k)) = local_coef;
        d(ind_sort(local_k+1:end)) = 1;
        D = diag(d);
        
           
        data.D = D / max(diag(D));
        opt.t_num = save_lbi_num;
        lbi = linear_split_same(data,opt,true);
        phi = lbi.phi;
        phi(abs(phi) < 1e-12) = 0;
        theta = lbi.theta;
        theta_revised = zeros(p,size(phi,2));
%         pred_tilde_beta_temp = zeros(size(phi,2),1);
%         pred_beta_temp = zeros(size(phi,2),1);
%         tilde_beta_nonzeros_temp = zeros(size(phi,2),1);
        for kk = 1:size(phi, 2)
            fprintf('%d\n', kk);
            %tmp = D(~phi(:, kk), :); tmp = tmp' * tmp;
            %theta_revised(:, kk) = (eye(p) - pinv(full(tmp)) * tmp) * theta(:, kk);
            %theta_revised_temp = theta_revised(:,kk);
            %theta_revised_temp(abs(theta_revised_temp)<=1e-6) = 0;
            theta_revised(phi(:, kk)~=0,kk) = theta(phi(:, kk)~=0,kk);
            %tilde_beta_nonzeros_temp(kk) = sum(theta_revised_temp~=0);
            y_tilde = X*theta_revised(:,kk);
            y_tilde(y_tilde > 0.5) = 1;
            y_tilde(y_tilde <= 0.5) = 0;
            
            y_beta = X*theta(:,kk);
            y_beta(y_beta > 0.5) = 1;
            y_beta(y_beta <= 0.5) = 0;
%             pred_tilde_beta_temp(kk) = sum(y == y_tilde);
%             pred_beta_temp(kk) = sum(y == y_beta);
        end
    end
%     pred_tilde_beta = pred_tilde_beta + pred_tilde_beta_temp;
%     pred_beta = pred_beta + pred_beta_temp;
%     tilde_beta_nonzeros = tilde_beta_nonzeros + tilde_beta_nonzeros_temp;
    Theta_each_sim(:,:,u) = theta;
    Theta_revised_each_sim(:,:,u) = theta_revised;
end
% tilde_beta_nonzeros = tilde_beta_nonzeros / p / unseen_num;
% pred.tilde_beta = pred_tilde_beta;
% pred.beta = pred_beta;
result_each_sim.Theta = Theta_each_sim;
result_each_sim.Theta_revised = Theta_revised_each_sim;
end
