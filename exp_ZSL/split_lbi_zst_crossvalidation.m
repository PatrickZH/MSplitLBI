function result_cross = split_lbi_zst_crossvalidation(seed_split,kfolds,train_index,data,file_image,save_lbi_num,opt,loss_name,local_k,local_coef)
p = length(train_index);
yy = ones(p,1);
cp = cvpartition(yy,'k',kfolds);
rng(seed_split);
index_all = [1:p]';
X = data';
acc_alpha_cross = 0;
acc_tilde_alpha_cross = 0;
acc_alpha_cross_5 = 0;
acc_tilde_alpha_cross_5 = 0;
for kkk = 1:kfolds
    index_test = find(test(cp,kkk));
    index_train = index_all(~ismember(index_all,index_test));
    index_test = train_index(index_test);
    index_train = train_index(index_train);
    index_train_set{kkk} = index_train;
    index_test_set{kkk} = index_test;
    X_train = X(:,index_train);
    Y_train = X(:,index_test);
    D = eye(length(index_train));
    %[~,~,result_validate_thistime] = split_lbi_zst_everysplit(X_train,Y_train,D,save_lbi_num,opt,loss_name);
    [~,~,result_validate_thistime] = split_lbi_zst_everysplit_local(X_train,Y_train,save_lbi_num,opt,loss_name,local_k,local_coef);
    
    % record alpha and tilde_alpha %
    alpha = result_validate_thistime.Theta;
    tilde_alpha = result_validate_thistime.Theta_revised;
    Alpha{kkk} = alpha;
    Tilde_Alpha{kkk} = tilde_alpha;
    % validation %
    fprintf('The %dth Validation ing...\n',kkk);
    [accuracy_lbi_alpha,accuracy_lbi_tilde_alpha] = split_lbi_zst_validation(index_train,index_test,file_image,data,save_lbi_num,alpha,tilde_alpha);
    acc_alpha_cross = acc_alpha_cross + length(index_test) * accuracy_lbi_alpha;
    acc_tilde_alpha_cross = acc_tilde_alpha_cross + length(index_test) * accuracy_lbi_tilde_alpha;
    
    [accuracy_lbi_alpha_5,accuracy_lbi_tilde_alpha_5] = split_lbi_zst_validation_5(index_train,index_test,file_image,data,save_lbi_num,alpha,tilde_alpha);
    acc_alpha_cross_5 = acc_alpha_cross_5 + length(index_test) * accuracy_lbi_alpha_5;
    acc_tilde_alpha_cross_5 = acc_tilde_alpha_cross_5 + length(index_test) * accuracy_lbi_tilde_alpha_5;
    
end
acc_alpha_cross = acc_alpha_cross / p;
acc_tilde_alpha_cross = acc_tilde_alpha_cross / p;
acc_alpha_cross_5 = acc_alpha_cross_5 / p;
acc_tilde_alpha_cross_5 = acc_tilde_alpha_cross_5 / p;

result_cross.acc_alpha = acc_alpha_cross;
result_cross.acc_tilde_alpha = acc_tilde_alpha_cross;
result_cross.acc_alpha_5 = acc_alpha_cross_5;
result_cross.acc_tilde_alpha_5 = acc_tilde_alpha_cross_5;

result_cross.alpha = Alpha;
result_cross.tilde_alpha = Tilde_Alpha;
result_cross.index_test = index_test_set;
result_cross.index_train = index_train_set;
end
