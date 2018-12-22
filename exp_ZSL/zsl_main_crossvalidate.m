clc;clear;

seed_split = 1;

% AwA
file_split = 'splits_AwA_default.mat';
file_embedding = 'AwA_Attribute.mat';
file_image = 'AwA_ImageFeatures_VGG_80d.mat';
num_allclasses = 50;
num_train = 40;
num_test = 10;



% CUB
% file_split = 'splits_CUB_default.mat';
% file_embedding = 'CUB_Attribute.mat';
% file_image = 'CUB_ImageFeatures_GoogRes_400d.mat';
% num_allclasses = 200;
% num_train = 150;
% num_test = 50;


load(file_embedding);
load(file_split);
load(file_image);

opt.nu = 3;
opt.kappa = 10;
opt.intercept = false;
opt.fast_init = false;
opt.t_ratio = 100;
save_lbi_num = 500;
loss_name = 'linear';
split_num_short = 300;
split_index_vec = [1:10];
local_k = round(length(num_allclasses) * 0.1);
local_coef = 1/4; % The code here can be modified: local_coef = 1/4 or = 1/8

%% computation of t0 and delta %%

split = load(file_split); % load split dataset %
test_data_index = split.splits;
[split_num,~] = size(test_data_index);

% load embedding dataset %
dataset_embedding = load(file_embedding); 
data = dataset_embedding.Attribute;
index_all = [1:size(data,1)]';

% load image dataset %
load(file_image);
split_num = min(split_num,split_num_short);

% normalization %
data_max = max(abs(data(:)));
data = data / data_max;

% computation of t0 and delta %
for s = 1:split_num
    test_index = test_data_index(s,:);
    train_index = index_all(~ismember(index_all,test_index));
    train_data = data(train_index,:);
    test_data = data(test_index,:);
    X = train_data';
    Y = test_data';
    Y(Y==0) = -1;
    [~,p] = size(X);
    D = eye(p);
    X_data{s} = X;
    Y_data{s} = Y;
end
[t0,delta] = lbi_t0_delta_setting_all(X_data,Y_data,D,opt,loss_name);
%[t0,delta] = lbi_t0_delta_setting_all_local(X_data,Y_data,opt,loss_name);
opt.t0 = t0;
opt.delta = delta;




N_iter = 1;
T = 1;
acc_lbi_alpha_cub = zeros(N_iter,T);
acc_lbi_tilde_alpha_cub = zeros(N_iter,T);
kfolds = 4; % the number of folds in cross-validation %
for iter = 1:1 % for different splits of seen/unseen classes
    
    
    
    %% computation of alpha and tilde_alpha %
    
    test_index = test_data_index(iter,:);
    train_index = index_all(~ismember(index_all,test_index));
    
    train_data = data(train_index,:);
    test_data = data(test_index,:);
    
    result_validate_thistime = split_lbi_zst_crossvalidation(seed_split,kfolds,train_index,data,file_image,save_lbi_num,opt,loss_name,local_k,local_coef);

    % record alpha and tilde_alpha %
    alpha = result_validate_thistime.alpha;
    tilde_alpha = result_validate_thistime.tilde_alpha;
    Theta_validate{iter} = alpha;
    Theta_revised_validate{iter} = tilde_alpha;
   
    % record validation result %
    [~,ind_max_alpha] = max(result_validate_thistime.acc_alpha);
    [~,ind_max_tilde_alpha] = max(result_validate_thistime.acc_tilde_alpha);
    ind_alpha{iter} = ind_max_alpha;
    ind_tilde_alpha{iter} = ind_max_tilde_alpha;
    acc_alpha{iter} = result_validate_thistime.acc_alpha;
    acc_tilde_alpha{iter} = result_validate_thistime.acc_tilde_alpha;
    
    
    % train on whole training set %
    X = X_data{iter};
    Y = Y_data{iter};
    [~,p] = size(X);
    D = eye(p);
    [~,~,result_train_tmp] = split_lbi_zst_everysplit(X,Y,D,save_lbi_num,opt,loss_name);
    %[~,~,result_train_tmp] = split_lbi_zst_everysplit_local(X,Y,save_lbi_num,opt,loss_name);
    Theta_train{iter} = result_train_tmp.Theta;
    Theta_revised_train{iter} = result_train_tmp.Theta_revised;
    
    % accuracy on test dataset %

    list_all = [1:num_allclasses]';
    list_test = splits(iter,:)';
    list_train = list_all;
    list_train(list_test) = [];
    disp(list_test');
     
     FeaTrain = []; % mean vectors of image features in each seen class
     WorTrain = []; % word vectors of each seen class name
     AttTrain = [];
     X_Train = [];
     Y_Train = [];
     for i = 1:length(list_train)
     index = find(Labels==list_train(i));
     X_Train = [X_Train;ImageFeatures(index,:)];
     Y_Train = [Y_Train;Labels(index,:)];
     FeaTrain = [FeaTrain;mean(ImageFeatures(index,:))];
     %         WorTrain = [WorTrain;WordVectors(list_train(i),:)];
     AttTrain = [AttTrain;Attribute(list_train(i),:)];
     end
     FeaTest = [];
     WorTest = [];
     AttTest = [];
     X = [];
     Y = [];
     for i = 1:length(list_test)
     index = find(Labels==list_test(i));
     X = [X;ImageFeatures(index,:)];
     Y = [Y;Labels(index)];
     FeaTest = [FeaTest;mean(ImageFeatures(index,:))];
     %         WorTest = [WorTest;WordVectors(list_test(i),:)];
     AttTest = [AttTest;Attribute(list_test(i),:)];
     end
    
    alpha = result_train_tmp.Theta(:,400,:);
    tilde_alpha = result_train_tmp.Theta_revised(:,ind_max_alpha,:);
    %ind_alpha = result_validate.ind_alpha{iter};
    %ind_tilde_alpha = result_validate.ind_tilde_alpha{iter};
    
    
    lbi_a = reshape(alpha,[num_train,num_test]);
    FeaRecon_A = (FeaTrain'*lbi_a)';
    [accuracy_A_Rec_LBI,~] = classifier_nearest(X,FeaRecon_A,list_test,Y);
    acc_lbi_alpha_cub(iter) = accuracy_A_Rec_LBI;
    
    lbi_a = reshape(tilde_alpha,[num_train,num_test]);
    FeaRecon_A = (FeaTrain'*lbi_a)';
    [accuracy_A_Rec_LBI,~] = classifier_nearest(X,FeaRecon_A,list_test,Y);
    acc_lbi_tilde_alpha_cub(iter) = accuracy_A_Rec_LBI;
    
    fprintf('This is the %d th time/n',iter);
end
result_validate_cub.alpha = Theta_validate;
result_validate_cub.tilde_alpha = Theta_revised_validate;
result_validate_cub.ind_alpha = ind_alpha;
result_validate_cub.ind_tilde_alpha = ind_tilde_alpha;
result_validate_cub.acc_alpha = acc_alpha;
result_validate_cub.acc_tilde_alpha = acc_tilde_alpha;
result_train_cub.alpha = Theta_train;
result_train_cub.tilde_alpha = Theta_revised_train;
lbi_alpha_acc_mean_cub = mean(acc_lbi_alpha_cub);
lbi_tilde_alpha_acc_mean_cub = mean(acc_lbi_tilde_alpha_cub);
