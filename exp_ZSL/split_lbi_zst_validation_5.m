function [accuracy_lbi_alpha,accuracy_lbi_tilde_alpha] = split_lbi_zst_validation_5(list_train,list_validate,file_image,Attribute,save_lbi_num,alpha,tilde_alpha)
load(file_image);
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
    AttTrain = [AttTrain;Attribute(list_train(i),:)];
end
FeaTest = [];
WorTest = [];
AttTest = [];
X = [];
Y = [];
for i = 1:length(list_validate)
    index = find(Labels==list_validate(i));
    X = [X;ImageFeatures(index,:)];
    Y = [Y;Labels(index)];
    FeaTest = [FeaTest;mean(ImageFeatures(index,:))];
    AttTest = [AttTest;Attribute(list_validate(i),:)];
end



accuracy_lbi_alpha = zeros(save_lbi_num,1);
accuracy_lbi_tilde_alpha = zeros(save_lbi_num,1);

for t = 1:save_lbi_num
    lbi_a = reshape(alpha(:,t,:),[length(list_train),length(list_validate)]);
    FeaRecon_A = (FeaTrain'*lbi_a)';
    [accuracy_A_Rec_LBI,~] = classifier_nearest_5(X,FeaRecon_A,list_validate,Y);
    accuracy_lbi_alpha(t) = accuracy_A_Rec_LBI;
    
    lbi_a = reshape(tilde_alpha(:,t,:),[length(list_train),length(list_validate)]);
    FeaRecon_A = (FeaTrain'*lbi_a)';
    [accuracy_A_Rec_LBI,~] = classifier_nearest_5(X,FeaRecon_A,list_validate,Y);
    accuracy_lbi_tilde_alpha(t) = accuracy_A_Rec_LBI;
end
end