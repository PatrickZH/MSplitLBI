function [accuracy,Labels_predict,acc_each] = classifier_nearest(Predicted,Signatures,label_list,groundtruth)
% % Predicted: features in common space, for traditional method, it is predicted attributes/wordvectors;...
% for Manifold Alignment it is aligned image features 
% % Signatures: label signatures in common space, for traditional methods,
% it is label signatures in attribute/wordvector space; for Manifold
% Aligment: use aligned attributes/wordvectors
% % label_list: labels of each signatures
% % groundtruth: ground truth label of each sample
% % Labels_predict: predicted labels by this nearest neighbour classifier
% % accuracy: average accuracy

N_labels = size(Signatures,1);
Labels_predict = zeros(size(groundtruth));
for i = 1:size(Predicted,1)
    dis = sum((repmat(Predicted(i,:),N_labels,1)-Signatures).^2,2);
    [v,label] = min(dis);
    Labels_predict(i) = label_list(label);
end

accuracy = sum(Labels_predict==groundtruth)/length(groundtruth);

acc_each = [];
% acc_each = zeros(length(label_list),1);
% C = Labels_predict==groundtruth;
% for i = 1:length(label_list)
%     index = groundtruth==label_list(i);
%     acc_each(i) = sum(C(index))./sum(index);
% end
% N = length(label_list);
% acc_each = zeros(N,1);
% num_each = zeros(N,1);
% acnum = zeros(N,1);
% precision_each = zeros(N,1);
% recall_each = zeros(N,1);
% result = groundtruth==Labels_predict;
% total = 0;
% for i =1:N
%     index = find(groundtruth==label_list(i));
%     num_each(i) = length(index);
%     acnum(i) = sum(result(index));
%     acc_each(i) = acnum(i)/num_each(i);
%     total = length(index) + total;
%     recall_each(i) = acnum(i)/num_each(i);
%     precision_each(i) = acnum(i)/length(find(Labels_predict == label_list(i)));
% end
% accuracy = sum(Labels_predict==groundtruth)/size(groundtruth,1);% [precision_each,recall_each];%

end