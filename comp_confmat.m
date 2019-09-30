function [CM, acc] = comp_confmat(Ytrues, Ypreds, K)
% Input:
%   Ytrues : N-by-1 ground truth label vector
%   Ypreds : N-by-1 predicted label vector
% Output:
%   CM : K-by-K confusion matrix, where CM(i,j) is the number of samples whose target is the ith class that was classified as j
%   acc : accuracy (i.e. correct classification rate)

    N = size(Ytrues,1);
    CM = zeros(K,K); % Rows are the actual values and columns the predicted values

    
    for i = 1:N
        x = Ytrues(i);
        y = Ypreds(i);
        CM(x+1,y+1) = CM(x+1,y+1) + 1; % Shift by one because indexes cannot be zero
    end 
    
    samples = sum(sum(CM, 1),2); % Total number of samples
    
    TP = sum(diag(CM)); % Samples that are correctly classified
    acc = TP/samples;
 
        
end
