function [Ypreds] = run_knn_classifier(Xtrain, Ytrain, Xtest, Ks)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector (uint8) for Xtrain
%   Xtest  : N-by-D test data matrix (double)
%   Ks     : 1-by-L vector (integer) of the numbers of nearest neighbours in Xtrain
% Output:
%   Ypreds : N-by-L matrix (uint8) of predicted labels for Xtest

    M = size(Xtrain,1);
    N = size(Xtest,1);
    L = size(Ks,2);
    Ds = zeros(M,N); % Matrix for the distances between test and training points
    Ypreds = zeros(N,L);
    
    % Precompute the distances
    Ds = MySqDist(Xtrain, Xtest)';

    % Sort the distances in ascending order (rows)
    [Ds,idx] = sort(Ds, 2, 'ascend');

    for i = 1:L
        index = idx(:,1:Ks(i)); % Take the first Ks indexes
        Ypreds(:,i) = mode(Ytrain(index),2); % Column vector with the most frequent value of each row
    end
    
end
