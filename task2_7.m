%
%
function [CM, acc] = task2_7(Xtrain, Ytrain, Xtest, Ytest, epsilon, ratio)
% Input:
%  Xtrain : M-by-D training data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for Xtrain
%  Xtest  : N-by-D test data matrix (double)
%  Ytest  : N-by-1 label vector (unit8) for Xtest
%  ratio  : scalar (double) - ratio of training data to use.
% Output:
%  CM     : K-by-K matrix (integer) of confusion matrix
%  acc    : scalar (double) of correct classification rate


    M = size(Xtrain,1);
    
    % Get the subset of training data, given the ratio of training data to use
    Xtrain_ss = Xtrain(1:ceil(M*ratio),:);
    Ytrain_ss = Ytrain(1:ceil(M*ratio),:);

    % Compute the Ypreds in order to get the confusion matrix and accuracy
    [Ypreds, ~, ~] = run_gaussian_classifiers(Xtrain_ss, Ytrain_ss, Xtest, epsilon);
    [CM, acc] = comp_confmat(Ytest, Ypreds, 10);

end
