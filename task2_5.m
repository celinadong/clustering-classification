%
%
function task2_5(Xtrain, Ytrain, Xtest, Ytest, epsilon)
% Input:
%  Xtrain : M-by-D training data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for Xtrain
%  Xtest  : N-by-D test data matrix (double)
%  Ytest  : N-by-1 label vector (unit8) for Xtest
%  epsilon : a scalar variable (double) for covariance regularisation

    M = size(Xtrain,1);
    N = size(Ytest,1);
    D = size(Xtrain,2);
    
    tic
    % Call classification function with epsilon = 0.01
    % epsilon = 0.01;
    [Ypreds, Ms, Covs] = run_gaussian_classifiers(Xtrain, Ytrain, Xtest, epsilon);
    toc
    
    % Confusion matrix
    [CM, acc] = comp_confmat(Ytest, Ypreds, 10);
    cm = CM;
    save('task2_5_cm.mat', 'cm');
    
    % Mean vector and covariance matrix for class 10
    M10 = Ms(10,:);
    save('task2_5_m10.mat', 'M10');
    Cov10 = Covs(10,:,:);
    save('task2_5_cov10.mat', 'Cov10');
    
    N = sum(sum(CM,1),2); % Number of test samples
    TP = sum(diag(CM)); % Number of samples correctly classified
    Nerrs = N - TP; % Number of wrongly classified test samples
    
    
    fprintf("Number of test samples: %d\n", N);
    fprintf("Number of wrongly classified test samples: %d\n", Nerrs);
    fprintf("Accuracy: %d\n", acc);

end
