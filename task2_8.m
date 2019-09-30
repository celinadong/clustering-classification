function task2_8(Xtrain, Ytrain, Xtest, Ytest, epsilon, L)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Xtrain : M-by-1 label vector (uint8) for Xtrain
%   Xtest  : N-by-D test data matrix (double)
%   Ytest  : N-by-1 label vector (uint8) for Xtest
%   epsilon : A scalar parameter for regularisation
%   L      : scalar (integer) of the number of Gaussian distributions per class



    % Measure time taken for the classification experiment
    tic
    % Call the classification function
    [Ypreds, MMs, MCovs] = run_mgcs(Xtrain, Ytrain, Xtest, epsilon, L);
    toc
    
    % Obtain confusion matrix
    [cm, acc] = comp_confmat(Ytest, Ypreds, 10);
    save('task2_8_cm_L.mat', 'cm');
    
    % Mean vectors and covariance matrices for class 1
    Ms1 = MMs(1:L,:);
    save('task2_8_gL_m1.mat', 'Ms1');
    Covs1 = MCovs(1:L,:,:);
    save('task2_8_gL_cov1.mat', 'Covs1');
    
    N = sum(sum(CM,1),2); % Number of test samples
    TP = sum(diag(CM)); % Number of samples correctly classified
    Nerrs = N - TP; % Number of wrongly classified test samples
    
    fprintf("Number of test samples: %d\n", N);
    fprintf("Number of wrongly classified test samples: %d\n", Nerrs);
    fprintf("Accuracy: %d\n", acc);
    
end
