%
%
function task2_1(Xtrain, Ytrain, Xtest, Ytest, Ks)
% Input:
%  Xtrain : M-by-D training data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for Xtrain
%  Xtest  : N-by-D test data matrix (double)
%  Ytest  : N-by-1 label vector (unit8) for Xtest
%  Ks     : 1-by-L vector (integer) of the numbers of nearest neighbours in Xtrain


    L = size(Ks,2);

    % Measure time taken for the classification experiment
    tic
    % Run classification experiment on the data set
    [Ypreds] = run_knn_classifier(Xtrain, Ytrain, Xtest, Ks);
    toc

    
    for i = 1:L

        [CM, acc] = comp_confmat(Ytest, Ypreds(:,i), 10);
        cm = CM; % Confusion matrix
        k = Ks(i); % Number of nearest neighbours
        N = sum(sum(CM,1),2); % Number of test samples
        TP = sum(diag(CM)); % Number of samples correctly classified
        Nerrs = N - TP; % Number of wrongly classified test samples
        save('task2_1_cm20.mat', 'cm');
        
        
        fprintf("Number of nearest neighbours: %d\n", k); 
        fprintf("Number of test samples: %d\n", N);
        fprintf("Number of wrongly classified test samples: %d\n", Nerrs);
        fprintf("Accuracy: %d\n", acc);  
        
    end

    
end
