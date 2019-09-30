%
%
function [EVecs, EVals, CumVar, MinDims] = task1_3(X)
% Input:
%  X : M-by-D data matrix (double)
% Output:
%  EVecs, Evals: same as in comp_pca.m
%  CumVar  : D-by-1 vector (double) of cumulative variance
%  MinDims : 4-by-1 vector (integer) of the minimum number of PCA dimensions
%            to cover 70%, 80%, 90%, and 95% of the total variance.

    D = size(X,2); % Number of columns of matrix X
    M = zeros(1,D); % Matrix for the sums of the columns
    n = length(X); % Total number of samples
    C = []; % Covariance matrix of X
    
    %CumVar = zeros(D,1);
    MinDims = zeros(4,1);
    
    % Calculate the total sum for each column of X
    for i = 1:n
        M(1,:) = M(1,:) + X(i,:);
    end
    
    M = MyMean(X); % Compute the mean for each column in the matrix
    C = MyCov(X,M); % Calculate covariance matrix
    [EVecs, EVals] = comp_pca(C); % Compute the evecs and evals
    CumVar = cumsum(EVals); % Compute the cumulative variance of each eigenvector
    
    % Plot the cumulative variances
    x = (1:784);
    plot(x,CumVar)
    title('Cumulative Variances');
    xlabel('# of principal components');
    ylabel('Cumulative variance');
    xticks(0:200:800);
    yticks(0:10:60);
    
    % Minimum number of PCA dimensions
    total = sum(EVals); % Total variance is the sum of all the eigenvalues
    for k = 1:D
        p = CumVar(k,1)/total; % Percentage of variance for the kth eigenvalue
        if p >= 0.7 && p < 0.8
            MinDims(1,1) = k;
        end
        if p >= 0.8 && p < 0.9
            MinDims(2,1) = k;
        end
        if p >= 0.9 && p < 0.95
            MinDims(3,1) = k;
        end
        if p >= 0.95
            MinDims(4,1) = k;
        end
    end

    
end


