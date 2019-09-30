function [Ypreds, Ms, Covs] = run_gaussian_classifiers(Xtrain, Ytrain, Xtest, epsilon)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector for Xtrain (uint8)
%   Xtest  : N-by-D test data matrix (double)
%   epsilon : A scalar variable (double) for covariance regularisation
% Output:
%  Ypreds : N-by-1 matrix (uint8) of predicted labels for Xtest
%  Ms     : K-by-D matrix (double) of mean vectors
%  Covs   : K-by-D-by-D 3D array (double) of covariance matrices

%YourCode - Bayes classification with multivariate Gaussian distributions.

    N = size(Xtest,1);
    D = size(Xtrain,2);
    
    % Compute mean values for each class
    S = zeros(11, size(Xtrain,2)); % Sums of the vectors for each class
    C = zeros(11,1); % Counter
    C(11) = size(Ytrain,1); % Total number of samples
    M = zeros(11, size(Xtrain,2)); % Mean vectors for each class and total for all classes

    % Compute the sums of vectors for each sample
    for i = 1:length(Ytrain)
        S(Ytrain(i)+1,:) = S(Ytrain(i)+1,:) + Xtrain(i,:);
        S(11,:) = S(11,:) + Xtrain(i,:);
        C(Ytrain(i)+1) = C(Ytrain(i)+1) + 1; % Number of samples for class Y(i)
    end

    % Compute the means of each vector class
    for j = 1:10
        M(j,:) = S(j,:)./C(j);   
    end
    % Mean vector of all samples
    M(11,:) = S(11,:)./C(11);
    
    
    
    
    
    
    % Iterate over each class
    for i = 1:10
        
        m_i = M(i,:); % Row mean vector
        covMatrix = MyCov(Xtrain(Ytrain == i-1,:), m_i);
        d = size(covMatrix,1);
        I = eye(d); % Identity matrix
        covMatrix = covMatrix + (epsilon * I);
        invCovMatrix = inv(covMatrix);
        
        Ms(i,:) = m_i;
        Covs(i,:,:) = reshape(covMatrix, [1, D, D]);
        
        % Compute the multivariate Gaussian distribution
        X = Xtest - (ones(N,1) * m_i); % Subtract mean from data point
        fact = sum(((X * invCovMatrix) .* X), 2);
        y = exp(-0.5 * fact);
        y = y./sqrt((2*pi)*(D/2)*logdet(covMatrix));
        %y = log(y) - (0.5*(D*(det(covMatrix)))*log(2*pi)); 
        test_prob(:,i) = y;
   
    end
    
    % Assign each data point in the Xtest with the highest probability 
    [~, Ypreds] = max(test_prob, [], 2);
    % Indexes are from 1 to 10, but the classes from 0 to 9, so we subtract
    % the Ypreds by 1
    Ypreds = Ypreds - 1;
    
end
