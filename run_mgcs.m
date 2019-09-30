function [Ypreds, MMs, MCovs] = run_mgcs(Xtrain, Ytrain, Xtest, epsilon, L)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector for Xtrain (uint8)
%   Xtest  : N-by-D test data matrix (double)
%   epsilon : A scalar parameter for regularisation (double)
%   L      : scalar (integer) of the number of Gaussian distributions per class
% Output:
%  Ypreds : N-by-1 matrix of predicted labels for Xtest (integer)
%  MMs     : (L*K)-by-D matrix of mean vectors (double)
%  MCovs   : (L*K)-by-D-by-D 3D array of covariance matrices (double)


    N = size(Xtest,1);
    D = size(Xtrain,2);
    
    MMs = zeros(L*10,D);
    MCovs = zeros(L*10,D,D);
    
    
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
 
    
    
    
    
    
    
    %maxIter = 500;
    counter = 1; % Counter for MMs and MCovs
    for i = 1:10
        
        samples = Xtrain(Ytrain(:,1) == i,:);
        [C,idx,~] = my_kMeansClustering(samples, L, samples(1:L,:)); % Apply k-means clustering
        idx = idx';
        
        % Retrieve all points that belong to a specific cluster
        for l = 1:L
            cluster = samples(idx(:,1) == l,:);
            mean_l = MyMean(cluster);
            I = eye(D);
            MMs(counter,:) = mean_l;
            MCovs(counter,:,:) = MyCov(cluster,MyMean(cluster)) + I;
            counter = counter + 1;
        end
        
    end
    
    % Convert s into 2D and compute probability of each data point
    counter = 1;
    for i = 1:10
        
        for l = 1:L
            m_i = M(i,:); % Row mean vector
            covMatrix = MyCov(Xtrain(Ytrain == i-1,:), m_i);
            d = size(covMatrix,1);
            I = eye(d);
            covMatrix = covMatrix + (epsilon * I);
            invCovMatrix = inv(covMatrix);

            % Compute the multivariate Gaussian distribution
            X = Xtest - (ones(N,1) * m_i); % Subtract mean from data point
            fact = sum(((X * invCovMatrix) .* X), 2);
            y = exp(-0.5 * fact);
            y = y./sqrt((2*pi)*(D/2)*logdet(covMatrix));
            test_prob(:,i) = y;
            counter = counter + 1;
        end
        
    end
   
   
    
    % Assign each data point in the Xtest with the highest probability 
    [~, Ypreds] = max(test_prob, [], 1);
    Ypreds = Ypreds - 1;
    

end
