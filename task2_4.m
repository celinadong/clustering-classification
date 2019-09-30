%
%
function [Corrs] = task2_4(Xtrain, Ytrain)
% Input:
%  Xtrain : M-by-D data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for X
% Output:
%  Corrs  : (K+1)-by-1 vector (double) of correlation $r_{12}$ 
%           for each class k = 1,...,K, and the last element holds the
%           correlation for the whole data, i.e. Xtrain.

    % Use eigenvectors to transform 784 dimensions data X into 2 dimensional space
    mean = MyMean(Xtrain);
    C = MyCov(Xtrain,mean); % Calculate covariance matrix
    [EVecs, ~] = comp_pca(C);
    %EVecs = importdata('task1_3_evecs.mat');
    PC_X = Xtrain * EVecs(:,1:2); % 2D X matrix (Mx2)
    
    % Compute mean values for each class in the 2 dimensional space
    S = zeros(11, size(PC_X,2)); % Sums of the vectors for each class
    C = zeros(11,1); % Counter
    C(11) = size(Ytrain,1); % Total number of samples
    M = zeros(11, size(PC_X,2)); % Mean vectors for each class and total for all classes
    R = zeros(28,28,1,11); % Reshaped mean vector values

    % Compute the sums of vectors for each sample
    for i = 1:length(Ytrain)
        S(Ytrain(i)+1,:) = S(Ytrain(i)+1,:) + PC_X(i,:);
        S(11,:) = S(11,:) + PC_X(i,:);
        C(Ytrain(i)+1) = C(Ytrain(i)+1) + 1; % Number of samples for class Y(i)
    end

    % Compute the means of each vector class
    for j = 1:10
        M(j,:) = S(j,:)./C(j);   
    end
    % Mean vector of all samples
    M(11,:) = S(11,:)./C(11);
    
    
    
    
    % Initialise Corrs matrix
    Corrs = zeros(11,1);
     
    
    
    
    % Correlation coefficient for k = 1..10
    for i = 1:10
        
        m_i = M(i,:); % Mean vector for ith class
        covMatrix = MyCov(PC_X(Ytrain == i-1,:),m_i); % Filter by class
        s11 = covMatrix(1,1);
        s12 = covMatrix(1,2);
        s22 = covMatrix(2,2);
        Corrs(i,1) = s12 / sqrt(s11 * s22);     
        
    end

    % Correlation coefficient for k = 11
    % Compute the covariance matrix for Xtrain data
    m11 = M(11,:);
    covMatrix = MyCov(PC_X, m11);
    s12 = covMatrix(1,2);
    s11 = covMatrix(1,1);
    s22 = covMatrix(2,2);
    Corrs(11,1) = s12 / sqrt(s11 * s22);
    save('task2_4_corrs.mat', 'Corrs');
   
end
