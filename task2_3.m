%
%
function task2_3(X, Y)
% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector for X (unit8)
    
    % Use eigenvectors to transform 784 dimensions data X into 2 dimensional space
    mean = MyMean(X);
    C = MyCov(X,mean); % Calculate covariance matrix
    [EVecs, ~] = comp_pca(C);
    %EVecs = importdata('task1_3_evecs.mat');
    PC_X = X * EVecs(:,1:2); % 2D X matrix (Mx2)
    
    % Compute mean values for each class in the 2 dimensional space
    S = zeros(11, size(PC_X,2)); % Sums of the vectors for each class
    C = zeros(11,1); % Counter
    C(11) = size(Y,1); % Total number of samples
    M = zeros(11, size(PC_X,2)); % Mean vectors for each class and total for all classes
    R = zeros(28,28,1,11); % Reshaped mean vector values

    % Compute the sums of vectors for each sample
    for i = 1:length(Y)
        S(Y(i)+1,:) = S(Y(i)+1,:) + PC_X(i,:);
        S(11,:) = S(11,:) + PC_X(i,:);
        C(Y(i)+1) = C(Y(i)+1) + 1; % Number of samples for class Y(i)
    end

    % Compute the means of each vector class
    for j = 1:10
        M(j,:) = S(j,:)./C(j);   
    end
    % Mean vector of all samples
    M(11,:) = S(11,:)./C(11);

    
    
    
    
    % Compute Gaussian distributions
    for i = 1:10
        
        m_i = M(i,:); % Mean row vector for the ith class
        mat = PC_X(Y == i-1, :);
        covM = MyCov(mat, m_i);
        
        % Eigenvalues of covM represent spread in the direction of the
        % eigenvectors
        [EVc, EVl] = eig(covM);
        t = linspace(0, 2 * pi);
        a = (EVc * sqrt(EVl)) * [cos(t(:))'; sin(t(:))'];
        
        % Plot contour of the distribution of each class
        plot(a(1,:) + m_i(1), a(2,:) + m_i(2));
        text(M(i,1),M(i,2), num2str(i-1)); % Number of the class on the centre
        hold on;
        
    end
    
    xlabel('1st principal component');
    ylabel('2nd principal component');
    xlim([-1 9]);
    ylim([-4 4]);

end
