%
%
function M = task1_2(X, Y)

% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector (unit8)
% Output:
%  M : (K+1)-by-D mean vector matrix (double)
%      Note that M(K+1,:) is the mean vector of X.

    colormap 'gray';

    S = zeros(11, size(X,2)); % Sums of the vectors for each class
    C = zeros(11,1); % Counter
    C(11) = size(Y,1); % Total number of samples
    M = zeros(11, size(X,2)); % Mean vectors for each class and total for all classes
    R = zeros(28,28,1,11); % Reshaped mean vector values

    % Compute the sums of vectors for each sample
    for i = 1:length(Y)
        S(Y(i)+1,:) = S(Y(i)+1,:) + X(i,:);
        S(11,:) = S(11,:) + X(i,:);
        C(Y(i)+1) = C(Y(i)+1) + 1; % Number of samples for class Y(i)
    end

    % Compute the means of each vector class
    for j = 1:10
        M(j,:) = S(j,:)./C(j);   
    end
    % Mean vector of all samples
    M(11,:) = S(11,:)./C(11);

    
    for k = 1:11
        R(:,:,1,k) = (reshape(M(k,:)*255.0, 28, 28)');
    end

    montage(R, 'Size', [3 4], 'DisplayRange', [0 255]);
    

end