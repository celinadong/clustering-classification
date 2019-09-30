function my_cov = MyCov(A,M)
% COVARIANCE MATRIX
% Compute the covariance matrix given a matrix A and mean (row) vector M

    rows = size(A,1);

    N = bsxfun(@minus, A, M);
    my_cov = (N' * N)/(rows);

end

