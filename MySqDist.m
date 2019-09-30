function sqd = MySqDist(X, Y)
% SQUARE DISTANCE
% Computes square distance between the two matrices X and Y

    [M, ~] = size(X);
    [N, ~] = size(Y);
    XX = zeros(M,1);
    YY = zeros(N,1);
    
    for m = 1:M
        XX(m,:) = X(m,:) * X(m,:)';
    end
    
    for m = 1:N
        YY(m,:) = Y(m,:) * Y(m,:)';
    end
    
    sqd = single(repmat(XX,1,N) - 2 *X*Y' + (repmat(YY,1,M))');
    %Ds = single(repmat(sum(U.*U,2),1,N) - (2*U*V') + repmat((sum(V.*V,2))',1,M));
    
end

