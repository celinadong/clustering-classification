function [EVecs, EVals] = comp_pca(X)
% Input: 
%   X:  N x D matrix (double)
% Output: 
%   EVecs: D-by-D matrix (double) contains all eigenvectors as columns
%       NB: follow the Task 1.3 specifications on eigenvectors.
%   EVals:
%       Eigenvalues in descending order, D x 1 vector (double)
%   (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)
  %% TO-DO
  
    [PC, V] = eig(X); % Each eigenvector is a unit vector
    
    V = diag(V); % Take the diagonal of matrix V as vector
    % Sort eigenvalues in descending order
    [~,idx] = sort(V, 1, 'descend');
    V = V(idx);
    PC = PC(:,idx);
    
    % If first element is negative, get opposite direction eigenvecto
    for i = 1:size(V,2)
        if V(1,i) < 0
            V(:,i) = V(:,i).*(-1);
        end
    end
    
    EVecs = PC;
    EVals = V;
 

end

