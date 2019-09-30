%
%
function task1_5(X, Ks)
% Input:
%  X  : M-by-D data matrix (double)
%  Ks : 1-by-L vector (integer) of the numbers of nearest neighbours
    tic
    D = size(X,2);
    L = size(Ks,2);
    maxIter = 500;
    
    
    for i = 1:L
        k = Ks(1,i); % Retrieve each k
        initialCentres = zeros(k,D);
        
        % Get the first k samples in X as initial cluster centres
        for c = 1:k
            initialCentres(c,:) = X(c,:);
            
        end
            
        [C, idx, SSE] = my_kMeansClustering(X, k, initialCentres, maxIter);
        
        
    end
    
    %Plotting the graph of SSE for each k
    a = (0:(size(SSE,1)-1));
    plot(a,SSE)
    title('SSE for k=20')
    xlabel('Iteration number')
    ylabel('SSE')
    xticks(0:20:100)
    xlim([0 100])
        
    toc
end
