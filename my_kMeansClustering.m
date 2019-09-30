%
function [C, idx, SSE] = my_kMeansClustering(X, k, initialCentres, maxIter)
% Input
%   X : N-by-D matrix (double) of input sample data
%   k : scalar (integer) - the number of clusters
%   initialCentres : k-by-D matrix (double) of initial cluster centres
%   maxIter  : scalar (integer) - the maximum number of iterations
% Output
%   C   : k-by-D matrix (double) of cluster centres
%   idx : N-by-1 vector (integer) of cluster index table
%   SSE : (L+1)-by-1 vector (double) of sum-squared-errors

  %% If 'maxIter' argument is not given, we set by default to 500
  
  if nargin < 4
    maxIter = 500;
  end
  
 
  L = 0; % Number of iterations done
  N = size(X,1);
  C = initialCentres; % Cluster centre matrix to be updated in each iteration
  idx = zeros(N,1);
  SSE = zeros(L+1,1); % Sum of squared distances of each point from its cluster centre, for each iteration
  D = zeros(k,N); % Matrix to store sq distances between cluster centres and samples
  
  M = zeros(1,N); % Mean values of each sample

  
  % When k=1, all the data points are assigned to that cluster centre and
  % thus, it is not necessary to recalculate the mean for the cluster
  % centres
  
  
  if k == 1
      
      C = initialCentres;
      idx = ones(N,1); % All data points assigned to 1
      SSE = zeros(maxIter,1); 
      
      return
      
  end
  
  
  % Calculate distances and SSE for given initialCentres
  % Compute distance for each sample to each cluster centre i
  
  for i = 1:k 
      D(i,:) = sum(bsxfun(@minus, X, C(i,:)).^2, 2)'; % Distances
      %D(i,:) = MySqDist(X,C(i,:));
  end


  % Assign data to clusters
  % Ds are the actual distances and idx the cluster assignments
  [Ds, centre] = min(D);
  SSE(1,1) = sum(Ds);
  idx = centre';

  % Update cluster centres
  for b = 1:k
      if sum(centre == b) == 0
          fprintf('k-means: cluster %d is empty', b);
      else
          C(b,:) = MyMean(X(centre == b,:));
      end
  end


  % Iterate until the assignments to cluster centres do not change anymore
  for i = 1:maxIter
      previdx = idx;

      for b = 1:k
          D(b,:) = sum(bsxfun(@minus, X, C(b,:)).^2, 2)'; % Distance between each sample and cluster centre
          %D(b,:) = MySqDist(X,C(b,:));
      end

      % Assign data to clusters
      % Ds are the actual distances and idx the cluster assignments
      [Ds, centre] = min(D); % Find min distance for each observation
      SSE(i+1,:) = sum(Ds);
      idx = centre';
      % Update cluster centres
      for b = 1:k
          % Check the number of samples assigned to this cluster
          if sum(centre == b) == 0
              fprintf('k-means: cluster %d is empty', b);
          else
              C(b,:) = MyMean(X(centre == b,:));
          end
      end
      L = L+1;

      % Stop iterating when the cluster centre assignments for the samples
      % do not change
      if isequal(idx,previdx) == 1
          break
      end   

  end
  
end
