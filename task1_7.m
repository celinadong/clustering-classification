%
%
function Dmap = task1_7(MAT_ClusterCentres, MAT_M, MAT_evecs, MAT_evals, posVec, nbins)
% Input:
%  MAT_ClusterCentres: MAT filename of cluster centre matrix
%  MAT_M     : MAT filename of mean vectors of (K+1)-by-D, where K is
%              the number of classes (which is 10 for the MNIST data)
%  MAT_evecs : MAT filename of eigenvector matrix of D-by-D
%  MAT_evals : MAT filename of eigenvalue vector of D-by-1
%  posVec    : 1-by-D vector (double) to specify the position of the plane
%  nbins     : scalar (integer) to specify the number of bins for each PCA axis
% Output
%  Dmap  : nbins-by-nbins matrix (uint8) - each element represents
%	   the cluster number that the point belongs to.

    % Load the data
    C = load(MAT_ClusterCentres);
    C = C.C; % Load the cluster centre matrix;
    M = load(MAT_M);
    M = M.M; % Load mean vectors
    EVectors = load(MAT_evecs);
    EVectors = EVectors.EVecs; % Load eigenvectors
    EValues = load(MAT_evals);
    EValues = EValues.EVals; % Load eigenvalues

    
    
    % Calculate the values to show on the 2D plane
    % Retrieve the first two principal components for the 2D plane axes
    PC1 = EValues(1,1);
    PC2 = EValues(2,1);
    % Get the first two eigenvectors
    EVec1 = EVectors(:,1);
    EVec2 = EVectors(:,2);
    % Retrieve the mean of all the digits
    mean = M(11,:);
    
    % Calculate the standard deviation
    sd1 = sqrt(PC1); 
    sd2 = sqrt(PC2);
    
    m1 = (mean - posVec) * EVec1; 
    m2 = (mean - posVec) * EVec2;
    

    
    % Colormap we will use to colour each classes.
    cmap = [0.80369089, 0.61814689, 0.46674357;
    0.81411766, 0.58274512, 0.54901962;
    0.58339103, 0.62000771, 0.79337179;
    0.83529413, 0.5584314 , 0.77098041;
    0.77493273, 0.69831605, 0.54108421;
    0.72078433, 0.84784315, 0.30039217;
    0.96988851, 0.85064207, 0.19683199;
    0.93882353, 0.80156864, 0.4219608 ;
    0.83652442, 0.74771243, 0.61853136;
    0.7019608 , 0.7019608 , 0.7019608];


    % Plot the Dmap axes
    Xplot = linspace((m1-5*sd1),(m1+5*sd1),nbins)';
    Yplot = linspace((m2-5*sd2),(m2+5*sd2),nbins)';
    
    % Obtain the grid vectors for the two dimensions
    [Xv,Yv] = meshgrid(Xplot, Yplot);
    

    D = size(M,2);
    A = size(Xv(:),1);
    gridX = zeros(A,D); % Grid Y padded with zeros
    % Fill in the first two columns with the actual values
    gridX(:,1) = Xv(:);
    gridX(:,2) = Yv(:);
    gridY = gridX*inv(EVectors)+posVec;
    
    distances = zeros(size(C,1),A);
    Dmap = zeros(nbins*nbins,1);

    
    % Compute the distances and assign closest cluster centre to each point
    distances = MySqDist(C,gridY);
    %{
    for b = 1:size(C,1)
        distances(b,:) = sum(bsxfun(@minus, gridY, C(b,:)).^2, 2)';
    end
    %}
    
    if size(C,1) ~= 1
        
        [~, idx] = min(distances);
        Dmap = idx';
        Dmap = reshape(Dmap, nbins, nbins);
        
    else
        
        idx = ones(1,size(Dmap,1)); % When k=1, all data points are labeled 1
        Dmap = idx';
        Dmap = reshape(Dmap, nbins, nbins);

    end
    
    % k=1 all the graph is the same colour
    figure;
    % Draw the decision boundaries
    [~,h] = contourf(Xplot(:), Yplot(:), Dmap);
    set(h,'LineColor','none');
    colormap(cmap);
    hold on;

			  
end
