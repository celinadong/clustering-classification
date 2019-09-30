%
%
function Dmap = task2_6(X, Y, epsilon, MAT_evecs, MAT_evals, posVec, nbins)
% Input:
%  X        : M-by-D data matrix (double)
%  Y        : M-by-1 label vector (uint8)
%  epsilon  : scalar (double) for covariance matrix regularisation
%  MAT_evecs : MAT filename of eigenvector matrix of D-by-D
%  MAT_evals : MAT filename of eigenvalue vector of D-by-1
%  posVec   : 1-by-D vector (double) to specity the position of the plane
%  nbins     : scalar (integer) denoting the number of bins for each PCA axis
% Output
%  Dmap  : nbins-by-nbins matrix (uint8) - each element represents
%	   the cluster number that the point belongs to.

    % Load the data
    EVectors = load(MAT_evecs);
    EVectors = EVectors.EVecs; % Load eigenvectors
    EValues = load(MAT_evals);
    EValues = EValues.EVals; % Load eigenvalues
    
    
    % Retrieve the first two principal components for the 2D plane axes
    PC1 = EValues(1,1);
    PC2 = EValues(2,1);
    
    % Get the first two eigenvectors, since it is a 2D PCA plane
    EVec1 = EVectors(:,1);
    EVec2 = EVectors(:,2);
    
    % Calculate the mean vector of all the data
    mean = MyMean(X); 
    
    % Calculate the standard deviation
    sd1 = sqrt(PC1); 
    sd2 = sqrt(PC2);
    
    m1 = (mean - posVec) * EVec1; 
    m2 = (mean - posVec) * EVec2;
    

    
    % Colormap we will use to coXtrainlour each classes.
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



    Xplot = linspace((m1-5*sd1),(m1+5*sd1),nbins)';
    Yplot = linspace((m2-5*sd2),(m2+5*sd2),nbins)';
    
    % Obtain the grid vectors for the two dimensions
    [Xv,Yv] = meshgrid(Xplot, Yplot);
    

    D = size(mean,2);
    A = size(Xv(:),1);
    gridX = zeros(A,D); % Grid Y padded with zeros
    % Fill in the first two columns with the actual values
    gridX(:,1) = Xv(:);
    gridX(:,2) = Yv(:);
    %gridY = zeros(A,D);
    gridY = gridX*inv(EVectors)+posVec;
    

    Dmap = zeros(nbins*nbins,1);
        
    
    % Gaussian classification
    [Ypreds, ~, ~] = run_gaussian_classifiers(X, Y, gridY, epsilon);

    Dmap = Ypreds;

    Dmap = reshape(Dmap, nbins, nbins);


    figure;
    
    % This function will draw the decision boundaries
    [~,h] = contourf(Xplot(:), Yplot(:), Dmap);
    set(h,'LineColor','none');
    colormap(cmap);
    hold on;			  



end
