%
%
function task1_6(MAT_ClusterCentres)
% Input:
%  MAT_ClusterCentres : file name of the MAT file that contains cluster centres C.
%       
% 
    % Load data
    L = load(MAT_ClusterCentres);
    C = L.C;
    Cs = zeros(28,28,1,size(C,1));
    
    % Take cluster centres
    for i = 1:size(C,1)
        Cs(:,:,:,i) = (reshape(C(i,:), 28, 28)');

    end

    % Display images of cluster centres
    montage(Cs);
  
end
