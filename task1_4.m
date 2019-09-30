%
%
function task1_4(EVecs)
% Input:
%  Evecs : the same format as in comp_pca.m
%

    %colormap 'gray';

    PA = []; % Matrix to store the first ten principal axes
    R = zeros(28,28,12,1); % 12 to fill in the last two gaps with the same colour gray

    for v = 1:10
        PA(:,v) = EVecs(:,v);
    end

    for k = 1:10
        R(:,:,k,1) = (reshape(PA(:,k)*255.0, 28, 28)');
    end


    I = mat2gray(R);
    montage(I);


end
