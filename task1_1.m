%
%
function task1_1(X, Y)
% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector (unit8)       
    colormap 'gray';
    
    C = cell(1,1,1,10); % Cell to store the data for each digit
    
    % Index for each class
    for k = 0:9
        samples = 1;
        % Iterate through the data until 10 data samples for label k have
        % been retrieved
        for i = 1:length(Y)
            if samples > 10
                break
            end
            if Y(i) == k
                C{1,samples} = (reshape(X(i,:), 28, 28)');
                samples = samples+1;
            end
        end
        
        figure
        montage(C);
    end
    
end