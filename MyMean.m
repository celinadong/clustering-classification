function my_mean = MyMean(A)
% MEAN
% Mean of a matrix

    rows = size(A,1);
    columns = size(A,2);
    my_mean = zeros(1,columns);


    my_mean = sum(A,1)./rows;

end

