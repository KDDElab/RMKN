function [obj] = objectfunction(tra,yyK,graph_w)
[num_data,~] = size(tra);
obj = 0;
for i = 1:num_data
    for j = 1:num_data
        obj = obj + (yyK(i,i)-2*yyK(i,j)+yyK(j,j))*graph_w(i,j);
    end
end
